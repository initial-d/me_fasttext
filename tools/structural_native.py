"""Original DATrie prefix/suffix planner and mark/compact native export.

Optional source-build module: bundled trie archives require Linux x86-64.
Key lookup always returns original numeric IDs; row forwarding is independent.
Joint sharing changes weights. Only layout / exact sharing preserve source bytes.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from .native_row_store import pack_native

MISSING = (1 << 64) - 1


def _keys(values, *, allow_empty=False):
    encoded = []
    for key in values:
        if not isinstance(key, str) or "\0" in key or (not allow_empty and not key):
            raise ValueError("keys must be nonempty UTF-8 strings without embedded NUL")
        encoded.append(key.encode("utf-8"))
    return (ctypes.c_char_p * len(encoded))(*encoded)


def _library(path):
    path = path or os.environ.get("MEFT_STRUCTURAL_LIBRARY")
    if not path:
        raise ValueError("pass the optional Linux DATrie library or set MEFT_STRUCTURAL_LIBRARY")
    lib = ctypes.CDLL(str(Path(path).resolve()))
    ptr = ctypes.c_void_p
    size = ctypes.c_size_t
    u64 = ctypes.POINTER(ctypes.c_uint64)
    strings = ctypes.POINTER(ctypes.c_char_p)
    lib.meft_structural_build.argtypes = [strings, u64, size, ptr, size, size, ctypes.c_int,
                                          ctypes.c_double, ctypes.c_double, ptr, size]
    lib.meft_structural_build.restype = ptr
    lib.meft_structural_close.argtypes = [ptr]
    lib.meft_structural_close.restype = None
    lib.meft_structural_maps.argtypes = [ptr, u64, u64, size, ptr, size]
    lib.meft_structural_maps.restype = ctypes.c_int
    lib.meft_structural_lookup.argtypes = [ptr, strings, size, u64, ctypes.c_int, ptr, size]
    lib.meft_structural_lookup.restype = ctypes.c_int
    lib.meft_structural_save_index.argtypes = [ptr, ctypes.c_char_p, ctypes.c_int, ptr, size]
    lib.meft_structural_save_index.restype = ctypes.c_int
    lib.meft_structural_open_index.argtypes = [ctypes.c_char_p, ptr, size]
    lib.meft_structural_open_index.restype = ptr
    return lib


class StructuralIndex:
    """Single-worker original-ID key lookup; open only trusted produced indices."""
    def __init__(self, path, library=None):
        self.lib = _library(library)
        error = ctypes.create_string_buffer(512)
        self.handle = self.lib.meft_structural_open_index(os.fsencode(path), error, len(error))
        if not self.handle:
            raise ValueError(error.value.decode())

    def lookup_into(self, query, output, *, index="datrie"):
        """Accept pre-encoded ctypes query/output arrays to avoid Python key loops."""
        if not self.handle or index not in ("flat", "datrie", "hash") or len(output) != len(query):
            raise ValueError("closed index, invalid index policy or query capacity")
        error = ctypes.create_string_buffer(512)
        if self.lib.meft_structural_lookup(self.handle, query, len(query), output,
                                           ("flat", "datrie", "hash").index(index), error, len(error)):
            raise ValueError(error.value.decode())

    def lookup(self, keys, *, index="datrie"):
        query = _keys(keys, allow_empty=True)
        output = (ctypes.c_uint64 * len(query))()
        self.lookup_into(query, output, index=index)
        return [None if value == MISSING else value for value in output]

    def close(self):
        if self.handle:
            self.lib.meft_structural_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class StructuralPlan(StructuralIndex):
    """Use actual production callbacks on float32 rows and real UTF-8 n-gram keys.

    keys maps original row IDs to unique strings. Unknown/padding rows can be
    omitted and remain identity-mapped. Hash bucket aliases need a separate
    provenance/structural certificate; this API rejects duplicate key-bound IDs.
    """
    def __init__(self, rows, dim, keys, library=None, *, mode="joint",
                 cosine=.99, norm_ratio=.99):
        if type(dim) is not int or not 0 < dim <= (1 << 31) - 1:
            raise ValueError("positive float32 dimension required")
        rows = bytes(rows)
        nrows, partial = divmod(len(rows), dim * 4)
        if not nrows or partial:
            raise ValueError("nonempty complete float32 rows required")
        if mode not in ("layout", "exact", "joint") or not 0 < cosine <= 1 or not 0 < norm_ratio <= 1:
            raise ValueError("invalid sharing mode or thresholds")
        original_ids = list(keys)
        if any(type(i) is not int or not 0 <= i < nrows for i in original_ids):
            raise ValueError("key-bound original ID out of range")
        query = _keys(keys.values())
        ids = (ctypes.c_uint64 * len(original_ids))(*original_ids)
        data = (ctypes.c_float * (nrows * dim)).from_buffer_copy(rows)
        self.lib = _library(library)
        error = ctypes.create_string_buffer(512)
        self.handle = self.lib.meft_structural_build(query, ids, len(query), data, nrows, dim,
            ("layout", "exact", "joint").index(mode), cosine, norm_ratio, error, len(error))
        if not self.handle:
            raise ValueError(error.value.decode())
        self.dim, self.nrows = dim, nrows
        self.mode, self.cosine, self.norm_ratio = mode, cosine, norm_ratio
        self.source_sha256 = hashlib.sha256(rows).hexdigest()
        representatives, order = (ctypes.c_uint64 * nrows)(), (ctypes.c_uint64 * nrows)()
        if self.lib.meft_structural_maps(self.handle, representatives, order, nrows, error, len(error)):
            self.close()
            raise ValueError(error.value.decode())
        self.representatives, self.row_order = list(representatives), list(order)
        width = dim * 4
        digest = hashlib.sha256()
        self.original_bytes_preserved = True
        for i, r in enumerate(self.representatives):
            row = rows[r*width:(r+1)*width]
            digest.update(row)
            self.original_bytes_preserved &= rows[i*width:(i+1)*width] == row
        self.reconstructed_sha256 = digest.hexdigest()

    def save_index(self, path, *, index="datrie"):
        if not self.handle or index not in ("flat", "datrie"):
            raise ValueError("closed plan or invalid index policy")
        error = ctypes.create_string_buffer(512)
        if self.lib.meft_structural_save_index(self.handle, os.fsencode(path), int(index == "datrie"),
                                                error, len(error)):
            raise ValueError(error.value.decode())

    def export(self, source, directory, *, block_rows=256, predictor="xor", index="datrie"):
        """Publish to a NEW owned directory; failed exports remove that directory.

        Source must match planned bytes. Existing reader/cache ABI is unchanged.
        Index returns original IDs; the table's dense forwarding map resolves
        surviving rows. Numeric/hash-only models may use table.men without index.
        """
        if not self.handle:
            raise ValueError("plan is closed")
        source, directory = Path(source), Path(directory)
        if hashlib.sha256(source.read_bytes()).hexdigest() != self.source_sha256:
            raise ValueError("source bytes differ from structural plan")
        directory.mkdir(parents=True, exist_ok=False)
        try:
            index_path = directory / ("keys.das" if index == "datrie" else "keys.flat")
            self.save_index(index_path, index=index)
            metadata = pack_native(source, directory / "table.men", self.dim * 4, block_rows,
                deduplicate=False, predictor=predictor, row_order=self.row_order,
                representatives=self.representatives)
            if hashlib.sha256(source.read_bytes()).hexdigest() != self.source_sha256:
                raise ValueError("source changed during structural export")
            metadata.update({"structural_mode": self.mode, "index": index,
                "index_bytes": index_path.stat().st_size, "source_sha256": self.source_sha256,
                "representatives_sha256": hashlib.sha256(bytes(
                    (ctypes.c_uint64 * self.nrows)(*self.representatives))).hexdigest(),
                "original_bytes_preserved": self.original_bytes_preserved,
                "minimum_cosine": self.cosine if self.mode == "joint" else None,
                "minimum_norm_ratio": self.norm_ratio if self.mode == "joint" else None,
                "contract": "original IDs retained; decoder exact to representative bytes; joint sharing may change weights"})
            (directory / "structural.json").write_text(json.dumps(metadata, indent=2) + "\n")
            # Standard frozen global-ID PyTorch adapter metadata. The C table
            # remains independent of PyTorch; no expanded table is staged.
            embedding = {"schema_version": 1, "table_file": "table.men",
                "shape": [self.nrows, self.dim], "dtype": "torch.float32", "byteorder": sys.byteorder,
                "head_sizes": [self.nrows], "offsets": [0], "backend": "native", "predictor": predictor,
                "weight_sha256": self.reconstructed_sha256, "source_weight_sha256": self.source_sha256,
                "artifact_sha256": hashlib.sha256((directory / "table.men").read_bytes()).hexdigest(),
                "input_bytes": self.nrows * self.dim * 4, "artifact_bytes": metadata["artifact_bytes"],
                "unique_rows": metadata["unique_rows"], "row_map_bytes": metadata["row_map_bytes"],
                "block_rows": block_rows, "approximate": not self.original_bytes_preserved,
                "contract": metadata["contract"]}
            (directory / "embedding.json").write_text(json.dumps(embedding, indent=2) + "\n")
            return metadata
        except BaseException:
            shutil.rmtree(directory)
            raise
