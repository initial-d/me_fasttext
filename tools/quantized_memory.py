"""Frozen CPU row codecs for measured quantization + native compaction.

FP16 and symmetric per-row INT8 are approximate to the float32 source. Native
packing is exact to their encoded bytes. INT8 stores its float32 scale inside
each row, so forwarding/dedup never drops the scale. No trainable weight API.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .native_row_store import NativeRows, pack_native


def row_width(dim, encoding):
    if type(dim) is not int or dim < 1 or encoding not in ("fp32", "fp16", "int8"):
        raise ValueError("positive dimension and fp32/fp16/int8 encoding required")
    return {"fp32": dim * 4, "fp16": dim * 2, "int8": dim + 4}[encoding]


def encode_rows(rows, encoding):
    rows = np.asarray(rows)
    if rows.ndim != 2 or min(rows.shape) < 1 or rows.dtype.kind != "f":
        raise ValueError("nonempty floating-point matrix required")
    rows = np.ascontiguousarray(rows, dtype="<f4")
    if not np.isfinite(rows).all():
        raise ValueError("quantization requires finite float32 source rows")
    width = row_width(rows.shape[1], encoding)
    if encoding in ("fp32", "fp16"):
        with np.errstate(over="ignore"):
            values = rows.astype("<f4" if encoding == "fp32" else "<f2")
        if not np.isfinite(values).all():
            raise ValueError("source overflows the requested encoding")
        return values.view("u1").reshape(len(rows), width).copy()
    maximum = np.max(np.abs(rows.astype(np.float64)), axis=1)
    # Positive subnormals must not produce a zero scale through float32 rounding.
    scales = np.maximum(maximum / 127, np.finfo(np.float32).tiny).astype("<f4")
    # Rounding max_float32 / 127 upward makes 127 * scale overflow at decode.
    finite_ceiling = np.nextafter(np.float32(float(np.finfo(np.float32).max) / 127), np.float32(0))
    scales = np.minimum(scales, finite_ceiling)
    scales[maximum == 0] = 1
    values = np.clip(np.rint(rows.astype(np.float64) / scales[:, None]), -127, 127).astype("i1")
    return np.concatenate((values.view("u1"), scales.view("u1").reshape(-1, 4)), axis=1)


def decode_rows(encoded, dim, encoding):
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or encoded.dtype != np.uint8 or encoded.shape[1] != row_width(dim, encoding):
        raise ValueError("encoded byte matrix has the wrong row width")
    encoded = np.ascontiguousarray(encoded)
    if encoding in ("fp32", "fp16"):
        return encoded.view("<f4" if encoding == "fp32" else "<f2").reshape(-1, dim).astype("<f4", copy=True)
    scales = np.ascontiguousarray(encoded[:, dim:]).view("<f4").reshape(-1, 1)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("invalid INT8 scale")
    return encoded[:, :dim].view("i1").astype("<f4") * scales


def pack_quantized(rows, directory, encoding, *, native=True, representatives=None,
                   row_order=None, deduplicate=False, max_block_bytes=8192):
    """Publish a new research bundle; include codec metadata in measured bytes."""
    rows = np.ascontiguousarray(rows, dtype="<f4")
    encoded = encode_rows(rows, encoding)
    if not native and (representatives is not None or row_order is not None or deduplicate):
        raise ValueError("raw baseline must retain original row order without compaction")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        source = directory / "staging.rows"
        source.write_bytes(encoded.tobytes())
        table = directory / ("table.men" if native else "table.rows")
        if native:
            storage = pack_native(source, table, encoded.shape[1], deduplicate=deduplicate,
                                  predictor="xor", representatives=representatives,
                                  row_order=row_order, max_block_bytes=max_block_bytes)
            source.unlink()
        else:
            source.rename(table)
            storage = {"artifact_bytes": table.stat().st_size, "unique_rows": len(rows),
                       "row_map_bytes": 0, "deduplicate": False}
        mapping = np.arange(len(rows)) if representatives is None else np.asarray(representatives)
        expected = decode_rows(encoded[mapping], rows.shape[1], encoding)
        metadata = {"schema_version": 1, "shape": list(rows.shape), "encoding": encoding,
                    "byteorder": "little", "backend": "native" if native else "raw_mmap",
                    "table_file": table.name, "row_bytes": encoded.shape[1], "storage": storage,
                    "source_float32_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                    "decoded_float32_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                    "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                    "source_bytes_preserved": expected.tobytes() == rows.tobytes(),
                    "int8_scale_bytes_per_physical_row": 4 if encoding == "int8" else 0,
                    "rounding": "nearest-even; symmetric [-127,127] per-row scale" if encoding == "int8" else encoding,
                    "contract": "frozen original IDs; exact encoded representative bytes; quantization/sharing may change float32 rows"}
        (directory / "codec.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    except BaseException:
        shutil.rmtree(directory)
        raise


class QuantizedRows:
    """Single-worker CPU lookup; trusted locally produced bundles only."""
    def __init__(self, directory, library=None, *, cache_bytes=4194304):
        directory = Path(directory)
        self.metadata = json.loads((directory / "codec.json").read_text())
        m = self.metadata
        if (m["schema_version"] != 1 or m["byteorder"] != "little" or len(m["shape"]) != 2
                or min(m["shape"]) < 1 or m["row_bytes"] != row_width(m["shape"][1], m["encoding"])
                or m["table_file"] not in ("table.men", "table.rows")):
            raise ValueError("invalid local quantized bundle metadata")
        self.nrows, self.dim = m["shape"]
        self.encoding, self.width = m["encoding"], m["row_bytes"]
        self.table = None
        path = directory / m["table_file"]
        if m["backend"] == "native":
            self.table = NativeRows(path, library, cache_bytes=cache_bytes, cache_policy="rows")
            if self.table.header != {"n_rows": self.nrows, "row_bytes": self.width}:
                self.close()
                raise ValueError("native table shape disagrees with codec")
        elif m["backend"] == "raw_mmap":
            if path.stat().st_size != self.nrows * self.width:
                raise ValueError("raw table length disagrees with codec")
            self.table = np.memmap(path, mode="r", dtype="u1", shape=(self.nrows, self.width))
        else:
            raise ValueError("unknown quantized backend")

    def encoded_rows(self, ids):
        if self.table is None:
            raise ValueError("quantized rows are closed")
        ids = np.asarray(ids)
        if ids.ndim == 1 and ids.size == 0:
            return np.empty((0, self.width), dtype="u1")
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.nrows):
            raise ValueError("one-dimensional original row IDs out of range")
        if isinstance(self.table, NativeRows):
            ids = np.ascontiguousarray(ids, dtype="<u8")
            output = np.empty((len(ids), self.width), dtype="u1")
            self.table.gather_into(ids.ctypes.data, len(ids), output.ctypes.data, output.nbytes)
            return output
        return np.asarray(self.table[ids], dtype="u1")

    def gather(self, ids):
        return decode_rows(self.encoded_rows(ids), self.dim, self.encoding)

    def close(self):
        if isinstance(self.table, NativeRows):
            self.table.close()
        elif isinstance(self.table, np.memmap):
            self.table._mmap.close()
        self.table = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
