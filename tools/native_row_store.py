"""Native C ABI bindings and exporter for the versioned MEFTNAT1 format."""
import ctypes
import os
from pathlib import Path
import struct
import tempfile
import zlib

HEADER = struct.Struct("<8s6Q")
DESC = struct.Struct("<4Q")
ROW = struct.Struct("<Q")


def pack_native(source, output, row_bytes, block_rows=256, deduplicate=True,
                predictor="none", row_order=None):
    """Stream blocks into a spool; exact dedup index still grows with unique rows.

    deduplicate=False avoids the unbounded unique-row dictionary for large raw
    tables, while preserving the identity forwarding map. No numeric changes.
    Incompressible blocks are stored raw, avoiding codec expansion per block.
    predictor="xor" tries previous-row XOR + zlib as a third block candidate
    and writes format version 2. Version 1 remains the default. Optional
    row_order is a full permutation of original IDs; its validation uses O(n)
    memory and reading becomes nonsequential. Original query IDs stay unchanged.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or row_bytes <= 0 or block_rows <= 0:
        raise ValueError("invalid paths or block dimensions")
    nrows, remainder = divmod(source.stat().st_size, row_bytes)
    if remainder:
        raise ValueError("partial input row")
    if predictor not in ("none", "xor"):
        raise ValueError("predictor must be none or xor")
    if row_order is not None:
        row_order = list(row_order)
        if (len(row_order) != nrows or
                any(type(i) is not int or not 0 <= i < nrows for i in row_order) or
                len(set(row_order)) != nrows):
            raise ValueError("row_order must be a permutation of all original row IDs")
    unique, blocks, seen, pending = 0, [], {}, []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output.parent) as map_file, tempfile.TemporaryFile(dir=output.parent) as spool:
        def flush():
            raw = b"".join(pending)
            compressed = zlib.compress(raw)
            codec = int(len(compressed) < len(raw))
            encoded = compressed if codec else raw
            if predictor == "xor" and len(pending) > 1:
                # Python integers perform the same bytewise operation in C.
                # A little-endian shift places the previous row at the current
                # row's position, leaving the first row unchanged. All buffers
                # remain bounded by this block; no dtype/float conversion occurs.
                residual = (int.from_bytes(raw, "little") ^
                            (int.from_bytes(raw[:-row_bytes], "little") << (8 * row_bytes)))
                residual = residual.to_bytes(len(raw), "little")
                candidate = zlib.compress(residual)
                if len(candidate) < len(encoded):
                    encoded, codec = candidate, 2
            blocks.append((spool.tell(), len(encoded), len(raw), codec))
            spool.write(encoded)
            pending.clear()

        with source.open("rb") as stream:
            for original_id in range(nrows) if row_order is None else row_order:
                if row_order is not None:
                    stream.seek(original_id * row_bytes)
                row = stream.read(row_bytes)
                if len(row) != row_bytes:
                    raise ValueError("source changed or truncated while exporting")
                index = seen.get(row) if deduplicate else None
                if index is None:
                    index = unique
                    unique += 1
                    if deduplicate:
                        seen[row] = index
                    pending.append(row)
                    if len(pending) == block_rows:
                        flush()
                if row_order is not None:
                    map_file.seek(original_id * ROW.size)
                map_file.write(ROW.pack(index))
            if os.fstat(stream.fileno()).st_size != nrows * row_bytes:
                raise ValueError("source size changed while exporting")
        if pending:
            flush()
        payload_start = HEADER.size + nrows * ROW.size + len(blocks) * DESC.size
        with output.open("wb") as dest:
            version = 2 if predictor == "xor" else 1
            dest.write(HEADER.pack(b"MEFTNAT1", version, nrows, row_bytes, unique, block_rows, len(blocks)))
            map_file.seek(0)
            for part in iter(lambda: map_file.read(1024 * 1024), b""):
                dest.write(part)
            for offset, stored, raw_size, codec in blocks:
                dest.write(DESC.pack(payload_start + offset, stored, raw_size, codec))
            spool.seek(0)
            for part in iter(lambda: spool.read(1024 * 1024), b""):
                dest.write(part)
    return {"format": "MEFTNAT1", "format_version": version, "input_bytes": nrows * row_bytes,
            "artifact_bytes": output.stat().st_size, "unique_rows": unique,
            "row_map_bytes": nrows * ROW.size, "n_rows": nrows, "row_bytes": row_bytes,
            "block_rows": block_rows, "blocks": len(blocks),
            "raw_blocks": sum(codec == 0 for _, _, _, codec in blocks),
            "compressed_blocks": sum(codec != 0 for _, _, _, codec in blocks),
            "zlib_blocks": sum(codec == 1 for _, _, _, codec in blocks),
            "xor_blocks": sum(codec == 2 for _, _, _, codec in blocks),
            "deduplicate": deduplicate, "predictor": predictor,
            "reordered": row_order is not None}


class NativeRows:
    """Single-worker C handle. Library path is explicit or MEFT_ROWS_LIBRARY."""
    def __init__(self, path, library=None, cache_bytes=0):
        library = library or os.environ.get("MEFT_ROWS_LIBRARY")
        if not library:
            raise ValueError("set MEFT_ROWS_LIBRARY or pass a compiled library path")
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= ctypes.c_size_t(-1).value:
            raise ValueError("cache bytes must fit a nonnegative size_t")
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.meft_rows_open.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.meft_rows_open.restype = ctypes.c_void_p
        self.lib.meft_rows_close.argtypes = [ctypes.c_void_p]
        self.lib.meft_rows_close.restype = None
        for name in ("count", "width", "last_block_reads", "last_decompressions"):
            method = getattr(self.lib, "meft_rows_" + name)
            method.argtypes = [ctypes.c_void_p]
            method.restype = ctypes.c_uint64
        self.lib.meft_rows_gather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.meft_rows_gather.restype = ctypes.c_int
        error = ctypes.create_string_buffer(256)
        self.handle = self.lib.meft_rows_open(os.fsencode(path), error, len(error))
        if not self.handle:
            raise ValueError(error.value.decode())
        self.header = {"n_rows": self.lib.meft_rows_count(self.handle),
                       "row_bytes": self.lib.meft_rows_width(self.handle)}
        # Keep zero-budget loading compatible with older compiled libraries.
        if cache_bytes:
            try:
                self.set_cache_bytes(cache_bytes)
            except Exception:
                self.close()
                raise

    def set_cache_bytes(self, budget):
        if not self.handle:
            raise ValueError("row store is closed")
        if type(budget) is not int or not 0 <= budget <= ctypes.c_size_t(-1).value:
            raise ValueError("cache bytes must fit a nonnegative size_t")
        try:
            setter = self.lib.meft_rows_set_cache_bytes
        except AttributeError as error:
            raise ValueError("native cache requires rebuilding the row library") from error
        setter.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        setter.restype = ctypes.c_int
        if setter(self.handle, budget):
            raise ValueError("cannot configure native cache")

    def _counter(self, name):
        if not self.handle:
            raise ValueError("row store is closed")
        method = getattr(self.lib, "meft_rows_" + name)
        method.argtypes = [ctypes.c_void_p]
        method.restype = ctypes.c_uint64
        return method(self.handle)

    @property
    def cache_bytes(self):
        return self._counter("cache_bytes")

    @property
    def last_cache_hits(self):
        return self._counter("last_cache_hits")

    def gather_into(self, ids_address, count, output_address, output_bytes):
        if not self.handle:
            raise ValueError("row store is closed")
        error = ctypes.create_string_buffer(256)
        status = self.lib.meft_rows_gather(self.handle, ids_address, count, output_address,
                                          output_bytes, error, len(error))
        if status:
            raise ValueError(error.value.decode())

    def rows(self, ids):
        ids = list(ids)
        if any(type(i) is not int or not 0 <= i < self.header["n_rows"] for i in ids):
            raise IndexError("row ID out of bounds")
        query = (ctypes.c_uint64 * len(ids))(*ids)
        output = ctypes.create_string_buffer(len(ids) * self.header["row_bytes"])
        self.gather_into(ctypes.addressof(query), len(ids), ctypes.addressof(output), len(output))
        return output.raw

    def row(self, row_id):
        return self.rows([row_id])

    @property
    def last_block_reads(self):
        return self.lib.meft_rows_last_block_reads(self.handle)

    @property
    def last_decompressions(self):
        return self.lib.meft_rows_last_decompressions(self.handle)

    def close(self):
        if self.handle:
            self.lib.meft_rows_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
