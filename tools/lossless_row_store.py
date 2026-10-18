"""Experimental immutable row store: exact-byte deduplication and block lookup.

Rows are opaque fixed-width bytes. Original row IDs are preserved through a
forwarding table. No vector arithmetic, quantization or approximate merge is
performed. This is a Python reference, independent of the legacy .z exporter.
"""
import argparse
from collections import OrderedDict
import json
import mmap
from pathlib import Path
import struct
import zlib

MAGIC = b"MEFTROW1"
PREFIX = struct.Struct("<8sQ")
ROW_ID = struct.Struct("<Q")


def pack_rows(source, output, row_bytes, block_rows=256):
    """Pack a raw fixed-width table; offline builder keeps unique rows in RAM."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("source and output must differ")
    if row_bytes <= 0 or block_rows <= 0:
        raise ValueError("row_bytes and block_rows must be positive")
    if source.stat().st_size % row_bytes:
        raise ValueError("input size is not a multiple of row_bytes")
    unique, ids, seen = [], bytearray(), {}
    with source.open("rb") as handle:
        while True:
            row = handle.read(row_bytes)
            if not row:
                break
            if len(row) != row_bytes:
                raise ValueError("incomplete input row")
            index = seen.get(row)
            if index is None:
                index = len(unique)
                seen[row] = index
                unique.append(row)
            ids.extend(ROW_ID.pack(index))
    blocks, payload, offset = [], [], len(ids)
    for start in range(0, len(unique), block_rows):
        raw = b"".join(unique[start:start + block_rows])
        encoded = zlib.compress(raw)
        blocks.append({"offset": offset, "bytes": len(encoded), "raw_bytes": len(raw)})
        payload.append(encoded)
        offset += len(encoded)
    header = {
        "version": 1, "codec": "zlib", "row_bytes": row_bytes,
        "n_rows": len(ids) // ROW_ID.size, "unique_rows": len(unique),
        "block_rows": block_rows, "blocks": blocks,
        "contract": "original row IDs resolve to identical input bytes",
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with output.open("wb") as handle:
        handle.write(PREFIX.pack(MAGIC, len(encoded_header)))
        handle.write(encoded_header)
        handle.write(ids)
        for block in payload:
            handle.write(block)
    return {**header, "input_bytes": source.stat().st_size,
            "artifact_bytes": output.stat().st_size, "row_map_bytes": len(ids)}


class MappedRows:
    """Random row access through mmap, with a bounded decompressed-block cache."""

    def __init__(self, path, cache_blocks=4):
        if cache_blocks < 0:
            raise ValueError("cache_blocks must be non-negative")
        self.cache_blocks = cache_blocks
        self.cache = OrderedDict()
        self.file = Path(path).open("rb")
        self.mapping = None
        try:
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            magic, size = PREFIX.unpack_from(self.mapping)
            if magic != MAGIC or size > len(self.mapping) - PREFIX.size:
                raise ValueError("invalid row-store header")
            self.header = json.loads(self.mapping[PREFIX.size:PREFIX.size + size])
            self.data_offset = PREFIX.size + size
            if self.header["version"] != 1 or self.header["codec"] != "zlib":
                raise ValueError("unsupported row-store version or codec")
            if self.header["n_rows"] * ROW_ID.size > len(self.mapping) - self.data_offset:
                raise ValueError("truncated row map")
        except Exception:
            self.close()
            raise

    def _locate(self, row_id):
        if not isinstance(row_id, int) or not 0 <= row_id < self.header["n_rows"]:
            raise IndexError("row ID outside original table")
        unique_id, = ROW_ID.unpack_from(self.mapping, self.data_offset + row_id * ROW_ID.size)
        if unique_id >= self.header["unique_rows"]:
            raise ValueError("invalid forwarding row ID")
        return divmod(unique_id, self.header["block_rows"])

    def _block(self, block_id):
        raw = self.cache.pop(block_id, None)
        if raw is None:
            block = self.header["blocks"][block_id]
            start = self.data_offset + block["offset"]
            encoded = self.mapping[start:start + block["bytes"]]
            if len(encoded) != block["bytes"]:
                raise ValueError("truncated compressed block")
            raw = zlib.decompress(encoded)
            if len(raw) != block["raw_bytes"]:
                raise ValueError("invalid decompressed block length")
        if self.cache_blocks:
            self.cache[block_id] = raw
            while len(self.cache) > self.cache_blocks:
                self.cache.popitem(last=False)
        return raw

    def row(self, row_id):
        block_id, within = self._locate(row_id)
        raw = self._block(block_id)
        width = self.header["row_bytes"]
        return raw[within * width:(within + 1) * width]

    def rows(self, row_ids):
        """Coalesce each block once per batch and restore original ID order.

        Working memory is output bytes, O(batch size) locations and one block,
        plus the configured persistent block cache.
        """
        groups = {}
        count = 0
        for position, row_id in enumerate(row_ids):
            block, within = self._locate(row_id)
            groups.setdefault(block, []).append((position, within))
            count += 1
        width = self.header["row_bytes"]
        output = bytearray(count * width)
        for block, locations in groups.items():
            raw = self._block(block)
            for position, within in locations:
                output[position * width:(position + 1) * width] = raw[within * width:(within + 1) * width]
        return bytes(output)

    def close(self):
        self.cache.clear()
        if self.mapping is not None:
            self.mapping.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw fixed-width row bytes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-bytes", type=int, required=True)
    parser.add_argument("--block-rows", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(pack_rows(args.input, args.output, args.row_bytes, args.block_rows), indent=2))


if __name__ == "__main__":
    main()
