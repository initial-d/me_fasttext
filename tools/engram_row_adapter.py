"""CPU inference gather adapter for Engram-demo-shaped multi-head embeddings.

Requires PyTorch. Hashing, tokenization, offsets and row bytes stay unchanged.
This prototype packs an immutable table, not a full Engram checkpoint.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

try:
    from .lossless_row_store import MappedRows, pack_rows
except ImportError:  # direct script imports from tools/
    from lossless_row_store import MappedRows, pack_rows

DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.float16, torch.bfloat16)}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def head_offsets(head_sizes):
    if not head_sizes or any(type(n) is not int or n <= 0 for n in head_sizes):
        raise ValueError("head sizes must be positive integers")
    offsets, total = [], 0
    for size in head_sizes:
        offsets.append(total)
        total += size
    return offsets


def export_multihead(weight, head_sizes, output_dir, block_rows=256):
    """Preserve every row, including unseen/unused rows, and the source dtype."""
    offsets = head_offsets(head_sizes)
    if weight.layout != torch.strided or weight.ndim != 2 or weight.shape[1] <= 0:
        raise ValueError("weight must be a dense two-dimensional embedding table")
    if str(weight.dtype) not in DTYPES or weight.shape[0] != sum(head_sizes):
        raise ValueError("unsupported dtype or row count inconsistent with head sizes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / "table.mer"
    manifest = output_dir / "embedding.json"
    if table.exists() or manifest.exists():
        raise FileExistsError("adapter artifacts already exist; use a fresh output directory")
    frozen = weight.detach().to("cpu").contiguous()
    raw = frozen.view(torch.uint8).numpy().tobytes()
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        source = Path(tmp) / "table.raw"
        source.write_bytes(raw)
        packed = pack_rows(source, table, frozen.shape[1] * frozen.element_size(), block_rows)
    metadata = {
        "schema_version": 1, "table_file": table.name,
        "shape": list(frozen.shape), "dtype": str(frozen.dtype), "byteorder": sys.byteorder,
        "head_sizes": list(head_sizes), "offsets": offsets,
        "weight_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_sha256": file_sha256(table),
        "input_bytes": len(raw), "artifact_bytes": table.stat().st_size,
        "unique_rows": packed["unique_rows"], "row_map_bytes": packed["row_map_bytes"],
        "block_rows": block_rows,
        "contract": "unchanged local-head IDs plus original offsets; byte-exact CPU gather",
    }
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def export_demo_embedding(module, head_sizes, output_dir, block_rows=256):
    """Accept the official demo's .embedding.weight / .offsets interface."""
    expected = torch.tensor(head_offsets(head_sizes), dtype=torch.int64)
    if not torch.equal(module.offsets.detach().cpu(), expected):
        raise ValueError("source offsets differ from declared head sizes")
    return export_multihead(module.embedding.weight, head_sizes, output_dir, block_rows)


class MappedMultiHeadEmbedding(nn.Module):
    """Serving-only CPU replacement for MultiHeadEmbedding.forward's gather.

    Only the embedding table is replaced. Input [..., heads] and output
    [..., heads, embedding_dim] preserve the official gather's layout.
    """

    def __init__(self, output_dir, cache_blocks=4):
        super().__init__()
        output_dir = Path(output_dir)
        meta = json.loads((output_dir / "embedding.json").read_text())
        if (meta["schema_version"] != 1 or meta["dtype"] not in DTYPES
                or meta["byteorder"] != sys.byteorder or meta["table_file"] != "table.mer"):
            raise ValueError("unsupported adapter metadata")
        offsets = head_offsets(meta["head_sizes"])
        if meta["offsets"] != offsets or meta["shape"][0] != sum(meta["head_sizes"]):
            raise ValueError("inconsistent head metadata")
        path = output_dir / meta["table_file"]
        if file_sha256(path) != meta["artifact_sha256"]:
            raise ValueError("row-store checksum mismatch")
        self.dtype = DTYPES[meta["dtype"]]
        self.num_heads = len(offsets)
        self.embedding_dim = meta["shape"][1]
        self.metadata = meta
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64))
        self.register_buffer("head_sizes", torch.tensor(meta["head_sizes"], dtype=torch.int64))
        self.table = MappedRows(path, cache_blocks)
        width = self.embedding_dim * torch.empty((), dtype=self.dtype).element_size()
        if self.table.header["n_rows"] != meta["shape"][0] or self.table.header["row_bytes"] != width:
            self.table.close()
            raise ValueError("row-store shape differs from embedding metadata")
        self.training = False

    def train(self, mode=True):
        if mode:
            raise RuntimeError("mapped embeddings are immutable and support inference only")
        return super().train(False)

    def forward(self, input_ids):
        if input_ids.device.type != "cpu" or self.offsets.device.type != "cpu":
            raise ValueError("adapter supports CPU gather only")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("head-local IDs must be int32 or int64")
        if input_ids.ndim < 1 or input_ids.shape[-1] != self.num_heads:
            raise ValueError("last input dimension must match the number of heads")
        if bool(((input_ids < 0) | (input_ids >= self.head_sizes)).any()):
            raise IndexError("head-local ID outside its head's vocabulary")
        shape = (*input_ids.shape, self.embedding_dim)
        if input_ids.numel() == 0:
            return torch.empty(shape, dtype=self.dtype)
        global_ids = (input_ids.to(torch.int64) + self.offsets).reshape(-1).tolist()
        raw = self.table.rows(global_ids)
        # A fresh owned tensor avoids exposing mutable mmap/cache backing data.
        return torch.frombuffer(bytearray(raw), dtype=self.dtype).clone().reshape(shape)

    def close(self):
        self.table.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
