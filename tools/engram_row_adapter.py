"""CPU inference gather adapter for Engram-demo-shaped multi-head embeddings.

Requires PyTorch. Hashing, tokenization, offsets and row bytes stay unchanged.
This prototype packs an immutable table, not a full Engram checkpoint.
"""
import hashlib
import ctypes
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

try:
    from .lossless_row_store import MappedRows, pack_rows
    from .native_row_store import NativeRows, pack_native
except ImportError:  # direct script imports from tools/
    from lossless_row_store import MappedRows, pack_rows
    from native_row_store import NativeRows, pack_native

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


def export_multihead(weight, head_sizes, output_dir, block_rows=256, backend="python", deduplicate=True,
                     predictor="none"):
    """Preserve every row, including unseen/unused rows, and the source dtype."""
    offsets = head_offsets(head_sizes)
    if weight.layout != torch.strided or weight.ndim != 2 or weight.shape[1] <= 0:
        raise ValueError("weight must be a dense two-dimensional embedding table")
    if str(weight.dtype) not in DTYPES or weight.shape[0] != sum(head_sizes):
        raise ValueError("unsupported dtype or row count inconsistent with head sizes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend not in ("python", "native"):
        raise ValueError("backend must be python or native")
    if predictor not in ("none", "xor") or (predictor != "none" and backend != "native"):
        raise ValueError("xor predictor requires the native backend")
    table = output_dir / ("table.men" if backend == "native" else "table.mer")
    manifest = output_dir / "embedding.json"
    if table.exists() or manifest.exists():
        raise FileExistsError("adapter artifacts already exist; use a fresh output directory")
    if type(block_rows) is not int or block_rows <= 0:
        raise ValueError("block rows must be a positive integer")
    if backend == "python" and not deduplicate:
        raise ValueError("non-deduplicating export requires the native backend")
    width = weight.shape[1] * weight.element_size()
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        source = Path(tmp) / "table.raw"
        # Copy/transfer at most one block instead of materializing another CPU
        # table and a table-sized Python bytes object. Caller must freeze weight
        # for the entire export; detach is not a snapshot of concurrent updates.
        digest = hashlib.sha256()
        with source.open("wb") as stream:
            for start in range(0, weight.shape[0], block_rows):
                chunk = weight[start:start + block_rows].detach().to("cpu").contiguous()
                raw = chunk.view(torch.uint8).numpy().tobytes()
                digest.update(raw)
                stream.write(raw)
        if backend == "native":
            packed = pack_native(source, table, width, block_rows, deduplicate, predictor)
        else:
            packed = pack_rows(source, table, width, block_rows)
    metadata = {
        "schema_version": 1, "table_file": table.name,
        "shape": list(weight.shape), "dtype": str(weight.dtype), "byteorder": sys.byteorder,
        "head_sizes": list(head_sizes), "offsets": offsets,
        "weight_sha256": digest.hexdigest(),
        "artifact_sha256": file_sha256(table),
        "input_bytes": weight.shape[0] * width, "artifact_bytes": table.stat().st_size,
        "unique_rows": packed["unique_rows"], "row_map_bytes": packed["row_map_bytes"],
        "block_rows": block_rows,
        "backend": backend,
        "predictor": predictor,
        "contract": "unchanged local-head IDs plus original offsets; byte-exact CPU gather",
    }
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def export_demo_embedding(module, head_sizes, output_dir, block_rows=256, backend="python", deduplicate=True,
                          predictor="none"):
    """Accept the official demo's .embedding.weight / .offsets interface."""
    expected = torch.tensor(head_offsets(head_sizes), dtype=torch.int64)
    if not torch.equal(module.offsets.detach().cpu(), expected):
        raise ValueError("source offsets differ from declared head sizes")
    return export_multihead(module.embedding.weight, head_sizes, output_dir, block_rows, backend, deduplicate,
                            predictor)


class MappedMultiHeadEmbedding(nn.Module):
    """Serving-only CPU replacement for MultiHeadEmbedding.forward's gather.

    Only the embedding table is replaced. Input [..., heads] and output
    [..., heads, embedding_dim] preserve the official gather's layout.
    """

    def __init__(self, output_dir, cache_blocks=4, native_library=None, native_cache_bytes=0,
                 query_chunk_rows=65536, *, native_cache_policy="blocks"):
        super().__init__()
        if query_chunk_rows is not None and (type(query_chunk_rows) is not int or
                                            not 1 <= query_chunk_rows <= ctypes.c_size_t(-1).value):
            raise ValueError("query_chunk_rows must be a positive size_t or None")
        self.query_chunk_rows = query_chunk_rows
        output_dir = Path(output_dir).resolve()
        meta = json.loads((output_dir / "embedding.json").read_text())
        backend = meta.get("backend", "python")
        if native_cache_policy not in ("blocks", "rows"):
            raise ValueError("native cache policy must be blocks or rows")
        if backend != "native" and (native_cache_bytes or native_cache_policy != "blocks"):
            raise ValueError("native_cache_bytes requires a native artifact")
        table_name = {"python": "table.mer", "native": "table.men"}.get(backend)
        if (meta["schema_version"] != 1 or meta["dtype"] not in DTYPES
                or meta["byteorder"] != sys.byteorder or meta["table_file"] != table_name):
            raise ValueError("unsupported adapter metadata")
        offsets = head_offsets(meta["head_sizes"])
        if meta["offsets"] != offsets or meta["shape"][0] != sum(meta["head_sizes"]):
            raise ValueError("inconsistent head metadata")
        self.dtype = DTYPES[meta["dtype"]]
        self.num_heads = len(offsets)
        self.embedding_dim = meta["shape"][1]
        self.metadata = meta
        self.backend = backend
        self._output_dir = str(output_dir)
        self._cache_blocks = cache_blocks
        self._native_library = native_library
        self._native_cache_bytes = native_cache_bytes
        self._native_cache_policy = native_cache_policy
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64, device="cpu"))
        self.register_buffer("head_sizes", torch.tensor(meta["head_sizes"], dtype=torch.int64, device="cpu"))
        self._open_table()
        if backend == "native":
            # Capture the resolved library, including an environment-selected
            # path, instead of depending on another worker's environment/cwd.
            self._native_library = self.table.lib._name
        self.training = False

    def _open_table(self):
        path = Path(self._output_dir) / self.metadata["table_file"]
        if file_sha256(path) != self.metadata["artifact_sha256"]:
            raise ValueError("row-store checksum mismatch")
        self.table = (NativeRows(path, self._native_library, self._native_cache_bytes,
                                 cache_policy=self._native_cache_policy)
                      if self.backend == "native" else MappedRows(path, self._cache_blocks))
        width = self.embedding_dim * torch.empty((), dtype=self.dtype).element_size()
        if (self.table.header["n_rows"] != self.metadata["shape"][0]
                or self.table.header["row_bytes"] != width):
            self.table.close()
            raise ValueError("row-store shape differs from embedding metadata")

    def __getstate__(self):
        """Serialize configuration/buffers, never mmap, C pointers or row caches."""
        closed = not self.table.handle if self.backend == "native" else self.table.file.closed
        if closed:
            raise ValueError("cannot serialize a closed mapped embedding")
        state = super().__getstate__()
        state.pop("table")
        return state

    def __setstate__(self, state):
        state.setdefault("query_chunk_rows", 65536)
        super().__setstate__(state)
        # Copy/deepcopy and spawned workers each own a checksum-verified handle.
        # A replaced table is rejected rather than silently changing snapshots.
        self._open_table()

    def train(self, mode=True):
        if mode:
            raise RuntimeError("mapped embeddings are immutable and support inference only")
        return super().train(False)

    def _global_ids(self, input_ids):
        """Validate original local-head addressing before any native writes."""
        if input_ids.device.type != "cpu" or self.offsets.device.type != "cpu":
            raise ValueError("adapter supports CPU gather only")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("head-local IDs must be int32 or int64")
        if input_ids.ndim < 1 or input_ids.shape[-1] != self.num_heads:
            raise ValueError("last input dimension must match the number of heads")
        if bool(((input_ids < 0) | (input_ids >= self.head_sizes)).any()):
            raise IndexError("head-local ID outside its head's vocabulary")
        return (input_ids.to(torch.int64) + self.offsets).contiguous()

    def forward(self, input_ids):
        global_tensor = self._global_ids(input_ids)
        shape = (*input_ids.shape, self.embedding_dim)
        if input_ids.numel() == 0:
            if self.backend == "native":
                self.table.gather_into(0, 0, 0, 0)
            elif self.table.file.closed:
                raise ValueError("row store is closed")
            return torch.empty(shape, dtype=self.dtype, device="cpu")
        if self.backend == "native":
            output = torch.empty(shape, dtype=self.dtype, device="cpu")
            self.table.gather_into(global_tensor.data_ptr(), global_tensor.numel(),
                                   output.data_ptr(), output.numel() * output.element_size(),
                                   max_query_rows=self.query_chunk_rows)
            return output
        raw = self.table.rows(global_tensor.reshape(-1).tolist())
        # A fresh owned tensor avoids exposing mutable mmap/cache backing data.
        return torch.frombuffer(bytearray(raw), dtype=self.dtype).clone().reshape(shape)

    def close(self):
        self.table.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_demo_embedding(output_dir, list_of_N, D, cache_blocks=4, native_library=None, native_cache_bytes=0,
                        query_chunk_rows=65536, *, native_cache_policy="blocks"):
    """Factory-compatible loader for one exported immutable CPU layer.

    Bind output_dir with functools.partial for Engram's proposed factory hook.
    Validate the requesting layer's layout; caller owns returned module.close().
    """
    head_offsets(list_of_N)
    if type(D) is not int or D <= 0:
        raise ValueError("embedding dimension must be a positive integer")
    mapped = MappedMultiHeadEmbedding(output_dir, cache_blocks, native_library, native_cache_bytes,
                                     query_chunk_rows, native_cache_policy=native_cache_policy)
    if list(list_of_N) != mapped.metadata["head_sizes"] or D != mapped.embedding_dim:
        mapped.close()
        raise ValueError("exported head sizes/dimension differ from requesting Engram layer")
    return mapped


def export_embedding(module, output_dir, block_rows=256, backend="native", deduplicate=True,
                     predictor="none"):
    """Export ordinary nn.Embedding without forward-time max_norm mutation."""
    if not isinstance(module, nn.Embedding) or module.max_norm is not None:
        raise ValueError("requires nn.Embedding without forward-time max_norm mutation")
    return export_multihead(module.weight, [module.num_embeddings], output_dir,
                            block_rows, backend, deduplicate, predictor)


class MappedEmbedding(nn.Module):
    """Ordinary global-ID embedding gather. No .weight parameter/tied-head support.

    Compatible with callers that only invoke embedding(ids); model-specific
    wrappers that inspect .weight or perform training need separate integration.
    """
    def __init__(self, output_dir, native_library=None, native_cache_bytes=0, query_chunk_rows=65536,
                 *, native_cache_policy="blocks"):
        super().__init__()
        self.gather = MappedMultiHeadEmbedding(output_dir, native_library=native_library,
                                             native_cache_bytes=native_cache_bytes,
                                             native_cache_policy=native_cache_policy,
                                             query_chunk_rows=query_chunk_rows)
        if self.gather.num_heads != 1:
            self.gather.close()
            raise ValueError("ordinary embedding needs a single global-ID table")
        self.num_embeddings = self.gather.metadata["shape"][0]
        self.embedding_dim = self.gather.embedding_dim
        self.training = False

    def train(self, mode=True):
        if mode:
            raise RuntimeError("mapped embeddings support inference only")
        return super().train(False)

    def forward(self, ids):
        return self.gather(ids.unsqueeze(-1)).squeeze(-2)

    def close(self):
        self.gather.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
