"""Immutable CPU EmbeddingBag over native lossless row storage.

No full [bags, entries, dim] gather tensor is built. The default stages distinct
query IDs once, then invokes PyTorch's native EmbeddingBag reduction. Optional
chunked C reduction bounds staged rows and accumulates in FP64. Storage remains
byte-exact; reduction order/precision can change model logits.
"""
import json
import ctypes
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .engram_row_adapter import MappedMultiHeadEmbedding, export_multihead
except ImportError:
    from engram_row_adapter import MappedMultiHeadEmbedding, export_multihead


class MappedEmbeddingBag(nn.Module):
    """Sum/mean of 2-D fixed bags or 1-D IDs plus complete ragged offsets.

    Ragged offsets start at 0; include_last_offset=True supplies the terminal
    count, otherwise it is appended. Empty/all-padding bags return zeros.
    No gradients, per-sample weights, max mode, .weight or tied-head support.
    One single-worker native handle per module; keep it on CPU and close it.
    """
    is_embedding_bag = True

    def __init__(self, output_dir, mode="sum", padding_idx=None, include_last_offset=False,
                 chunk_rows=1024, native_library=None, native_cache_bytes=0, strategy="unique"):
        super().__init__()
        if mode not in ("sum", "mean") or type(include_last_offset) is not bool:
            raise ValueError("sum/mean mode and boolean include_last_offset required")
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= ctypes.c_size_t(-1).value:
            raise ValueError("chunk_rows must be a positive integer")
        if strategy not in ("unique", "chunked"):
            raise ValueError("strategy must be unique or chunked")
        self.gather = MappedMultiHeadEmbedding(output_dir, native_library=native_library,
                                             native_cache_bytes=native_cache_bytes)
        self.num_embeddings, self.embedding_dim = self.gather.metadata["shape"]
        if (self.gather.backend != "native" or self.gather.num_heads != 1
                or self.gather.metadata["byteorder"] != "little"):
            self.close()
            raise ValueError("embedding bags need a single native little-endian table")
        if padding_idx is not None:
            if type(padding_idx) is not int or not -self.num_embeddings <= padding_idx < self.num_embeddings:
                self.close()
                raise ValueError("padding_idx must be an existing row or None")
            padding_idx %= self.num_embeddings
        self.mode, self.padding_idx = mode, padding_idx
        self.include_last_offset, self.chunk_rows = include_last_offset, chunk_rows
        self.strategy, self.last_unique_rows = strategy, None
        self.dtype = self.gather.dtype
        self.dtype_code = {torch.float32: 1, torch.float16: 2, torch.bfloat16: 3}[self.dtype]
        self.training = False

    def train(self, mode=True):
        if mode:
            raise RuntimeError("mapped embedding bags are immutable and support inference only")
        return super().train(False)

    def forward(self, input, offsets=None, per_sample_weights=None):
        if input.device.type != "cpu" or self.gather.offsets.device.type != "cpu":
            raise ValueError("embedding bags support CPU only")
        if input.dtype not in (torch.int32, torch.int64):
            raise TypeError("bag IDs must be int32 or int64")
        if per_sample_weights is not None:
            raise ValueError("per-sample weights are not supported")
        if input.ndim == 2:
            if offsets is not None:
                raise ValueError("2-D bags do not take offsets")
            boundary = torch.arange(input.shape[0] + 1, dtype=torch.int64) * input.shape[1]
        elif input.ndim == 1:
            if offsets is None or offsets.ndim != 1 or offsets.device.type != "cpu":
                raise ValueError("1-D bags need CPU vector offsets")
            if offsets.dtype not in (torch.int32, torch.int64):
                raise TypeError("offsets must be int32 or int64")
            boundary = offsets.to(torch.int64).contiguous()
            if self.include_last_offset:
                if not boundary.numel():
                    raise ValueError("offsets must include a terminal count")
            else:
                boundary = torch.cat([boundary, torch.tensor([input.numel()], dtype=torch.int64)])
        else:
            raise ValueError("bag IDs must have one or two dimensions")
        ids = input.to(torch.int64).reshape(-1).contiguous()
        if bool(((ids < 0) | (ids >= self.num_embeddings)).any()):
            raise IndexError("bag row ID out of bounds")
        if (boundary[0] != 0 or boundary[-1] != ids.numel()
                or bool((boundary[1:] < boundary[:-1]).any())):
            raise ValueError("offsets must be nondecreasing and span all IDs")
        bags = boundary.numel() - 1
        if self.strategy == "unique":
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            self.last_unique_rows = unique.numel()
            if not unique.numel():
                # Validate the handle and reset last-query counters for empty input.
                self.gather.table.gather_into(ids.data_ptr(), 0, 0, 0)
                return torch.zeros((bags, self.embedding_dim), dtype=self.dtype)
            rows = torch.empty((unique.numel(), self.embedding_dim), dtype=self.dtype)
            self.gather.table.gather_into(unique.data_ptr(), unique.numel(), rows.data_ptr(),
                                         rows.numel() * rows.element_size())
            padding = None
            if self.padding_idx is not None:
                positions = (unique == self.padding_idx).nonzero().reshape(-1)
                if positions.numel():
                    padding = int(positions[0])
            return F.embedding_bag(inverse, rows, boundary, mode=self.mode,
                                   include_last_offset=True, padding_idx=padding)
        self.last_unique_rows = None
        output = torch.empty((bags, self.embedding_dim), dtype=torch.float32)
        self.gather.table.embedding_bag_into(
            ids.data_ptr(), ids.numel(), boundary.data_ptr(), bags, self.dtype_code,
            self.mode == "mean", self.padding_idx if self.padding_idx is not None else 2**64 - 1,
            self.chunk_rows, output.data_ptr(), output.numel(),
        )
        return output.to(self.dtype)

    def close(self):
        self.gather.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def export_embedding_bag(module, output_dir, block_rows=256, deduplicate=True, predictor="none"):
    """Export nn.EmbeddingBag's rows and its inference reduction configuration."""
    if (not isinstance(module, nn.EmbeddingBag) or module.mode not in ("sum", "mean")
            or module.max_norm is not None):
        raise ValueError("requires sum/mean nn.EmbeddingBag without max_norm mutation")
    metadata = export_multihead(module.weight, [module.num_embeddings], output_dir, block_rows,
                               "native", deduplicate, predictor)
    metadata["embedding_bag"] = {"mode": module.mode, "padding_idx": module.padding_idx,
                                 "include_last_offset": module.include_last_offset}
    (Path(output_dir) / "embedding.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def open_embedding_bag(output_dir, **kwargs):
    """Load the mode/padding/offset convention exported by export_embedding_bag."""
    config = json.loads((Path(output_dir) / "embedding.json").read_text()).get("embedding_bag")
    if config is None:
        raise ValueError("no exported embedding-bag configuration; use MappedEmbeddingBag explicitly")
    return MappedEmbeddingBag(output_dir, **config, **kwargs)
