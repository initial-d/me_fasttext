"""Installable native memory API; Torch/Transformers adapters load on demand."""
from importlib import import_module

from ._runtime.native_row_store import NativeRows, pack_native

__version__ = "0.3.0"

_ADAPTERS = {
    "StructuralPlan": "structural_native",
    "StructuralIndex": "structural_native",
    "MappedMultiHeadEmbedding": "engram_row_adapter",
    "StagedMultiHeadEmbedding": "device_row_adapter",
    "GatherTicket": "device_row_adapter",
    "MappedEmbedding": "engram_row_adapter",
    "export_multihead": "engram_row_adapter",
    "export_embedding": "engram_row_adapter",
    "MappedEmbeddingBag": "mapped_embedding_bag",
    "TinyEngramModel": "tinyengram_checkpoint",
    "TinyEngramCache": "tinyengram_checkpoint",
    "export_tinyengram_checkpoint": "tinyengram_checkpoint",
    "load_tinyengram_checkpoint": "tinyengram_checkpoint",
    "export_fasttext_model": "fasttext_native_bundle",
}


def __getattr__(name):
    if name not in _ADAPTERS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}._runtime.{_ADAPTERS[name]}"), name)


__all__ = ["NativeRows", "pack_native", *_ADAPTERS]
