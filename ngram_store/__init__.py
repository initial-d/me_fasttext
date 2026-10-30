"""NGramStore engine API; historical meft_memory imports remain compatible."""
import meft_memory as _compatible
from meft_memory import NativeRows, pack_native, __version__

__all__ = list(_compatible.__all__)


def __getattr__(name):
    return getattr(_compatible, name)
