"""Run outside the checkout after installing a wheel; needs no Torch extra."""
import ctypes
import os
from pathlib import Path
import tempfile

import meft_memory
from meft_memory import NativeRows, pack_native


def main():
    os.environ.pop("MEFT_ROWS_LIBRARY", None)
    with tempfile.TemporaryDirectory() as tmp:
        source, packed = Path(tmp) / "raw", Path(tmp) / "table"
        source.write_bytes(b"ABCDEFGHabcdefgh")
        pack_native(source, packed, 8, 1, False)
        with NativeRows(packed) as store:
            assert Path(store.lib._name).parent == Path(meft_memory.__file__).resolve().parent / "_native"
            query = (ctypes.c_uint64 * 3)(1, 0, 1)
            output = ctypes.create_string_buffer(24)
            store.gather_into(ctypes.addressof(query), 3, ctypes.addressof(output), len(output), max_query_rows=1)
            assert output.raw == b"abcdefghABCDEFGHabcdefgh"
            assert store.last_block_reads == 3
        print("Installed native wheel, automatic library lookup and chunked query: passed")


if __name__ == "__main__":
    main()
