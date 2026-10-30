"""Run outside the checkout after installing a wheel; needs no Torch extra."""
import ctypes
from importlib.metadata import distribution
import os
from pathlib import Path
import platform
import json
import struct
import subprocess
import sys
import tempfile

import meft_memory
import ngram_store
from meft_memory import NativeRows, pack_native


def main():
    os.environ.pop("MEFT_ROWS_LIBRARY", None)
    tags = [line for line in distribution("meft-memory").read_text("WHEEL").splitlines() if line.startswith("Tag:")]
    assert tags and all(tag.endswith("_" + platform.machine()) for tag in tags), tags
    assert callable(meft_memory.export_fasttext_model)
    assert ngram_store.NativeRows is meft_memory.NativeRows
    assert ngram_store.StructuralPlan is meft_memory.StructuralPlan
    assert 'torch' not in sys.modules
    assert any(entry.name == "ngram-store" and entry.value == "ngram_store.cli:main"
               for entry in distribution("meft-memory").entry_points)
    assert any(entry.name == "meft-fasttext-export" and
               entry.value == "meft_memory._runtime.fasttext_native_bundle:main"
               for entry in distribution("meft-memory").entry_points)
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
        floats=Path(tmp)/'input.f32'; floats.write_bytes(struct.pack('<6f',1,2,3,4,1,2))
        bundle=Path(tmp)/'engine'
        subprocess.run(['ngram-store','pack','--input',str(floats),'--output',str(bundle),'--dim','2'],check=True,capture_output=True)
        queried=subprocess.run(['ngram-store','query','--store',str(bundle),'--ids','2','0','1'],check=True,capture_output=True,text=True)
        assert json.loads(queried.stdout)['vectors']==[[1,2],[1,2],[3,4]]
        subprocess.run(['ngram-store','fasttext-export','--help'],check=True,capture_output=True)
        print('NGramStore installed CLI/import and historical imports: passed')


if __name__ == "__main__":
    main()
