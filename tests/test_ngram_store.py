"""Engine entry points and corpus-to-trainer protocol; fixtures are correctness only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import unittest

import meft_memory
import ngram_store
from ngram_store.cli import main,pack_store
from tools.prepare_fasttext_corpus import prepare_corpus

ROOT=Path(__file__).resolve().parents[1]


class CorpusTest(unittest.TestCase):
    def test_adjacent_pairs_utf8_order_line_boundaries_and_failure_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'plain.txt'; output=Path(tmp)/'pairs.txt'
            source.write_text('Alpha beta gamma delta epsilon\n记忆 查询 向量\nalone\n')
            report=prepare_corpus(source,output,words_per_line=3,lowercase=True)
            self.assertEqual(output.read_text(),'p#0 alpha\x01beta beta\x01gamma\np#0 delta\x01epsilon\np#0 记忆\x01查询 查询\x01向量\n')
            self.assertEqual((report['pair_tokens'],report['prepared_lines'],report['skipped_singletons']),(5,3,1))
            before=output.read_bytes()
            with self.assertRaises(FileExistsError):prepare_corpus(source,output)
            self.assertEqual(output.read_bytes(),before)
            bad=Path(tmp)/'bad.txt'; bad.write_text('valid words\ninvalid\x01token x\n')
            failed=Path(tmp)/'failed.txt'
            with self.assertRaises(ValueError):prepare_corpus(bad,failed)
            self.assertFalse(failed.exists())
            bad.write_text('singleton\n')
            with self.assertRaises(ValueError):prepare_corpus(bad,failed)
            self.assertFalse(failed.exists())


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)
        cls.library=cls.root/('rows.dylib' if platform.system()=='Darwin' else 'rows.so')
        subprocess.run(['cc','-std=c11','-O2','-fPIC','-dynamiclib' if platform.system()=='Darwin' else '-shared',
                        str(ROOT/'native/meft_rows.c'),'-lz','-o',str(cls.library)],check=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_old_imports_share_objects_and_all_bits_survive_cli_and_adapter_metadata(self):
        self.assertIs(ngram_store.NativeRows,meft_memory.NativeRows)
        self.assertIs(ngram_store.StructuralPlan,meft_memory.StructuralPlan)
        self.assertEqual(ngram_store.__all__,meft_memory.__all__)
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            root=Path(tmp); source=root/'input.f32'; bundle=root/'store'
            raw=struct.pack('<6I',0,0x80000000,0x7fc00042,0x3f800000,0,0x80000000)
            source.write_bytes(raw)
            result=pack_store(source,bundle,2,native_library=self.library,deduplicate=True,max_block_bytes=8)
            self.assertTrue(result['original_bytes_preserved'])
            self.assertEqual(result['source_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['reconstructed_sha256'],result['source_sha256'])
            self.assertEqual(result['storage']['unique_rows'],2)
            with ngram_store.NativeRows(bundle/'table.men',self.library) as table:
                self.assertEqual(table.rows([2,1,0,2]),raw[16:24]+raw[8:16]+raw[:8]+raw[16:24])
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured):main(['query','--store',str(bundle),'--ids','2','1','0','2','--hex','--native-library',str(self.library)])
            self.assertEqual(json.loads(captured.getvalue())['rows_hex'],[raw[i*8:i*8+8].hex() for i in [2,1,0,2]])
            metadata=json.loads((bundle/'embedding.json').read_text())
            self.assertEqual(metadata['shape'],[3,2]); self.assertEqual(metadata['head_sizes'],[3])
            self.assertEqual(metadata['weight_sha256'],result['source_sha256'])
            with self.assertRaises(FileExistsError):pack_store(source,bundle,2,native_library=self.library)
            bad=root/'partial.f32'; bad.write_bytes(b'123')
            with self.assertRaises(ValueError):pack_store(bad,root/'failed',2,native_library=self.library)
            self.assertFalse((root/'failed').exists())

    def test_packed_engine_store_opens_existing_torch_adapter(self):
        try:import torch
        except ImportError:self.skipTest('optional torch dependency')
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            root=Path(tmp); source=root/'trained-fixture.f32'; source.write_bytes(struct.pack('<6f',1,2,3,4,1,2))
            bundle=root/'store'; pack_store(source,bundle,2,native_library=self.library)
            model=ngram_store.MappedEmbedding(bundle,native_library=self.library)
            try:self.assertTrue(torch.equal(model(torch.tensor([2,0,1])),torch.tensor([[1.,2.],[1.,2.],[3.,4.]])))
            finally:model.close()


if __name__=='__main__':unittest.main()
