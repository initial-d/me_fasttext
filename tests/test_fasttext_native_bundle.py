"""Actual legacy Linux loader/DASTrie/native facade; fixtures test contracts only."""
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
import unittest

from tools.fasttext_native_bundle import export_fasttext_model, read_keys

ROOT=Path(__file__).resolve().parents[1]


class RegistryTest(unittest.TestCase):
    def test_rejects_truncated_duplicate_and_out_of_range_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'keys'
            valid=b'MEFTKEY1'+struct.pack('<Q',1)+struct.pack('<QQ',2,2)+b'ab'
            path.write_bytes(valid)
            self.assertEqual(read_keys(path,3),{2:'ab'})
            for raw in (valid[:-1],valid+b'x',valid[:8],b'MEFTKEY1'+struct.pack('<Q',4),
                        b'MEFTKEY1'+struct.pack('<Q',1)+struct.pack('<QQ',3,2)+b'ab'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):read_keys(path,3)


@unittest.skipUnless(platform.system()=='Linux' and platform.machine() in ('x86_64','AMD64'),
                     'legacy bundled DATrie/FastText require Linux x86-64')
class NativeFastTextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)
        cls.executable=cls.root/'fixture'; cls.library=cls.root/'rows.so'; cls.structural=cls.root/'structural.so'
        cls.object=cls.root/'rows.o'
        sanitizers=(['-fsanitize=address,undefined','-fno-omit-frame-pointer']
                    if os.environ.get('MEFT_FUSION_SANITIZERS')=='1' else [])
        subprocess.run(['cc','-std=c11','-O1','-g','-fPIC',*sanitizers,'-c',str(ROOT/'native/meft_rows.c'),'-o',str(cls.object)],check=True)
        sources=['args','dictionary','productquantizer','matrix','qmatrix','vector','model','utils','fasttext','ftindex']
        subprocess.run(['c++','-std=c++11','-O1','-g','-pthread',*sanitizers,'-DMEFT_NATIVE_FASTTEXT',
            '-I'+str(ROOT/'src'),'-I'+str(ROOT/'native'),'-I'+str(ROOT/'mylib/include'),
            str(ROOT/'tests/fasttext_native_fixture.cc'),*[str(ROOT/f'src/{n}.cc') for n in sources],
            str(ROOT/'native/fasttext_bundle.cc'),str(cls.object),'-L'+str(ROOT/'mylib/lib'),'-ltrie','-lpub','-lz',
            '-o',str(cls.executable)],check=True)
        subprocess.run(['make','fasttext-native','native','structural-native'],cwd=ROOT,check=True,capture_output=True)
        cls.binary=ROOT/'build/fasttext-native'
        cls.library=ROOT/'build/libmeft_rows.so'; cls.structural=ROOT/'build/libmeft_structural.so'

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_real_model_all_rows_and_legacy_inference_are_exact(self):
        for dim in (7,32):
            with self.subTest(dim=dim),tempfile.TemporaryDirectory(dir=self.root) as tmp:
                root=Path(tmp); model=root/'original.bin'
                subprocess.run([str(self.executable),'create',str(model),str(dim)],cwd=root,check=True,capture_output=True)
                for mode in ('layout','exact'):
                    bundle=root/mode
                    report=export_fasttext_model(model,bundle,fasttext_binary=self.binary,
                        structural_library=self.structural,native_library=self.library,mode=mode,
                        max_block_bytes=dim*4*2,cache_bytes=dim*4*3,query_chunk_rows=3)
                    self.assertTrue(report['original_bytes_preserved'])
                    self.assertEqual(report['source_rows_sha256'],report['reconstructed_rows_sha256'])
                    metadata=json.loads((bundle/'embedding.json').read_text())
                    self.assertEqual(metadata['block_rows'],2)
                    check=subprocess.run([str(self.executable),'check',str(model),str(bundle)],cwd=root,capture_output=True,text=True)
                    self.assertEqual(check.returncode,0,check.stdout+check.stderr)
                    self.assertFalse((bundle/'forbidden.bin').exists())
                    words='alpha\nbeta\nalphax\n你好\n'
                    dense=subprocess.run([str(self.binary),'print-word-vectors',str(model)],input=words,text=True,capture_output=True,check=True)
                    native=subprocess.run([str(self.binary),'print-word-vectors',str(bundle)],input=words,text=True,capture_output=True,check=True)
                    self.assertEqual([line for line in dense.stdout.splitlines() if line.split(' ')[0] in words.split()],native.stdout.splitlines())
                    with self.assertRaises(FileExistsError):
                        export_fasttext_model(model,bundle,fasttext_binary=self.binary,structural_library=self.structural)
                failed=root/'failed'
                invalid=root/'invalid.bin'; invalid.write_bytes(b'not a dense model')
                with self.assertRaises(RuntimeError):
                    export_fasttext_model(invalid,failed,fasttext_binary=self.binary,structural_library=self.structural)
                self.assertFalse(failed.exists())


if __name__=='__main__':unittest.main()
