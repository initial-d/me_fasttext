"""Guard trained-row provenance and fresh-process capacity oracles."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_NUMPY = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_TORCH and HAS_NUMPY, 'training requires CPU Torch and NumPy')
class CapacityProvenanceTest(unittest.TestCase):
    def test_initialized_rows_are_rejected_or_explicitly_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root/'corpus.zip'
            text = ('alpha beta gamma delta alphabet beta gamma delta\n'*60
                    + 'uniquea isolated uniqueb uniquec isolated uniqued\n')
            # Keep the isolation fragment once inside the 90% training split.
            with zipfile.ZipFile(corpus,'w') as archive:
                archive.writestr('enwik8',text + 'alpha beta gamma delta\n'*60)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1')
            common = [sys.executable,'-m','tools.train_capacity_memory','--corpus-zip',str(corpus),
                      '--order','4','--dim','8','--epochs','2','--random-pairs','256',
                      '--batch-size','16','--threads','1']
            strict = root/'strict'
            rejected = subprocess.run(common+['--output-dir',str(strict)],env=env,cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(rejected.returncode,0,rejected.stdout+rejected.stderr)
            failure = json.loads((strict/'training.json').read_text())
            self.assertEqual(failure['status'],'incomplete-training-coverage')
            self.assertFalse((strict/'trained.f32').exists())
            exported = root/'exported'
            accepted = subprocess.run(common+['--output-dir',str(exported),'--exclude-untrained'],
                                      env=env,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(accepted.returncode,0,accepted.stdout+accepted.stderr)
            report = json.loads((exported/'training.json').read_text())
            self.assertGreater(report['excluded_untrained_rows'],0)
            self.assertEqual(report['registered_ngram_rows'],report['ngram_rows']+report['excluded_untrained_rows'])
            self.assertEqual(report['changed_rows'],report['ngram_rows'])
            self.assertTrue(report['all_rows_changed'])
            self.assertEqual((exported/'trained.f32').stat().st_size,report['ngram_rows']*8*4)
            self.assertEqual((exported/'trained-source-ids.u64').stat().st_size,report['ngram_rows']*8)
            self.assertEqual(len(json.loads((exported/'keys.json').read_text())),report['ngram_rows'])
            if not shutil.which('cc'):
                return
            library = root/('native.dylib' if sys.platform=='darwin' else 'native.so')
            subprocess.run(['cc','-std=c11','-O2','-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared',
                            str(ROOT/'native/meft_rows.c'),'-lz','-o',str(library)],check=True,capture_output=True)
            capacity = root/'capacity'
            measured = subprocess.run([sys.executable,'-m','tools.benchmark_memory_capacity',
                '--training-dir',str(exported),'--output-dir',str(capacity),'--native-library',str(library),
                '--repeat','1','--query-rows','32'],env=env,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(measured.returncode,0,measured.stdout+measured.stderr)
            result = json.loads((capacity/'report.json').read_text())
            self.assertEqual(result['status'],'passed')
            self.assertEqual(set(result['workers']),{'resident','raw-mmap','native-ordinary'})
            for variant in result['workers'].values():
                self.assertTrue(all(case['all_query_bytes_exact'] for case in variant['result']['cases']))
            sparse = root / 'sparse'
            ablation = subprocess.run([sys.executable, '-m', 'tools.benchmark_sparse_blocks',
                '--training-dir', str(exported), '--query-manifest', str(capacity / 'queries-ordinary.json'),
                '--output-dir', str(sparse), '--native-library', str(library), '--repeat', '1',
                '--block-rows', '1', '4', '256', '--reuse-baseline', str(capacity / 'native-ordinary.men')],
                env=env, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(ablation.returncode, 0, ablation.stdout + ablation.stderr)
            layouts = json.loads((sparse / 'report.json').read_text())
            self.assertEqual(layouts['status'], 'passed')
            for name, packed in layouts['variants'].items():
                self.assertEqual(packed['reconstructed_sha256'], layouts['source_sha256'])
                self.assertTrue(packed['all_original_ids_byte_exact'])
                self.assertEqual(packed['reused_artifact'], name == 'block-256')
                worker = layouts['workers'][name]['result']
                for case in worker['cases']:
                    self.assertTrue(case['all_query_bytes_exact'])
                    self.assertTrue(all(isinstance(work['last_decoded_bytes'], int)
                                        for work in case['first_codec_work']))


@unittest.skipUnless(HAS_NUMPY, 'capacity workers require NumPy')
class CapacityImportsTest(unittest.TestCase):
    def test_workers_do_not_import_torch_or_initialize_cuda(self):
        check = subprocess.run([sys.executable,'-c',
            'import sys; import tools.benchmark_memory_capacity; import tools.benchmark_sparse_blocks; assert "torch" not in sys.modules'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(check.returncode,0,check.stderr)


if __name__ == '__main__':
    unittest.main()
