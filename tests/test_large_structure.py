"""Correctness fixtures only; fabricated values are never experimental evidence."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipUnless(np is not None,'NumPy required')
class StructureAuditTest(unittest.TestCase):
    def test_actual_id_sampling_and_sibling_boundary_control(self):
        from tools.benchmark_large_structure import sample_pairs,sample_siblings,related,shared_affix
        keys=['ab','ac','<a','abc','abd','<ab','<ac','xbc','ybc','zzc','mnop','mnox']
        a=sample_pairs(keys,100,42)
        self.assertEqual(a,sample_pairs(keys,100,42))
        for entry in a:
            for source,target in entry['pairs']:
                self.assertEqual(len(keys[source]),entry['order'])
                self.assertEqual(len(keys[target]),entry['target_order'])
                self.assertTrue(related(keys[source],keys[target]))
            for source,target in entry['controls']:
                self.assertFalse(related(keys[source],keys[target]))
        siblings=sample_siblings(keys,100,42)
        self.assertTrue(siblings)
        for entry in siblings:
            for i,j in entry['pairs']:
                self.assertEqual(len(keys[i]),len(keys[j]))
                self.assertGreaterEqual(shared_affix(keys[i],keys[j],entry['direction'].split('-')[1]),2)
                self.assertFalse(related(keys[i],keys[j]))

    def test_cosine_ratio_ablation_and_direct_error(self):
        from tools.benchmark_large_structure import pair_summary,row_metrics
        rows=np.array([[1.,0.],[2.,0.],[0.,1.],[0.,0.]],dtype='<f4')
        result=pair_summary(rows,[(0,1),(0,2),(0,3)],[.99],.99)['thresholds'][0]
        self.assertAlmostEqual(result['cosine_only_fraction'],1/3)
        self.assertAlmostEqual(result['ratio_only_fraction'],1/3)
        self.assertEqual(result['joint_fraction'],0)
        self.assertEqual(result['cosine_only_error']['max'],1)
        self.assertAlmostEqual(result['ratio_only_error']['max'],2**.5)
        self.assertFalse(row_metrics(rows[3:],rows[3:])['nonzero'][0])

    @unittest.skipUnless(shutil.which('cc'),'C compiler required')
    def test_full_encoded_forwarding_and_tamper_guard(self):
        from tools.benchmark_large_structure import build,verify,write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); training=root/'training';training.mkdir()
            library=root/('rows.dylib' if sys.platform=='darwin' else 'rows.so')
            source_c=Path(__file__).resolve().parents[1]/'native/meft_rows.c'
            subprocess.run(['cc','-std=c11','-O2','-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared',
                str(source_c),'-lz','-o',str(library)],check=True,capture_output=True)
            rows=np.array([[1.,.1],[1.,.1001],[0.,1.],[0.,.9999]],dtype='<f4')
            (training/'trained.f32').write_bytes(rows.tobytes())
            write_json(training/'keys.json',['ab','abc','xy','xyz'])
            write_json(training/'training.json',dict(ngram_rows=4,dim=2))
            np.array([0,1,2,3],dtype='<u8').tofile(training/'queries.u64')
            mapping=root/'rows.map';np.array([0,0,2,2],dtype='<u8').tofile(mapping)
            order=root/'rows.order';np.array([0,2,1,3],dtype='<u8').tofile(order)
            cfg=Namespace(training_dir=training,output_dir=root,native_library=library,map_file=mapping,
                order_file=order,policy='joint',threshold=.99,norm_ratio=.99,max_block_bytes=16,chunk_rows=2,
                worker_report=root/'worker.json')
            for encoding in ('fp32','fp16','int8'):
                cfg.encoding=encoding;cfg.bundle=root/encoding
                build(cfg);verify(cfg)
                result=json.loads(cfg.worker_report.read_text())
                self.assertEqual(result['all_rows_verified'],4)
                self.assertEqual(result['overall']['merged_rows'],2)
                self.assertTrue(result['all_encoded_and_decoded_original_ids_byte_exact_to_representative'])
            # A valid idempotent but incorrect oracle must fail byte verification.
            np.array([0,1,2,3],dtype='<u8').tofile(mapping)
            with self.assertRaisesRegex(AssertionError,'representative bytes differ'):
                verify(cfg)


if __name__=='__main__':
    unittest.main()
