"""Correctness fixtures only; fabricated values are never experimental evidence."""
from argparse import Namespace
import hashlib
import copy
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

    def test_complete_control_denominator_and_missing_round_rejection(self):
        from tools.analyze_large_structure import analyze
        info=dict(shape=[4,2],source_sha256='fixture',source_commit='fixture',
            keys_bytes=10000,settings=dict(encodings=['fp32'],rounds=1),limitations=['fixture'])
        variants=[]
        workers=[dict(phase='audit',name='pair-audit',returncode=0,result=dict(status='passed',source_commit='fixture',by_order={},strata=[]))]
        for policy,size in [('raw',1000),('identity',800),('joint',850)]:
            name='fp32-'+policy
            fidelity=dict(rows=4,merged_rows=1)
            verification=dict(all_rows_verified=4,all_encoded_and_decoded_original_ids_byte_exact_to_representative=True,
                overall=fidelity,merged_only={},by_order={},heldout_query_weighted_relative_frobenius=0,
                heldout_query_weighted_added_frobenius=0,composed_ngram_mean=None)
            built=dict(bundle_bytes=size,metadata=dict(storage={}),build_seconds=0,
                memory_after=dict(peak_rss_bytes=1))
            variants.append(dict(name=name,encoding='fp32',policy=policy,threshold=.99 if policy=='joint' else None,
                build=built,verification=verification))
            for case in ('heldout-lexical','uniform-id-stress'):
                value=dict(status='passed',source_commit='query-fixture',all_first_and_second_query_bytes_exact=True)
                value.update({stage:dict(wall=dict(p50_ms=1),cpu=dict(p50_ms=1))
                    for stage in ('first_trace','second_trace','hot_trace')})
                value.update({stage:dict(peak_rss_bytes=1) for stage in ('memory_open','memory_first','memory_second','memory_hot')})
                workers.append(dict(phase='query',name='query-'+name+'-'+case+'-0',returncode=0,result=value))
        report=dict(status='passed',provenance=info,workers=workers,variants=variants,index_bytes={'keys.das':5000})
        summarized=analyze(report)
        self.assertEqual(summarized['worker_source_commits'],['fixture','query-fixture'])
        joint=next(v for v in summarized['variants'] if v['policy']=='joint')
        self.assertAlmostEqual(joint['additional_saving_vs_best_same_encoding_control'],-50/10800)
        self.assertEqual(joint['bundle_with_datrie_bytes'],5850)
        missing=copy.deepcopy(report);missing['workers'].pop()
        with self.assertRaisesRegex(ValueError,'missing independent query rounds'):
            analyze(missing)
        corrupt=copy.deepcopy(report);corrupt['variants'][0]['verification']['all_rows_verified']=3
        with self.assertRaisesRegex(ValueError,'full original-ID verification missing'):
            analyze(corrupt)

    @unittest.skipUnless(shutil.which('cc'),'C compiler required')
    def test_full_encoded_forwarding_and_tamper_guard(self):
        from tools.benchmark_large_structure import build,verify,query,write_json
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
            np.array([3,1,0,2],dtype='<u8').tofile(root/'heldout-lexical.u64')
            mapping=root/'rows.map';np.array([0,0,2,2],dtype='<u8').tofile(mapping)
            order=root/'rows.order';np.array([0,2,1,3],dtype='<u8').tofile(order)
            cfg=Namespace(training_dir=training,output_dir=root,native_library=library,map_file=mapping,
                order_file=order,policy='joint',threshold=.99,norm_ratio=.99,max_block_bytes=16,chunk_rows=2,
                worker_report=root/'worker.json')
            cfg.case='heldout-lexical';cfg.batch_rows=2;cfg.cache_bytes=32;cfg.repeat=2
            for encoding in ('fp32','fp16','int8'):
                cfg.encoding=encoding;cfg.bundle=root/encoding
                build(cfg);verify(cfg)
                result=json.loads(cfg.worker_report.read_text())
                self.assertEqual(result['all_rows_verified'],4)
                self.assertEqual(result['overall']['merged_rows'],2)
                self.assertTrue(result['all_encoded_and_decoded_original_ids_byte_exact_to_representative'])
                query(cfg)
                self.assertTrue(json.loads(cfg.worker_report.read_text())['all_first_and_second_query_bytes_exact'])
            # A valid idempotent but incorrect oracle must fail byte verification.
            np.array([0,1,2,3],dtype='<u8').tofile(mapping)
            with self.assertRaisesRegex(AssertionError,'representative bytes differ'):
                verify(cfg)


if __name__=='__main__':
    unittest.main()
