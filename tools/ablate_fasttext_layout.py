"""Reuse a completed CPU trained-model reproduction for matched row-order control.

Same source row bytes, original-ID index, forwarding map size, block cap/cache,
codec and serving probe. Only structural layout is replaced by identity order.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.native_row_store import pack_native
from tools.benchmark_fasttext_native import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduction-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); source=args.reproduction_dir.resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    reference=json.loads((source/'report.json').read_text())
    if reference['status']!='passed':raise ValueError('need a completed lossless reproduction')
    shape=reference['exports']['exact']['manifest']['shape']; rows,dim=shape
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    report={'status':'running','category':'matched identity-row-order control on actual trained me_fasttext input',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_probe_commit':reference['commit'],'python':sys.version,'platform':platform.platform(),
        'exact_command':shlex.join([sys.executable,*sys.argv]),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'source_model_sha256':digest(source/'trained.bin'),'reference_report_sha256':digest(source/'report.json'),
        'limitations':reference['limitations'],'commands':[]}
    try:
        command=[str(ROOT/'build/fasttext-native'),'dump-native',str(source/'trained.bin'),str(out/'extracted')]
        report['commands'].append(shlex.join(command))
        with (out/'extract.txt').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        raw=out/'extracted/input.f32'
        if digest(raw)!=reference['exports']['exact']['manifest']['source_rows_sha256']:raise AssertionError('different original rows')
        bundle=out/'identity'; bundle.mkdir()
        begin=time.perf_counter()
        storage=pack_native(raw,bundle/'table.men',dim*4,32,deduplicate=False,predictor='xor',
                            row_order=range(rows),representatives=range(rows),max_block_bytes=8192)
        report['pack_seconds']=time.perf_counter()-begin; report['storage']=storage
        for name in ['keys.das','model.meta','reader.cfg']:shutil.copyfile(source/'exact'/name,bundle/name)
        report['same_index_sha256']=digest(bundle/'keys.das'); report['same_metadata_sha256']=digest(bundle/'model.meta')
        report['same_reader_config_sha256']=digest(bundle/'reader.cfg')
        # Structural metadata is not needed by the C++ facade. Compare payload,
        # identical model metadata/index/config separately from JSON manifests.
        report['serving_payload_bytes']=sum(p.stat().st_size for p in bundle.iterdir())
        for p in bundle.iterdir():digest(p)
        runs=[]
        for repeat in range(len(reference['serving']['exact'])):
            oracle=out/f'identity-{repeat}.f32'; result=out/f'identity-{repeat}.json'
            command=list(map(str,[source/'probe','serve',bundle,source/'queries.tsv',oracle,result]))
            report['commands'].append(shlex.join(command))
            with (out/f'serve-{repeat}.txt').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            checksum=digest(oracle)
            if checksum!=reference['serving']['dense'][0]['oracle_sha256']:raise AssertionError('different inference bytes')
            record=json.loads(result.read_text()); record['oracle_sha256']=checksum; runs.append(record)
        report['serving']=runs; report['status']='passed'
    except BaseException as exc:
        report['status']='failed'; report['failure']=repr(exc); raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'artifact_bytes':report['storage']['artifact_bytes']},indent=2))


if __name__=='__main__':main()
