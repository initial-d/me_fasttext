"""Replay recorded scale query commands in fresh workers after harness fixes.

Keep the primary report immutable. New report retains original build/plan/verify
evidence and replaces query workers explicitly, with both source fingerprints.
"""
import argparse
import json
from pathlib import Path
import shlex
import random
import subprocess
import sys
import time

from .benchmark_large_structure import write_json
from .prepare_public_benchmark import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    cfg=p.parse_args()
    report=json.loads(cfg.report.read_text())
    if report['status'] not in ('passed','builds-completed') or any(w['returncode'] for w in report['workers']):
        raise ValueError('primary experiment must complete successfully first')
    cfg.output_dir.mkdir(parents=True,exist_ok=False)
    queries=[w for w in report['workers'] if w['phase']=='query']
    if report['status']=='builds-completed':
        existing={w['name'] for w in queries};queries=[]
        settings=report['provenance']['settings'];rng=random.Random(settings['seed'])
        for round_id in range(settings['rounds']):
            jobs=[(v,case) for v in report['variants'] for case in ('heldout-lexical','uniform-id-stress')]
            rng.shuffle(jobs)
            for variant,case in jobs:
                name=f"query-{variant['name']}-{case}-{round_id}"
                if name in existing:
                    continue
                options=dict(variant['options'],case=case,training_dir=settings['training_dir'],
                    output_dir=settings['output_dir'],native_library=settings['native_library'],
                    structural_library=settings['structural_library'],worker_report='replaced-below',
                    batch_rows=settings['batch_rows'],cache_bytes=settings['cache_bytes'],repeat=settings['repeat'])
                command=[sys.executable,'-m','tools.benchmark_large_structure','query']
                for k,v in options.items():
                    if v is not None:command.extend(['--'+k.replace('_','-'),str(v)])
                queries.append(dict(name=name,command=shlex.join(command)))
    else:
        report['workers']=[w for w in report['workers'] if w['phase']!='query']
    report['query_replay']=dict(primary_report_sha256=file_sha256(cfg.report),
        primary_report=str(cfg.report),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_file_sha256=file_sha256(Path(__file__).with_name('benchmark_large_structure.py')),
        command=shlex.join(getattr(sys,'orig_argv',[sys.executable,*sys.argv])),
        correction='numeric workers no longer parse unused keys or retain the full offline representative map after oracle construction; first/second output bytes remain checked',
        primary_workers_preserved_in_original_report=True,query_workers=len(queries))
    report['status']='replay-running'
    root=Path(__file__).resolve().parents[1]
    for worker in queries:
        command=shlex.split(worker['command'])
        if command[1:4]!=['-m','tools.benchmark_large_structure','query']:
            raise ValueError('unexpected recorded worker command')
        command[0]=sys.executable
        output=cfg.output_dir/(worker['name']+'.json');log=cfg.output_dir/(worker['name']+'.log')
        command[command.index('--worker-report')+1]=str(output)
        started=time.perf_counter()
        with log.open('w') as stream:
            process=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        entry=dict(phase='query',name=worker['name'],command=shlex.join(command),returncode=process.returncode,
            wall_seconds=time.perf_counter()-started,log_sha256=file_sha256(log),
            result=json.loads(output.read_text()) if output.exists() else None)
        report['workers'].append(entry)
        write_json(cfg.output_dir/'report.json',report)
        print(json.dumps(dict(name=worker['name'],returncode=process.returncode,seconds=entry['wall_seconds'])),flush=True)
        if process.returncode:
            raise RuntimeError('replay failed; inspect '+str(log))
    report['status']='passed'
    write_json(cfg.output_dir/'report.json',report)


if __name__=='__main__':
    main()
