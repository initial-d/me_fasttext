"""Fresh-process CPU capacity/lookup controls on an actually trained table.

Includes resident and raw mmap controls. Full verification is separate from
timed workers, whose oracle is per-query hashes. OS page cache is warm; process
RSS includes resident file pages and is not a cache-payload budget. No GPU use.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

import numpy as np

from .prepare_public_benchmark import file_sha256
from .native_row_store import NativeRows, pack_native


def memory():
    result = {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
    for name in ('status', 'smaps_rollup'):
        path = Path('/proc/self') / name
        if path.exists():
            for line in path.read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('VmRSS', 'VmSize', 'Rss', 'Pss', 'Private_Clean', 'Private_Dirty', 'Shared_Clean', 'Shared_Dirty', 'Anonymous'):
                    result[key + '_bytes'] = int(value.strip().split()[0]) * 1024
    return result


def environment():
    cpu = platform.processor()
    path = Path('/proc/cpuinfo')
    if path.exists():
        cpu = next((line.split(':',1)[1].strip() for line in path.read_text().splitlines() if line.startswith('model name')), cpu)
    return {'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(), 'cpu': cpu,
            'command': shlex.join([sys.executable, *sys.argv]),
            'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}


def samples(values):
    return {'p50_ms':float(np.median(values)), 'p95_ms':float(np.percentile(values,95)), 'samples_ms':values}


def table_path(directory, variant):
    if variant in ('native-exact', 'native-joint'):
        return directory/variant.removeprefix('native-')/'table.men'
    return directory/(variant+'.men')


def worker(args):
    label = 'joint' if args.worker == 'native-joint' else 'ordinary'
    manifest = json.loads((args.output_dir/('queries-'+label+'.json')).read_text())
    source = args.training_dir/'trained.f32'
    shape = tuple(manifest['shape'])
    width = shape[1]*4
    report = {'status':'running', **environment(), 'variant':args.worker,
              'cache_budget_bytes':args.cache_bytes if args.worker.startswith('native') else 0,
              'memory_before_open':memory(), 'cases':[]}
    started = time.perf_counter()
    if args.worker == 'resident':
        table = np.fromfile(source,dtype='<f4').reshape(shape)
    elif args.worker == 'raw-mmap':
        table = np.memmap(source,mode='r',dtype='<f4',shape=shape)
    else:
        table = NativeRows(table_path(args.output_dir,args.worker),args.native_library,
                           cache_bytes=args.cache_bytes,cache_policy='rows')
    report['open_ms'] = (time.perf_counter()-started)*1000
    report['memory_after_open'] = memory()
    try:
        for case in manifest['cases']:
            queries = []
            for query in case['queries']:
                ids = np.asarray(query['ids'],dtype='<u8')
                out = ctypes.create_string_buffer(len(ids)*width)
                queries.append((ids,out,query['sha256']))
            if isinstance(table,NativeRows):
                table.set_cache_bytes(args.cache_bytes)

            def gather(item):
                ids,out,_ = item
                if isinstance(table,NativeRows):
                    table.gather_into(ids.ctypes.data,len(ids),ctypes.addressof(out),len(out))
                else:
                    result = table[ids]
                    ctypes.memmove(ctypes.addressof(out),result.ctypes.data,result.nbytes)

            def check(item):
                if hashlib.sha256(item[1].raw).hexdigest() != item[2]:
                    raise AssertionError('query bytes differ from the precomputed trained-row oracle')

            first = []
            start_memory = memory()
            for item in queries:
                started = time.perf_counter_ns()
                gather(item)
                first.append((time.perf_counter_ns()-started)/1e6)
                check(item)
            first_memory = memory()
            hot = queries[:min(8,len(queries))]
            # No large gather result survives outside these bounded buffers.
            for item in hot:
                gather(item)
            warm = []
            cpu = []
            for _ in range(args.repeat):
                for item in hot:
                    started,cpu_started = time.perf_counter_ns(),time.process_time_ns()
                    gather(item)
                    warm.append((time.perf_counter_ns()-started)/1e6)
                    cpu.append((time.process_time_ns()-cpu_started)/1e6)
                    check(item)
            report['cases'].append({'name':case['name'],'query_rows':sum(len(q[0]) for q in queries),
                'distinct_rows':len(set(int(i) for q in queries for i in q[0])),
                'first_trace':samples(first),'hot_repeated':samples(warm),
                'hot_cpu':samples(cpu),'all_query_bytes_exact':True,
                'memory_before_queries':start_memory,'memory_after_first_trace':first_memory,
                'memory_after_hot':memory(),
                'cache_payload_bytes':table.cache_bytes if isinstance(table,NativeRows) else 0})
            report['cases'][-1]['query_output_buffer_bytes'] = sum(len(item[1]) for item in queries)
        report['memory_final'] = memory()
        report['status'] = 'passed'
    finally:
        if isinstance(table,NativeRows):
            table.close()
    args.worker_report.write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--native-library',type=Path,required=True)
    parser.add_argument('--structural-library',type=Path)
    parser.add_argument('--block-rows',type=int,default=256)
    parser.add_argument('--cache-bytes',type=int,default=4194304)
    parser.add_argument('--repeat',type=int,default=20)
    parser.add_argument('--query-rows',type=int,default=512)
    parser.add_argument('--worker',choices=('resident','raw-mmap','native-ordinary','native-exact','native-joint'))
    parser.add_argument('--worker-report',type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('set CUDA_VISIBLE_DEVICES to empty')
    if args.worker:
        worker(args)
        return
    args.output_dir.mkdir(parents=True,exist_ok=False)
    training = json.loads((args.training_dir/'training.json').read_text())
    source = args.training_dir/'trained.f32'
    if training['status'] != 'passed' or not training['all_rows_changed'] or file_sha256(source) != training['trained_sha256']:
        raise ValueError('need verified, finite, genuinely trained original rows')
    shape = (training['ngram_rows'],training['dim'])
    rows = np.memmap(source,mode='r',dtype='<f4',shape=shape)
    report = {'status':'running', **environment(), 'category':'public-text finite SGNS CPU capacity reference',
              'training_report_sha256':file_sha256(args.training_dir/'training.json'),
              'training':training, 'raw_table_bytes':source.stat().st_size,
              'cost':'Existing shared CPU; no GPU, new rented compute or paid API. Monetary/energy cost unmeasured.',
              'native_library_sha256':file_sha256(args.native_library), 'variants':{},
              'limitations':['Resident NumPy and raw mmap are mandatory controls; generic mmap savings are not algorithm-specific.',
                             'Fresh workers, shared host, warm OS page cache. No disk-cold or statistical significance claim.',
                             'RSS includes mapped file residency; bounded cache payload excludes metadata/scratch/output/pages.',
                             'No enforced process memory cap, OOM experiment, GPU memory or full-model inference.',
                             'Joint compaction is approximate, with vector error only; no task-quality evaluation.',
                             'One seed/order and finite nonconverged training; no learned DeepSeek checkpoint or paper reproduction.']}
    oracle = {'ordinary':None, 'exact':None}
    started = time.perf_counter()
    ordinary = pack_native(source,args.output_dir/'native-ordinary.men',shape[1]*4,args.block_rows,False,'xor')
    report['variants']['native-ordinary'] = {'storage':ordinary,'export_seconds':time.perf_counter()-started,'export_memory':memory()}
    if args.structural_library:
        from .structural_native import StructuralPlan
        report['structural_library_sha256'] = file_sha256(args.structural_library)
        keys = dict(enumerate(json.loads((args.training_dir/'keys.json').read_text())))
        for mode in ('exact','joint'):
            started = time.perf_counter()
            with StructuralPlan(source.read_bytes(),shape[1],keys,args.structural_library,mode=mode,cosine=.99,norm_ratio=.99) as plan:
                directory = args.output_dir/mode
                metadata = plan.export(source,directory,block_rows=args.block_rows)
                plan.save_index(directory/'keys.flat',index='flat')
                report['variants']['native-'+mode] = {'storage':metadata,'export_seconds':time.perf_counter()-started,
                    'index_bytes':{'flat':(directory/'keys.flat').stat().st_size,'datrie':(directory/'keys.das').stat().st_size},
                    'merged_rows':sum(i!=r for i,r in enumerate(plan.representatives)), 'export_memory':memory()}
                if mode == 'joint':
                    oracle['joint'] = np.asarray(plan.representatives,dtype=np.int64)
                    maximum, squared, original_squared = 0.,0.,0.
                    for begin in range(0,len(rows),4096):
                        a = np.asarray(rows[begin:begin+4096],dtype=np.float64)
                        b = np.asarray(rows[oracle['joint'][begin:begin+4096]],dtype=np.float64)
                        error = ((a-b)**2).sum(1)
                        norms = (a*a).sum(1)
                        maximum = max(maximum,float(np.sqrt(error/np.maximum(norms,1e-30)).max()))
                        squared += float(error.sum())
                        original_squared += float(norms.sum())
                    report['variants']['native-joint']['vector_error'] = {'max_relative_l2':maximum,
                        'global_relative_l2':float(np.sqrt(squared/max(original_squared,1e-30)))}
    # Check every decoded original ID in a separate, untimed preparation process.
    for label in report['variants']:
        mapping = oracle[label.removeprefix('native-')]
        digest = hashlib.sha256()
        with NativeRows(table_path(args.output_dir,label),args.native_library) as table:
            for begin in range(0,len(rows),4096):
                ids = np.arange(begin,min(begin+4096,len(rows)))
                expected = rows[ids if mapping is None else mapping[ids]].tobytes()
                actual = table.rows(ids.tolist())
                if actual != expected:
                    raise AssertionError('full row reconstruction mismatch')
                digest.update(actual)
        report['variants'][label]['all_original_ids_exact_to_representatives'] = True
        report['variants'][label]['original_bytes_preserved'] = mapping is None
        report['variants'][label]['reconstructed_sha256'] = digest.hexdigest()
    traces = [('heldout-lexical',np.fromfile(args.training_dir/'queries.u64',dtype='<u8')),
              ('uniform-id-stress',np.random.default_rng(42).choice(len(rows),min(65536,len(rows)),replace=False).astype('<u8'))]
    # The stress trace queries genuine trained rows; only its access pattern is
    # synthetic. It is not substituted for a language-model query distribution.
    manifest = {'shape':shape,'cases':[]}
    for label in ('ordinary','exact','joint'):
        if label not in oracle:
            continue
        manifest['cases'] = []
        for name,trace in traces:
            queries = []
            for begin in range(0,len(trace),args.query_rows):
                ids = trace[begin:begin+args.query_rows]
                expected = rows[ids if oracle[label] is None else oracle[label][ids]].tobytes()
                queries.append({'ids':ids.tolist(),'sha256':hashlib.sha256(expected).hexdigest()})
            manifest['cases'].append({'name':name,'queries':queries})
        (args.output_dir/('queries-'+label+'.json')).write_text(json.dumps(manifest)+'\n')
        variants = ['resident','raw-mmap','native-ordinary'] if label=='ordinary' else ['native-'+label]
        if label=='exact' and 'native-exact' not in report['variants']:
            continue
        for variant in variants:
            worker_report = args.output_dir/(variant+'.json')
            cmd = [sys.executable,'-m','tools.benchmark_memory_capacity','--training-dir',str(args.training_dir),
                   '--output-dir',str(args.output_dir),'--native-library',str(args.native_library),
                   '--cache-bytes',str(args.cache_bytes),'--repeat',str(args.repeat),
                   '--worker',variant,'--worker-report',str(worker_report)]
            result = subprocess.run(cmd,capture_output=True,text=True)
            (args.output_dir/(variant+'.log')).write_text(result.stdout+result.stderr)
            report.setdefault('workers',{})[variant] = {'returncode':result.returncode}
            if worker_report.exists():
                report['workers'][variant]['result'] = json.loads(worker_report.read_text())
            (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if result.returncode:
                raise RuntimeError('worker failed: '+variant)
    report['status'] = 'passed'
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'shape':shape,'raw_bytes':source.stat().st_size}),flush=True)


if __name__ == '__main__':
    main()
