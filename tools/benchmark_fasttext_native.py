"""Bounded CPU reproduction: actual legacy training -> lossless native serving.

Needs an existing enwik8.zip; does not download data or use GPUs. This is a custom
positional/source SGNS storage/serving reference, not upstream fastText training
or evidence of downstream task quality. All original input rows are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shlex
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['args','dictionary','productquantizer','matrix','qmatrix','vector','model','utils','fasttext','ftindex']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def tokens(raw):
    # This finite reference ignores XML tags and retains lowercase ASCII words.
    return re.findall(r'[a-z]+',re.sub(r'<[^>]*>',' ',raw.decode('utf-8',errors='replace').lower()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enwik8-zip',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--train-words',type=int,default=65536)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine()!='x86_64':parser.error('bundled libraries require Linux x86-64')
    if args.train_words<64 or args.repeats<1:parser.error('positive repeat and at least 64 training words required')
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    report={'status':'running','category':'public-text, actually trained custom me_fasttext SGNS storage/serving reference',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'exact_command':shlex.join([sys.executable,*sys.argv]),'python':sys.version,'platform':platform.platform(),
        'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
        'cpu':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),''),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'device':'CPU','pytorch':None,'commands':[],
        'limitations':['Finite 64-dim public-text reference; not an Engram checkpoint or downstream quality result.',
                       'Filesystem prewarmed; first pass is reader-cache cold, not disk cold.',
                       'Single worker on a shared server; no GPU or other user process touched.',
                       'Dictionary (including legacy 200 MB word2int allocation) and output matrix remain resident.',
                       'No inference negative-sampling table in either loader; training builds one.',
                       '4 MiB cache payload budget is not an RSS bound; native mmap pages and index remain resident.',
                       'Conversion is offline and copies full input/maps; no bounded exporter memory claim.']}

    def run(command,name):
        command=list(map(str,command)); report['commands'].append(shlex.join(command))
        begin=time.perf_counter()
        with (out/name).open('w') as log:
            result=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        result.check_returncode(); return time.perf_counter()-begin

    try:
        with zipfile.ZipFile(args.enwik8_zip) as archive:raw=archive.read('enwik8')
        split=len(raw)*9//10
        train=tokens(raw[:split])[:args.train_words]; held=tokens(raw[split:])[:8192]
        if len(train)!=args.train_words or len(held)<2048:raise ValueError('insufficient public text')
        corpus=out/'pairs.txt'
        with corpus.open('w') as stream:
            for begin in range(0,len(train)-63,64):
                block=train[begin:begin+64]
                stream.write('p#0 '+' '.join(a+'\x01'+b for a,b in zip(block,block[1:]))+'\n')
        rng=random.Random(42); query=out/'queries.tsv'
        words=[held[rng.randrange(len(held))] for _ in range(1024)]
        sentences=[' '.join(held[i:i+8]) for i in [rng.randrange(len(held)-8) for _ in range(128)]]
        query.write_text(''.join('w\t'+w+'\n' for w in words)+''.join('s\t'+s+'\n' for s in sentences))
        report['data']={'source':'enwik8 (existing archive, no download)','zip_sha256':digest(args.enwik8_zip),
            'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'raw_split_byte':split,
            'transform':'Split raw 90/10 first; strip XML tags, lowercase, [a-z]+; use first requested train words. Each 64-word block emits p#0 and 63 adjacent query\\x01source pairs.',
            'train_words':len(train),'training_pair_tokens':(len(train)//64)*63,'training_lines':len(train)//64,
            'corpus_sha256':digest(corpus),'query_sha256':digest(query),'query_seed':42,
            'heldout_source_words':len(held),'word_queries':1024,'sentence_queries':128,'sentence_length':8}
        del raw,train,held
        run(['make','-C',ROOT,'fasttext-native','native','structural-native'],'build-native.txt')
        executable=out/'probe'
        run(['c++','-std=c++11','-O2','-pthread','-DMEFT_NATIVE_FASTTEXT','-I'+str(ROOT/'src'),
            '-I'+str(ROOT/'native'),'-I'+str(ROOT/'mylib/include'),ROOT/'tools/fasttext_serving_probe.cc',
            *[ROOT/f'src/{n}.cc' for n in SOURCES],ROOT/'native/fasttext_bundle.cc',ROOT/'build/meft_rows_fasttext.o',
            '-L'+str(ROOT/'mylib/lib'),'-ltrie','-lpub','-lz','-o',executable],'build-probe.txt')
        model=out/'trained.bin'
        run([executable,'train',corpus,model,out/'training.json'],'training.txt')
        report['training']=json.loads((out/'training.json').read_text())
        report['source_model_sha256']=digest(model); report['source_model_bytes']=model.stat().st_size
        report['exports']={}
        for mode in ('layout','exact'):
            bundle=out/mode
            seconds=run([sys.executable,ROOT/'tools/export_fasttext_native_cli.py','--model',model,'--output-dir',bundle,
                '--fasttext-binary',ROOT/'build/fasttext-native','--structural-library',ROOT/'build/libmeft_structural.so',
                '--native-library',ROOT/'build/libmeft_rows.so','--mode',mode],'export-'+mode+'.txt')
            manifest=json.loads((bundle/'fasttext.json').read_text())
            report['exports'][mode]={'seconds':seconds,'bundle_bytes':sum(p.stat().st_size for p in bundle.iterdir() if p.is_file()),
                'manifest':manifest}
        report['serving']={}
        oracle=None
        for name in ('dense','layout','exact'):
            path=model if name=='dense' else out/name
            # Explicit warm filesystem baseline; no cache dropping on shared host.
            for file in ([path] if path.is_file() else list(path.iterdir())):
                if file.is_file():digest(file)
            runs=[]
            for repeat in range(args.repeats):
                oracle_path=out/f'{name}-{repeat}.f32'; result_path=out/f'{name}-{repeat}.json'
                run([executable,'serve',path,query,oracle_path,result_path],f'serve-{name}-{repeat}.txt')
                checksum=digest(oracle_path)
                if oracle is None:oracle=checksum
                if checksum!=oracle:raise AssertionError('trained-model inference bytes differ')
                result=json.loads(result_path.read_text()); result['oracle_sha256']=checksum; runs.append(result)
            report['serving'][name]=runs
        report['all_inference_outputs_byte_equal']=True
        report['artifacts_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.tsv','.cc')}
        report['status']='passed'
    except BaseException as exc:
        report['status']='failed'; report['failure']=repr(exc); raise
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'shape':[report['training']['rows'],64],
                      'output_dir':str(out)},indent=2))


if __name__=='__main__':main()
