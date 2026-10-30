set -eu
ROOT=/llm-align/duyimin/meft-cpu-capacity-20261030
PY=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
test "$(cat "$ROOT/worker2.status")" = completed
cd "$ROOT/source-final"
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir "$ROOT/replay"
mkdir "$ROOT/replay/exact" "$ROOT/replay/joint"
ln -s "$ROOT/capacity/native-ordinary.men" "$ROOT/replay/native-ordinary.men"
for mode in exact joint; do
  ln -s "$ROOT/capacity/native-$mode.men" "$ROOT/replay/$mode/table.men"
  cp "$ROOT/capacity/$mode/embedding.json" "$ROOT/capacity/$mode/structural.json" "$ROOT/capacity/$mode/keys.das" "$ROOT/replay/$mode/"
done
cp "$ROOT/capacity/queries-ordinary.json" "$ROOT/capacity/queries-joint.json" "$ROOT/replay/"
for variant in resident raw-mmap; do
  nice -n 10 "$PY" -m tools.benchmark_memory_capacity --training-dir "$ROOT/training2" --output-dir "$ROOT/replay" --native-library "$ROOT/source-final/build/libmeft_rows.so" --cache-bytes 4194304 --repeat 20 --worker "$variant" --worker-report "$ROOT/replay/$variant.json" > "$ROOT/replay/$variant.log" 2>&1
done
for variant in native-ordinary native-exact native-joint; do
  cp "$ROOT/capacity/$variant.json" "$ROOT/capacity/$variant.log" "$ROOT/replay/"
done
"$PY" - "$ROOT" <<'MEFT_REPORT'
import hashlib,json,sys,subprocess
from pathlib import Path
root=Path(sys.argv[1])
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()
old=json.loads((root/'capacity/report.json').read_text())
old['preparation_source_commit']=old['source_commit']
old['preparation_command']=old['command']
old['preparation_report_sha256']=sha(root/'capacity/report.json')
old['workers']={}
for variant in ('resident','raw-mmap','native-ordinary','native-exact','native-joint'):
    value=json.loads((root/'replay'/f'{variant}.json').read_text())
    assert value['status']=='passed'
    old['workers'][variant]={'returncode':0,'result':value}
old['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
old['command']='bash /llm-align/duyimin/meft-cpu-capacity-20261030/final-replay.sh'
old['replay_note']='Final-source fresh resident/raw-mmap workers use NumPy take directly into the preallocated output. Existing native workers already used direct gather_into, whose query loop/cache/library are unchanged. Each worker retains its actual source commit; prepared source/timings and original preliminary double-copy NumPy controls are preserved separately.'
old['limitations'].append('Some registered training rows never changed; 5014/1860921 are explicitly excluded, with source-ID provenance. Updated does not mean converged.')
files=['enwik8.zip','training2/trained.f32','training2/keys.json','training2/trained-source-ids.u64',
       'training2/queries.u64','capacity/native-ordinary.men','capacity/native-exact.men','capacity/native-joint.men',
       'capacity/exact/keys.das','capacity/exact/keys.flat','capacity/joint/keys.das','capacity/joint/keys.flat']
old['artifact_manifest']={name:{'bytes':(root/name).stat().st_size,'sha256':sha(root/name)} for name in files}
(root/'replay/report.json').write_text(json.dumps(old,indent=2)+'\n')
print(json.dumps({'status':old['status'],'raw_table_bytes':old['raw_table_bytes'],'source_commit':old['source_commit']}))
MEFT_REPORT
