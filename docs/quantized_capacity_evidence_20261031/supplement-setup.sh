#!/bin/bash
set -eu
ROOT=/llm-align/duyimin/ngram-scale-20261031
PY=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$ROOT/tmp"
test "$(cat "$ROOT/status")" = completed
cd "$ROOT"
git clone auto.bundle source-auto > auto-clone.log 2>&1
cd source-auto
test "$(git rev-parse HEAD)" = 9485969066ee4be466673cde07823330da1a7658
nice -n 19 "$PY" -m unittest tests.test_quantized_memory tests.test_quantized_capacity > "$ROOT/auto-tests.log" 2>&1
cat > "$ROOT/auto-build.py" <<'PY'
import json,sys,time
from pathlib import Path
root=Path('/llm-align/duyimin/ngram-scale-20261031')
sys.path.insert(0,str(root/'source-auto'))
from tools.benchmark_quantized_capacity import software_environment, memory
from tools.quantized_memory import pack_quantized_file
source=Path('/llm-align/duyimin/meft-cpu-capacity-20261030/training2/trained.f32')
report=dict(status='running',**software_environment(),memory_before=memory(),
            label='auto INT8 native/raw bundle selection',shape=[1855907,256])
started,cpu=time.perf_counter(),time.process_time()
metadata=pack_quantized_file(source,root/'auto-int8',256,'int8',native='auto',chunk_rows=4096,max_block_bytes=8192)
report.update(status='passed',metadata=metadata,wall_seconds=time.perf_counter()-started,
              cpu_seconds=time.process_time()-cpu,memory_after=memory(),
              bundle_bytes=sum(p.stat().st_size for p in (root/'auto-int8').iterdir()))
current=json.loads((root/'results/build-int8-raw.json').read_text())['metadata']
for key in ('source_float32_sha256','decoded_float32_sha256','table_sha256'):
 if current[key] != metadata[key]: raise AssertionError('auto/raw differs: '+key)
if metadata['backend'] != 'raw_mmap': raise AssertionError('expected measured INT8 expansion to select raw')
report['auto_matches_verified_raw_source_decoded_and_table_hashes']=True
(root/'auto-build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','wall_seconds','cpu_seconds','memory_after','bundle_bytes','auto_matches_verified_raw_source_decoded_and_table_hashes')}))
PY
cat > "$ROOT/supplement-run.sh" <<'RUN'
#!/bin/bash
set -eu
ROOT=/llm-align/duyimin/ngram-scale-20261031
PY=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$ROOT/tmp"
echo running > "$ROOT/supplement-status"
trap 'echo failed > "$ROOT/supplement-status"' ERR
bash "$ROOT/old-run.sh" > "$ROOT/old-run.log" 2>&1
cd "$ROOT/source-auto"
nice -n 19 "$PY" "$ROOT/auto-build.py" > "$ROOT/auto-run.log" 2>&1
nice -n 19 "$PY" -m tools.benchmark_quantized_capacity --phase verify \
  --source /llm-align/duyimin/meft-cpu-capacity-20261030/training2/trained.f32 \
  --bundle "$ROOT/auto-int8" --dim 256 --encoding int8 --backend raw \
  --worker-report "$ROOT/auto-verify.json" \
  --native-library "$ROOT/source/build/libmeft_rows.so" --chunk-rows 4096 \
  --query-manifest "$ROOT/results/queries.json" > "$ROOT/auto-verify.log" 2>&1
echo completed > "$ROOT/supplement-status"
RUN
nohup bash "$ROOT/supplement-run.sh" > "$ROOT/supplement-run.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/supplement.pid"
cat "$ROOT/auto-tests.log"
cat "$ROOT/supplement.pid"
