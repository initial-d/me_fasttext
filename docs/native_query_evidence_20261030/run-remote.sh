set -euo pipefail
MEFT_WORK=/llm-align/duyimin/meft-cuda-20261030
cd "$MEFT_WORK/source"
export OMP_NUM_THREADS=1 USE_TF=0 CUDA_VISIBLE_DEVICES=
for order in 4 8 12; do
  "$MEFT_WORK/compat/bin/python" -m tools.benchmark_native_query \
    --fixture "$MEFT_WORK/preparation/fixtures/order-$order-seed-3" \
    --library before="$MEFT_WORK/preparation/native-query/before.so" \
    --library blocks="$MEFT_WORK/preparation/native-query/after.so" \
    --row-library rows="$MEFT_WORK/preparation/native-query/after.so" \
    --output-dir "$MEFT_WORK/preparation/native-query/cpu-policy-order-$order" \
    --cache-bytes 1048576,4194304 --warmup 3 --repeat 10
 done
export CUDA_VISIBLE_DEVICES=7 MEFT_TINYENGRAM_SOURCE="$MEFT_WORK/upstream"
"$MEFT_WORK/compat/bin/python" - <<'PY'
import json,os,platform,sys,subprocess,unittest
from pathlib import Path
import torch
out=Path('/llm-align/duyimin/meft-cuda-20261030/preparation/native-query')
report={'category':'CUDA correctness under shared GPU load, no performance claim', 'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'os':platform.platform(),'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'command':'selected GPU7, allocator fraction .01; unittest device-row + TinyEngram checkpoint suites', 'gpu_before':subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()}
if int(report['gpu_before'].split(',')[1]) < 2048:
 report['status']='blocked';report['reason']='less than 2GiB free for small correctness fixtures'
else:
 torch.cuda.set_per_process_memory_fraction(.01)
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tests.test_device_row_adapter','tests.test_tinyengram_checkpoint'))
 with (out/'cuda-tests.txt').open('w') as stream:
  result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 report.update(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,skips=len(result.skipped),failures=len(result.failures),errors=len(result.errors),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
report['gpu_after']=subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
(out/'cuda-correctness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
PY
"$MEFT_WORK/compat/bin/python" -m tools.benchmark_device_gather \
  --fixture "$MEFT_WORK/preparation/fixtures/order-12-seed-3" \
  --native-library "$MEFT_WORK/preparation/native-query/after.so" \
  --device cuda --cache-policy rows --cache-bytes 4194304 --staging-rows 8192 \
  --output-dir "$MEFT_WORK/preparation/native-query/cuda-gather-probe" --repeat 10 --warmup 3 || true
