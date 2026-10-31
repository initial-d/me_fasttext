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
