#!/bin/bash
set -eu
ROOT=/llm-align/duyimin/ngram-scale-20261031
PY=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT/source"
echo running > "$ROOT/status"
trap 'echo failed > "$ROOT/status"' ERR
nice -n 19 "$PY" -m tools.benchmark_quantized_capacity \
  --training-dir /llm-align/duyimin/meft-cpu-capacity-20261030/training2 \
  --query-manifest /llm-align/duyimin/meft-cpu-capacity-20261030/capacity/queries-ordinary.json \
  --output-dir "$ROOT/results" --native-library "$ROOT/source/build/libmeft_rows.so" \
  --chunk-rows 4096 --max-block-bytes 8192 --cache-bytes 4194304 \
  --repeat 20 --process-repeats 3 --seed 42 --scale-rows 100000 500000 \
  --int8-extra-block-rows 1 4
echo completed > "$ROOT/status"
