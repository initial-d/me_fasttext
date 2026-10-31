#!/bin/bash
set -eu
ROOT=/llm-align/duyimin/ngram-scale-20261031
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT/source-old"
test "$(cat "$ROOT/status")" = completed
nice -n 19 /llm-align/duyimin/miniconda3/envs/gemma4/bin/python "$ROOT/old-build.py"
