#!/bin/bash
set -eu
ROOT=/llm-align/duyimin/ngram-scale-20261031
PY=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$ROOT/tmp"
export TMPDIR="$ROOT/tmp"
cd "$ROOT"
git clone checks.bundle checks-source > checks-clone.log 2>&1
cd checks-source
nice -n 19 "$PY" -m unittest discover -s tests > "$ROOT/final-tests.log" 2>&1
tail -8 "$ROOT/final-tests.log"
