#!/bin/bash
set -eu
ROOT=/llm-align/duyimin/ngram-scale-20261031
PY=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -x "$PY"
cd "$ROOT"
git clone source.bundle source > clone.log 2>&1
cd source
test "$(git rev-parse --short=7 HEAD)" = 20d111c
mkdir build
cc -std=c11 -O2 -fPIC -shared native/meft_rows.c -lz -o build/libmeft_rows.so > "$ROOT/build.log" 2>&1
cc --version > "$ROOT/compiler.txt"
nice -n 19 "$PY" -m unittest tests.test_quantized_capacity tests.test_quantized_memory > "$ROOT/tests.log" 2>&1
cat > "$ROOT/run.sh" <<'RUN'
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
RUN
nohup bash "$ROOT/run.sh" > "$ROOT/run.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/run.pid"
cat "$ROOT/tests.log"
cat "$ROOT/run.pid"
