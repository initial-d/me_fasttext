set -euo pipefail
TASK_DIR=/llm-align/duyimin/ngram-structure-20261031
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$TASK_DIR/source"
printf 'running\n' > "$TASK_DIR/status"
trap 'printf "failed\n" > "$TASK_DIR/status"' ERR
nice -n 19 /llm-align/duyimin/miniconda3/envs/gemma4/bin/python -m tools.benchmark_large_structure run \
  --training-dir /llm-align/duyimin/meft-cpu-capacity-20261030/training2 \
  --output-dir "$TASK_DIR/results" \
  --native-library "$TASK_DIR/source/build/libmeft_rows.so" \
  --structural-library "$TASK_DIR/source/build/libmeft_structural.so" \
  --corpus-zip /llm-align/duyimin/meft-cpu-capacity-20261030/enwik8.zip \
  --encodings fp32,fp16,int8 --thresholds .99,.98,.97,.95,.90 \
  --norm-ratio .99 --sample-per-order 4096 --trace-rows 8192 \
  --batch-rows 512 --rounds 2 --repeat 10 --bag-count 1024 --seed 42
printf 'completed\n' > "$TASK_DIR/status"
