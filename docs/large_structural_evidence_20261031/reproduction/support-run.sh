set -euo pipefail
TASK_DIR=/llm-align/duyimin/ngram-structure-20261031
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$TASK_DIR/support-source"
printf 'running\n' > "$TASK_DIR/support-status"
trap 'printf "failed\n" > "$TASK_DIR/support-status"' ERR
nice -n 19 /llm-align/duyimin/miniconda3/envs/gemma4/bin/python -m tools.audit_training_support \
  --training-dir /llm-align/duyimin/meft-cpu-capacity-20261030/training2 \
  --corpus-zip /llm-align/duyimin/meft-cpu-capacity-20261030/enwik8.zip \
  --report "$TASK_DIR/training-support.json" --sample-per-order 256 --seed 42
printf 'completed\n' > "$TASK_DIR/support-status"
