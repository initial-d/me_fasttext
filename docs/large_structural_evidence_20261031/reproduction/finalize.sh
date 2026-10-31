set -euo pipefail
TASK_DIR=/llm-align/duyimin/ngram-structure-20261031
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf 'running\n' > "$TASK_DIR/final-status"
trap 'printf "failed\n" > "$TASK_DIR/final-status"' ERR
nice -n 19 /llm-align/duyimin/miniconda3/envs/gemma4/bin/python "$TASK_DIR/finalize.py"
printf 'completed\n' > "$TASK_DIR/final-status"
