set -euo pipefail
TASK_DIR=/llm-align/duyimin/ngram-structure-20261031
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$TASK_DIR/query-source"
printf 'waiting-primary\n' > "$TASK_DIR/query-status"
trap 'printf "failed\n" > "$TASK_DIR/query-status"' ERR
while test "$(cat "$TASK_DIR/status")" = running; do sleep 10; done
test "$(cat "$TASK_DIR/status")" = completed
printf 'running\n' > "$TASK_DIR/query-status"
nice -n 19 /llm-align/duyimin/miniconda3/envs/gemma4/bin/python -m tools.replay_large_structure_queries \
  --report "$TASK_DIR/results/report.json" --output-dir "$TASK_DIR/query-corrected"
printf 'completed\n' > "$TASK_DIR/query-status"
