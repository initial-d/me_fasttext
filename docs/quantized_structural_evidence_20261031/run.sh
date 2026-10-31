#!/usr/bin/env bash
set -uo pipefail
cd /llm-align/duyimin/ngram-quantized-20261031/source
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
time nice -n 19 /llm-align/duyimin/meft-cuda-20261030/compat/bin/python -m tools.benchmark_quantized_structural \
 --sgns-dir artifacts/native-ngram-20261018/trained-reference \
 --gru-dir artifacts/causal-lm-compression-20261024 \
 --corpus artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt \
 --native-library build/libmeft_rows.so --structural-library build/libmeft_structural.so \
 --planner native --output-dir /llm-align/duyimin/ngram-quantized-20261031/results \
 --orders 4,8,12 --seeds 3,7,11 --thresholds .99,.98,.97,.95,.90 \
 --norm-ratio .99 --repeat 40 2> /llm-align/duyimin/ngram-quantized-20261031/resources.txt
result=$?
printf '%s\n' "$result" > /llm-align/duyimin/ngram-quantized-20261031/exit.txt
exit "$result"
