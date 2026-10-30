#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /llm-align/duyimin/meft-cpu-capacity-20261030/source-sparse
/llm-align/duyimin/meft-cuda-20261030/compat/bin/python -m tools.benchmark_sparse_blocks --training-dir /llm-align/duyimin/meft-cpu-capacity-20261030/training2 --query-manifest /llm-align/duyimin/meft-cpu-capacity-20261030/capacity/queries-ordinary.json --output-dir /llm-align/duyimin/meft-cpu-capacity-20261030/sparse --native-library /llm-align/duyimin/meft-cpu-capacity-20261030/source-sparse/build/libmeft_rows.so --block-rows 8 32 256 --reuse-baseline /llm-align/duyimin/meft-cpu-capacity-20261030/capacity/native-ordinary.men --repeat 20
