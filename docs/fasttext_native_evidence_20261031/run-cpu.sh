set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/llm-align/duyimin/meft-cpu-capacity-20261030/fusion-20261031
cd "$root/source"
python=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
cpu=$($python -c 'import os; print(min(os.sched_getaffinity(0)))')
nice -n 10 taskset -c "$cpu" "$python" tools/benchmark_fasttext_native.py --enwik8-zip "$root/../enwik8.zip" --output-dir "$root/reproduction" > "$root/reproduction.txt" 2>&1
echo "$?" > "$root/reproduction.exit"
MEFT_FUSION_SANITIZERS=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 nice -n 10 "$python" -m unittest tests.test_fasttext_native_bundle -v > "$root/sanitizers.txt" 2>&1
echo "$?" > "$root/sanitizers.exit"
