# Native order sweep — synthetic interface and locality controls, 2026-10-18

**Random weight tables in this report are controls only.** They verify original
hash IDs and byte-exact gather, and expose low-redundancy overhead. They do not
model co-trained prefix/suffix embeddings or decide the learned compression
hypothesis. Read the [trained experiment](trained_ngram_similarity_20261018.md)
for the primary evidence about shared training.

[Raw control report](native_ngram_evidence_20261018/order-control.json) |
[provenance](native_ngram_evidence_20261018/provenance.json) |
[tests](native_ngram_evidence_20261018/tests.txt) |
[ASan/UBSan](native_ngram_evidence_20261018/sanitizers.txt)

## Separate key locality from vector storage

Front coding measures sorted n-gram token-key bytes, with exact decoding. It
is an offline encoding, not an implemented random-access trie/query backend.
Each maximum adds two heads/order at fixed head capacities 257/263, dim16
float32. Workloads contain diverse token IDs or shared-prefix phrase templates.

| Max order | Diverse keys front-coded/raw | Shared-prefix keys front-coded/raw | Random native bytes/raw | Diverse batch rows | Native p50 µs | Python p50 µs | Tensor p50 µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1.123 | 1.345 | 1.090 | 256 | 129.7 | 286.5 | 10.3 |
| 3 | 1.081 | 1.159 | 1.080 | 512 | 222.4 | 347.9 | 5.5 |
| 4 | 1.061 | 1.053 | 1.076 | 768 | 229.4 | 417.1 | 5.0 |
| 6 | 1.041 | 0.929 | 1.073 | 1280 | 376.3 | 679.2 | 5.7 |
| 8 | 1.031 | 0.854 | 1.072 | 1792 | 525.5 | 946.3 | 6.8 |
| 12 | 1.022 | 0.766 | 1.071 | 2816 | 830.5 | 1546.7 | 8.9 |

Shared-prefix key encoding improves at higher orders, while diverse keys still
expand. Random native tables also expand, including `embedding.json`; no row
dedup savings are present. Native gather is faster than the uncached Python
reference but slower than resident tensor gather. Total requests grow with
order, so these timings do not isolate matched-request scaling. Recorded hash-driven query outputs remain byte-exact. Native tests also check
all original rows of their byte-table fixtures. Both workload timings
and p95 are retained in JSON.

Latency is a warm 30-call CPU microbenchmark: no cold I/O, RSS, throughput,
concurrent worker or GPU result. Native and Python byte gathers have no persistent
decoded-block cache in this comparison. High-order key compression, trained
vector similarity, lossless row bytes and query speed are distinct measurements.

## Reproduce and environment

Commit: `edee7b2ad4608db40308a2b3ad801b5ec1b51464`. Official Engram commit:
`fb7f84a21f91223715394a33a1dc24bbfb7f788e`, SHA256 checked by the harness before extracting
unchanged hash/gather definitions. Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4,
Apple M5 / 10 logical CPUs / 32 GiB / macOS 26.6.2 arm64, one torch/interop thread,
Apple clang 21.0.0. Seed7 tokens, per-order seed100+N weights. Local CPU only; no
paid API or rented compute, no monetary/energy estimate.

```bash
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
curl -fL https://raw.githubusercontent.com/deepseek-ai/Engram/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py \
  -o /tmp/engram_demo_v1.py
python tools/benchmark_ngram_native.py \
  --official-source /tmp/engram_demo_v1.py \
  --native-library build/libmeft_rows.dylib \
  --output-dir artifacts/native-ngram-20261018/order-sweep
python -m unittest discover -s tests
python tools/check_native_sanitizers.py
```

The native code compiles as C11 with `-Wall -Wextra -Werror`. All 20 tests
pass with PyTorch/NumPy available, including a tiny synthetic untied causal
decoder with identical logits. ASan/UBSan checks a valid artifact plus 64 seeded
mutated/truncated files. These checks support the tested ABI boundary, not
full-model deployment or broad malformed-input security guarantees.
See the [C ABI and model guide](native_ngram_memory.md).
