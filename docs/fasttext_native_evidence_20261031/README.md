# Actual me_fasttext model → native FastText serving

The native backend now integrates with the original FastText facade, rather
than requiring a separate numeric-row application. A genuinely trained legacy
`.bin` converts through the original DATrie registry and structural planner;
the same CLI and C++ query APIs can load the native bundle. See the
[build/export/query guide](../fasttext_native_bundle.md) and
[PR #36](https://github.com/initial-d/me_fasttext/pull/36).

**This establishes functional fusion and exact inference. Storage and latency
gains are limited on this small model.** The complete native bundle saves only
0.46%; warm queries are slower than resident dense queries. Structural layout
is slightly worse than a matched identity layout here. These are CPU public-text
storage/serving results, not Engram checkpoint or downstream quality evidence.

## Fidelity and historical compatibility

The actual legacy trainer produces **90,380 × 64 float32 input rows**, including
14,813 word/source/position rows and 75,567 registered n-gram rows. Training
changes 88,725 input rows and 75,545 n-gram rows relative to the exact seed-1
initializer. Every input row is retained; unused rows are not filtered out.

Layout and exact exports reconstruct the full source input SHA-256:

`88f2a1a9609eff9b09d0fe7f20cf2bae328511de10684112a118eb521000eaa9`

All nine fresh-process serving runs (dense/layout/exact, three each) produce
the same 1,024 held-out word vectors and 128 held-out eight-word sentence
vectors, with output SHA-256:

`da1b7fb1a0dc0c354308ac99a6c3c008fbf59ee32794cba4de758fa738bc7058`

The **pre-change main** at `bea4302bb7fb1a6c0108fdde141ce667f2389920`, new
dense facade and native facade additionally produce byte-identical outputs on
another 1,024 registered-word and 128 sentence queries. See
[historical-report.json](historical-report.json) and the recorded
[build/run script](historical-run.sh). This control excludes prior OOV
registration and hash-bucket helper bugs; their invalid/out-of-bounds behavior
is intentionally corrected. C++ callers must recompile.

The Linux correctness fixtures separately cover dimensions 7/32, UTF-8/OOV,
all input IDs, ordered repeated IDs, chunks, softmax/HS class scores, failure
preserving the loaded model, immutable guards and dense reload. Fixture rows
are mathematical correctness data, never trained compression evidence.

## Storage and matched structural control

| Same trained source | Input row artifact, bytes | Saving vs raw input |
| --- | ---: | ---: |
| Raw float32 input | 23,137,280 | — |
| Native, identity order | 22,298,237 | 3.63% |
| Native, structural layout/exact | 22,401,608 | 3.18% |

All native controls use the same codec, 32-row / 8,192-byte blocks, 723,040-byte
original-ID forwarding map, source rows and 4 MiB block cache. The identity
control retains the **identical original-ID DASTrie**, model metadata and reader
configuration; only physical row order changes. Its outputs match the same
dense oracle. See [identity-layout-report.json](identity-layout-report.json).

Exact structural sharing finds **zero byte-identical rows to merge**. All
90,380 rows survive mark-compact. Structural order adds 103,371 bytes (0.46%
relative to the identity native table) rather than improving this codec on
this training distribution. This experiment demonstrates that the algorithm
is connected, not extra lossless sharing or a universal row-order advantage.
[Earlier trained structural ablations](../structural_native_evidence_20261030/README.md)
separately measure approximate joint sharing and key-index controls.

The dense `.bin` is 27,204,018 bytes. Exact native **whole-bundle** size is
27,078,261 bytes (0.46% saving), counting input table, 605,232-byte DASTrie,
original dictionary/output metadata, config, manifests and extraction log.
Conversion takes 11.94 seconds, including verification; layout takes 11.61.
These are one export per mode, not a converter throughput distribution.

## End-to-end query costs

Three fresh processes per configuration, with one CPU affinity, prewarmed
filesystem and identical query order. First-pass cache starts empty; each
process then repeats the same trace three more times. Values below are medians
of the three process-level p50s; p95 and all raw passes are in
[trained-report.json](trained-report.json) and [summary.json](summary.json).

| Query / state | Dense .bin | Native exact | Native identity control |
| --- | ---: | ---: | ---: |
| Word first-pass p50 | 2.79 µs | 7.96 µs | 8.10 µs |
| Sentence first-pass p50 | 18.02 µs | 647.29 µs | 511.87 µs |
| Word hot p50 | 2.44 µs | 4.33 µs | 4.33 µs |
| Sentence hot p50 | 16.55 µs | 27.38 µs | 27.45 µs |

Native exact first-pass p95 is 749.40 µs for words and 1,745.41 µs for sentences.
Structural order improves neither cold sentence latency nor warm latency here.
The repeated working set fits the cache, so hot numbers do not represent an
unbounded changing query stream. Timed calls include original word/subword
resolution, float32 row aggregation, sentence normalization and decoding;
oracle file I/O and SHA verification are outside query timing. Batch totals also
include loop/string/oracle-copy overhead and are not kernel throughput.

After the trace, median anonymous RSS is **221.11 MiB dense → 202.26 MiB exact**;
total RSS is **224.88 → 223.55 MiB** because native touches about 21.29 MiB of
file-backed pages. The legacy 200 MB unused `word2int_` allocation and output
matrix remain in both paths. Neither loader allocates training negatives.
Anonymous memory shifts to mapped pages; this is not a large total-memory
reduction. Warm-filesystem median load time is 0.187 → 0.124 seconds.

## Reproduce and inspect failures

Execution source: `bd6bd86eb38c5dec9ac63c35eb7ee3cfc895a2b8`; identity-control
source: `355c299` (adds the ablation tool without changing serving). Python
3.11.15, GCC/G++ 11.4.0, Linux 5.4.0-42 / glibc 2.35, AMD EPYC 7742; no Torch
or GPU used by these drivers. The run is pinned to one available CPU core,
nice level 10, OMP/BLAS/MKL threads 1, on a shared machine. Build commands,
data/model/query hashes, affinity, raw timings and limitations are in the report.

The existing enwik8 archive is split at raw byte 90,000,000 **before** processing.
Each partition strips XML tags, lowercases and keeps `[a-z]+` words. The first
65,536 train words become 1,024 lines of `p#0` plus 63 adjacent
`query\x01source` pairs. This is necessary because this repository's custom
positional/source reader does not train on ordinary whitespace text. The
reference is finite SGNS: dim 64, n-gram lengths 2–12, minCount 2, window 3,
five negatives, two epochs, one thread, learning rate 0.025 and subsampling
threshold 1. It is not ordinary upstream fastText or the paper's large dataset.
Held-out queries sample from the first 8,192 words in the independent 10% raw
partition, seed 42. No quality target or threshold is selected on them.

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
meft_cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
# Existing archive; this script does not download data. Use a fresh directory.
nice -n 10 taskset -c "$meft_cpu" python tools/benchmark_fasttext_native.py \
  --enwik8-zip /path/to/enwik8.zip --output-dir artifacts/fasttext-native
nice -n 10 taskset -c "$meft_cpu" python tools/ablate_fasttext_layout.py \
  --reproduction-dir artifacts/fasttext-native --output-dir artifacts/fasttext-identity

MEFT_FUSION_SANITIZERS=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  python -m unittest tests.test_fasttext_native_bundle -v
```

[tests-initial-failed.txt](tests-initial-failed.txt) preserves the initial
registry-export failure: the DATrie callback returned stop after one key.
Commit `900f548` fixes its continuation convention. The corrected
[Linux roundtrip](tests-registry-fixed.txt) and
[ASan/UBSan/LeakSanitizer run](sanitizers.txt) pass. Bundled trie archives are
linked but uninstrumented. Final CI adds an explicit sanitized facade job,
including HS and zero-vector string helpers. Metadata/index files remain
trusted host-specific artifacts; a cache budget does not bound RSS, and offline
planning copies the full input/maps. No GPU process was touched.
