# Query ID compaction without full-batch sorting

The previous unique-query bag path sorted every queried ID. Repeated high-order
queries made that cost large enough to erase the gather/reduction savings.
`unique_method="auto"` now builds a histogram and an original-ID lookup table
when their combined allocation fits 8 MiB and the table has at most four rows
per query entry. Distinct original IDs remain sorted, and the inverse index
restores every occurrence in its original bag. Empty, sparse or over-budget
queries retain `torch.unique`; `unique_method="sort"` preserves that baseline.

The budget only covers the histogram and lookup table. Unique IDs, inverse
indices, an optional unused padding row, staged embeddings, C buffers/cache and
output still consume memory. No table rows are dropped based on a query trace.
The reduction kernel, padding behavior, stored bytes and learned weights stay
the same. This is a Python/PyTorch query-planning improvement over the C gather
API; the bounded C bag reducer remains available separately.

## Controlled serving comparison

Existing learned character-GRU memory, max orders 4/8/12, seed 3, dim 64.
32768 bags from cyclic replay of the complete chronological held-out ID stream;
no random weights, retraining, query pruning or new data download. Source
`8f39b1d`, Apple M5 / macOS 26.6.2, Python 3.12.14 / PyTorch 2.14.1 / NumPy 2.5.4,
CPU threads 1/1, 2 warmups / 5 repeats, fresh process per order/mode, 4 MiB cache.

| Max order | Resident PyTorch bag | Native full gather | Sorted unique IDs | Automatic compaction |
|---|---:|---:|---:|---:|
| 4 | 1.416 ms / 224.9 MB | 15.041 ms / 263.3 MB | 14.213 ms / 236.2 MB | 8.790 ms / 232.7 MB |
| 8 | 2.193 ms / 238.6 MB | 32.070 ms / 314.5 MB | 35.516 ms / 257.2 MB | 23.472 ms / 255.6 MB |
| 12 | 2.350 ms / 246.9 MB | 37.503 ms / 356.6 MB | 53.174 ms / 272.3 MB | 27.983 ms / 266.9 MB |

Cells are median query latency / peak process RSS. All three automatic cases
selected dense compaction. Compared with sorted unique, observed query speedups
are **1.62x / 1.51x / 1.90x**. Compared with full native gather, speedups are
**1.71x / 1.37x / 1.34x** with **11.6% / 18.7% / 25.1% less peak RSS**. Resident
dense PyTorch remains faster in every case. Storage compression ratios are
unchanged. The slow chunked C cases remain in the raw report.

Every native stored row is byte-exact. Serving output hashes agree across
resident, sorted unique and automatic compaction at each order. The complete
original test task (13043 targets) preserves logits, cross-entropy, perplexity
and top-1 predictions exactly in all three models. This observed equality is
for these CPU reference checkpoints; it does not establish trained Engram or
Transformer behavior, arbitrary-device bit equality or full-model throughput.

RSS is captured before byte/oracle checks. All modes retain source checkpoint
weights as a common diagnostic baseline; RSS includes the framework, input,
output, allocator caching and mmap/native cache. It is not minimal deployment
RSS. Fixed mode order, warm allocator/cache, one CPU/seed/corpus and repeated
held-out IDs limit generalization; timings are compared within this run rather
than to the earlier day's snapshot. No GPU or paid services were used.

[report.json](report.json) records all 18 cases/three task comparisons, source
commit, exact parent/worker commands, environment, corpus/split/checkpoint/query
hashes, samples, peak RSS and native counters; no failures. [run.txt](run.txt)
preserves output. Text and weights remain local and can be regenerated through
the [trained-model runner](../causal_lm_evidence_20261024/README.md).
[local-tests.txt](local-tests.txt): all 59 tests passed, including dense/sort
equality, sparse fallback, zero/insufficient/exact budgets, original IDs and the
existing padding/dtype tests. C is unchanged from the preceding sanitized PR.
Commands/provenance: [local-checks.json](local-checks.json).
