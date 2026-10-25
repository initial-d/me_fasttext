# Native embedding bags on trained n-gram memory

`MappedEmbeddingBag` avoids constructing the full `[bags, entries, dim]` tensor
before sum/mean reduction. The default `strategy="unique"` gathers each distinct
original query ID once through the native C reader, restores bag membership with
an inverse index, then uses PyTorch's native `embedding_bag` reduction. It stages
`[distinct_query_rows, dim]` plus index buffers; diversity can grow this staging
to the entire table. It does not promise a fixed memory cap.

`strategy="chunked"` instead calls the new C `meft_rows_embedding_bag` ABI. It
stages at most `chunk_rows` embeddings across bag boundaries, accumulates one bag
at a time in FP64 and writes FP32 output. Native gather still sorts requests by
storage block within each chunk. Repeated block decompression can be expensive.
The cap covers staged embeddings, not total process memory, output, mmap,
request bookkeeping, decoded scratch or optional cache.

## Serving comparison

Existing learned character-GRU memory checkpoints, orders 4/8/12, seed 3, dim 64.
The complete chronological held-out memory-ID sequence is repeated cyclically
to make 32768 bags. These are learned weights and held-out queries, not random
embedding vectors; this enlarged replay is not a production trace or a model
throughput measurement. No retraining, query-based pruning or Engram weights.

| Max order | Resident PyTorch bag | Native full gather + mean | Native unique IDs + PyTorch bag | C bag, 1024 rows | C bag, 16384 rows |
|---|---:|---:|---:|---:|---:|
| 4 | 1.065 ms / 224.9 MB | 9.924 ms / 263.2 MB | 8.959 ms / 236.4 MB | 414.358 ms / 229.3 MB | 35.133 ms / 233.4 MB |
| 8 | 1.608 ms / 238.9 MB | 25.636 ms / 314.5 MB | 27.645 ms / 257.1 MB | 1173.660 ms / 245.7 MB | 225.937 ms / 249.3 MB |
| 12 | 1.869 ms / 246.9 MB | 30.416 ms / 357.5 MB | 41.640 ms / 272.3 MB | 1361.474 ms / 252.8 MB | 382.243 ms / 257.1 MB |

Each cell is median query latency / peak process RSS. Unique mode stages
6604/15152/16746 distinct rows for these three workloads. Relative to full native
gather it reduces RSS by 10.2%/18.2%/23.8%; latency changes by -9.7%/+7.8%/+36.9%.
It substantially reduces the chunked path's repeated decoding penalty, but the
resident dense table remains faster in every case. These are serving-memory
tradeoffs, not a new storage-compression ratio or a universal speedup.

Apple M5 CPU / macOS 26.6.2, Python 3.12.14 / PyTorch 2.14.1 / NumPy 2.5.4;
threads 1/1, 2 warmups / 5 repeats, a fresh subprocess per order/mode, warm
allocator/page cache, 4 MiB optional native cache, fixed mode order. RSS includes
framework, inputs, mmap/cache, output and the source checkpoint weight retained
in **every** mode for diagnostics. It is not minimal mapped deployment RSS.
RSS is captured before the separate byte/oracle checks. No GPU or paid services.

## Stored bytes and model quality

Every stored row's bytes agree with the learned source tensor in all native
cases. Maximum sampled FP64-oracle error is <=1.20e-7 for unique/full/resident
reduction and <=5.97e-8 for chunked C reduction. Numeric aggregation is a distinct
contract from byte-preserving storage.

On the original complete test split (13043 next-character targets after common
burn-in), replacing the trained model's memory with the default unique bag path
preserves logits, cross-entropy and top-1 predictions exactly for these three
checkpoints. Character perplexity is 5.0046/5.6575/5.7354. This observed equality
does not establish bit equality across other devices, reduction kernels or
models.

The earlier C-only run is preserved in [chunked-report.json](chunked-report.json)
and [chunked-run.txt](chunked-run.txt), at source `4da855f`. Its FP64 reduction
changed logits by up to 2.39e-6; mean prediction KL was about 1.8e-14 and no top-1
predictions changed. Its much slower small-chunk queries remain evidence.

[report.json](report.json) / [run.txt](run.txt) record all 15 cases and three task
comparisons at source `80a0be4`, exact parent/worker commands, environment,
checkpoint/corpus/split/original-ID hashes, native-library hash, timing samples,
RSS and read/decode counters. Both runs have no recorded failures. Weights,
registry keys and text remain local; regenerate the inputs using the earlier
[trained-model runner](../causal_lm_evidence_20261024/README.md). One seed/corpus
and a small character-GRU do not establish trained Engram or Transformer results.

[Local tests](local-tests.txt): 58 passed. [ASan/UBSan](native-sanitizers.txt):
format versions 1/2 and 128 mutated/truncated files, including C bag calls.
The empty-input lifecycle follow-up is covered separately in
[empty-input-tests.txt](empty-input-tests.txt). Commands and code provenance are
in [local-checks.json](local-checks.json).
