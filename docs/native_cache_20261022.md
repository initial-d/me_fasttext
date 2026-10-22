# Bounded decoded-block cache for native queries — 2026-10-22

The C reader can now reuse decoded blocks between calls with an explicit byte
budget. It remains disabled by default. Files, original row IDs, dtype bits and
outputs are unchanged; the tradeoff is extra decoded memory for fewer zlib/XOR
decodes. On a short replay of original character-ngram corpus IDs, a 2 MiB
cache improves warm p50 about **6.9× at maximum order 4 and 1.7× at orders 8/12**.
A 256 KiB cache has no hits on that trace and is slightly slower. These are
small trained SGNS rows and controlled repeated traces, not Engram serving evidence.

Raw evidence: [all nine tables and three cache budgets](native_cache_evidence_20261022/trained-cache.json),
[initial small-cache pilot](native_cache_evidence_20261022/small-cache-pilot.json),
[official full module](native_cache_evidence_20261022/official-forward.json),
[factory module](native_cache_evidence_20261022/factory-forward.json),
[tests](native_cache_evidence_20261022/tests.txt),
[sanitizers](native_cache_evidence_20261022/sanitizers.txt), and
[provenance](native_cache_evidence_20261022/provenance.json).

## Contract and configuration

The additive C ABI exposes `meft_rows_set_cache_bytes`, `meft_rows_cache_bytes`
and `meft_rows_last_cache_hits`. Opening still disables persistent caching.
Setting any budget clears cached entries, including when the budget is unchanged.
Each immutable single-worker handle owns an LRU of at most 64 decoded compressed
blocks. Raw blocks are read directly from mmap; oversized compressed blocks use
scratch without caching. Only successful decompression and XOR reconstruction
are cached. Optional cache allocation failure leaves a valid scratch result.

Eviction happens before allocation, so cached payload bytes never exceed the
budget, including insertion. The payload counter does **not** include the fixed
64-entry metadata (about 1.5 KiB on 64-bit targets), allocator overhead, existing
maximum-block scratch buffer, per-query locations/output, mmap/page cache or
process RSS. There is no full-table allocation at open. One handle per worker
is still required, and callers must keep its mapped file immutable.

Batch grouping still decodes each requested block at most once per call. A cache
hit is counted once per requested block, not per repeated row. `last_block_reads`
now counts blocks accessed from the artifact on cache misses, including raw
blocks; hits are separate. With cache disabled, existing counter behavior stays
the same. The byte budget is a limit, not a promise of full capacity utilization:
the independent 64-entry cap can limit larger budgets.

```python
from tools.native_row_store import NativeRows
from tools.engram_row_adapter import MappedMultiHeadEmbedding, open_demo_embedding

with NativeRows("table.men", "build/libmeft_rows.so", cache_bytes=2 * 1024**2) as table:
    vectors = table.rows([0, 42, 0])
    vectors = table.rows([0, 42, 0])
    print(table.last_cache_hits, table.last_decompressions, table.cache_bytes)
    table.set_cache_bytes(0)  # Frees all entries and disables persistent reuse.

# The existing CPU gather interface and exported table need no changes.
mapped = MappedMultiHeadEmbedding("/path/to/layer", native_library="build/libmeft_rows.so",
                                 native_cache_bytes=2 * 1024**2)
# open_demo_embedding and MappedEmbedding accept native_cache_bytes too.
# Explicitly close mapped when finished; choose budget from measured workload.
```

`cache_blocks` still configures only the Python backend. New `native_cache_bytes`
defaults to zero and requires a native artifact. Rebuild the C library before
enabling this feature. Zero-budget loads work with older compiled libraries;
requesting caching from one gives a clear rebuild error and closes the handle.

## Traces and results

The same nine hashed learned float32 tables from the
[SGNS study](trained_ngram_similarity_20261018.md) were packed with lexical
order, no deduplication, 64-row blocks and adaptive XOR. All original rows were
verified through C for each setting. No new training or external data download.
Three traces isolate reuse on the learned rows:

- `corpus_words`: the first 128 retained words in original corpus order, eight
  words per call, enumerated through the original 2..N character-ngram function.
  OOV words and standalone word rows are excluded. Each of 16 calls requests
  60–123 rows at order 4, 65–185 at 8, and 65–188 at 12.
- `uniform_table`: 16 batches of 64 numeric IDs sampled uniformly across the
  full table. These are random IDs over trained weights, not random embeddings.
- `hot_8_blocks`: 16 batches of 64 IDs confined to eight 32 KiB decoded blocks.
  This explicitly favorable locality control fits exactly in 256 KiB.

Queries preserve order and duplicates. Each mode starts with an empty cache,
warms the entire trace three times, then times five additional passes over the
same trace. Input/output are preallocated; only ctypes dispatch and C gather
are timed. Byte checking and counters occur outside the timed interval. Each
run retains 80 samples per mode/trace, p50/p95, hit/decode counts and peak cached
payload bytes. The seed-7 mode order is reversed. Data are filesystem-warm;
this is not a cold-cache, held-out or production request distribution.

The table reports the median of each seed's p50, in microseconds, across three
seeds. The raw report preserves every sample rather than claiming significance.

| Max order | Trace | Cache off p50 µs | 256 KiB p50 µs | 2 MiB p50 µs |
| --- | --- | ---: | ---: | ---: |
| 4 | Corpus words | 1,511.63 | 1,518.44 | 217.81 |
| 8 | Corpus words | 2,086.71 | 2,109.85 | 1,248.21 |
| 12 | Corpus words | 2,256.15 | 2,279.54 | 1,304.81 |
| 4 | Uniform table | 2,400.44 | 2,421.42 | 967.79 |
| 8 | Uniform table | 3,029.88 | 3,058.33 | 2,278.69 |
| 12 | Uniform table | 3,084.60 | 3,085.42 | 2,280.38 |
| 4 | Hot eight blocks | 426.83 | 1.75 | 1.75 |
| 8 | Hot eight blocks | 436.65 | 1.75 | 1.75 |
| 12 | Hot eight blocks | 435.98 | 1.75 | 1.75 |

The hotspot result is a deliberately favorable warm-cache control; its roughly
244–249× gather ratio is **not a real model speedup**. Once warm, all eight blocks
hit and measured decompressions are zero. Both budgets retain only 256 KiB there.
On corpus/uniform traces the 256 KiB cache has zero measured hits and can add
about 1% overhead. Larger random batches exceed its capacity, and sorted block
processing can evict useful blocks before reuse. Increasing capacity is helpful
in this fixed replay, but a changing real trace may behave differently.

For corpus traces, summed measured decodes across three seeds fall from
6,975/9,435/9,960 to 1,080/5,265/6,075 at orders 4/8/12 with 2 MiB. Actual cached
payload never exceeds its configured budget. Larger order raises lookup fan-out
and working-set size here; it does not guarantee more cache benefit. There is
no resident tensor/raw-mmap timing baseline, process RSS, GPU transfer, task
quality or end-to-end model speed measurement in this run.

## Verification and reproduction

35 tests pass with PyTorch; system Python skips 17 optional tests. New checks
cover repeated reuse, LRU eviction, actual tail-block size, undersized budgets,
reset/disable, entry cap, raw bypass, failed decode not cached, invalid-query
recovery, old-library behavior and float32/float16/bfloat16 special bit patterns.
ASan/UBSan exercise cache-enabled repeated calls, budget changes, valid versions
1/2 and 128 deterministic mutated/truncated files. Pinned official Engram and
proposed factory each pass nine full-output cases with a 4 KiB native cache;
random weights remain interface-only fixtures. No real tokenizer, trained Engram,
backbone logits or GPU was tested.

```bash
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
python tools/benchmark_native_cache.py \
  --training-dir artifacts/native-ngram-20261018/trained-reference \
  --corpus artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt \
  --native-library build/libmeft_rows.dylib \
  --output-dir artifacts/native-cache-reproduction --repeat 5 \
  --cache-bytes 262144 --larger-cache-bytes 2097152
python -m unittest discover -s tests
python tools/check_native_sanitizers.py
```

Linux uses `make native` and `build/libmeft_rows.so`. Exact commands, commits,
source/library/artifact/trace hashes are in the linked reports. Query code and
benchmark commit: `1e187dc1a621cb67deed75587c51c99146bd4d09`; training commit:
`ba564539489c44ff2582a7d83039a57c7dde7b9e`. Python 3.12.14, PyTorch 2.14.1,
NumPy 2.5.4, Apple M5, 10 logical CPUs, 32 GiB, macOS 26.6.2 arm64, Apple clang
21.0.0, one torch/interop thread, CPU only. Local compute without paid API or
rented compute; monetary/energy cost unmeasured. Original public corpus license
was not established, so corpus/weights are not redistributed. Raw artifacts
and queries are byte-exact; performance and fit depend on workload.
