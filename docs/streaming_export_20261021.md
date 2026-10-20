# Faster XOR packing and bounded embedding staging — 2026-10-21

The native XOR exporter is now about **1.9× faster** on the nine existing
trained SGNS tables, with baseline and optimized files identical. Embedding
export also stages CPU bytes one block at a time instead of copying the entire
table into a Python bytes object. The file format, row IDs, source dtype, hash,
compression choices and C query implementation are unchanged.

Evidence: [export timings and process RSS](export_evidence_20261021/export.json),
[official CPU module](export_evidence_20261021/official-forward.json),
[factory integration](export_evidence_20261021/factory-forward.json),
[31 tests](export_evidence_20261021/tests.txt),
[standard-library run](export_evidence_20261021/stdlib-tests.txt), and
[provenance](export_evidence_20261021/provenance.json).

## Two changes

Previous-row XOR is a bit operation. Converting one block to a little-endian
Python integer, shifting the preceding rows by one row width, and XORing the
two integers computes exactly the same bytes in C-backed integer operations.
Explicit byte length preserves leading/trailing zeros and odd widths. This
removes the Python per-byte loop, adds no dependency, and keeps work buffers
bounded by the block size. Ordinary zlib/XOR/raw candidate selection is unchanged.

`export_multihead` now iterates through source row slices of `block_rows`, copies
each slice to contiguous CPU storage, writes its bytes to the temporary raw file,
and updates SHA-256 incrementally. It avoids a whole-table CPU contiguous copy
and whole-table Python bytes staging. `export_demo_embedding` and
`export_embedding` inherit this change with no new arguments.

This remains an **immutable checkpoint export**. The caller must prevent source
weight updates for the entire export; per-block `detach()` does not snapshot
concurrent training. CPU, noncontiguous CPU and three supported dtypes were
tested. GPU slices use the same bounded transfer code, but GPU export and transfer
timings were not tested. The raw temporary file, output file and spools still
need disk space. Native descriptors grow with block count; exact deduplication
still retains an unbounded unique-row dictionary. The Python row-store builder
still retains unique rows/payloads. Use native `deduplicate=False` to avoid that
dictionary; this change alone does not make every builder scale to huge tables.

## Original trained tables: timing and equality

Baseline: `2fd270fd2a6ef4e69fed38c1c27a1ce90917008e` (merged native XOR work).
Optimized code: `70a2f360777e37d2ae421ebd34dbd28c0958e0d4`.
Source hashes are in the report. The baseline exporter is loaded from the exact
Git commit rather than copied into the repository.

All nine existing order/seed tables are unchanged public-text-trained SGNS
float32 rows, 128 components each, from the
[original study](trained_ngram_similarity_20261018.md). No new data download or
training was performed. Each original table is packed three times per exporter
with 64-row blocks, no deduplication, original lexical row order and XOR enabled.
Run order alternates. Source and artifact SHA-256 are checked, and C reconstructs
every original row after each run. **All 27 baseline/optimized file pairs are
byte-identical**, not just similar vectors.

The table reports the median of nine samples per order (three seeds × three
repetitions). These are warm local export times, not cold-disk or query timings.

| Max order | Original rows | Baseline XOR export s | Optimized XOR export s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 4 | 5,333 | 0.1573 | 0.0820 | 1.92× |
| 8 | 13,424 | 0.4122 | 0.2169 | 1.90× |
| 12 | 14,132 | 0.4486 | 0.2394 | 1.87× |

All adverse results and samples remain in JSON. No confidence intervals or
formal significance analysis are claimed. Trying two compressors still costs
more than ordinary zlib; the optimization reduces the previous XOR overhead.
No additional storage saving or query-speed improvement is claimed because
the artifacts and C reader are unchanged.

## Separate memory-scaling fixture

To expose table-sized staging overhead, a second test repeats the trained
order-12/seed-7 rows to 131,072 rows: **64 MiB of float32 source weights**.
This is deliberately duplicated trained data for an export-memory test, not a
representative large checkpoint or new compression result. No independent
random vectors are used as compression evidence. Both exporters use native XOR
with `deduplicate=False` and reconstruct every original row.

Each exporter runs in a fresh subprocess, three times, alternating order.
`ru_maxrss` is captured after source-weight construction and immediately after
export, before verification. Reported additional peak is the difference between
those process high-water marks; it is not an exact sum of live allocations.
The median across three processes is:

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Full embedding export time | 4.121 s | 2.207 s |
| Process peak RSS after export | 336.89 MiB | 272.47 MiB |
| Additional peak above pre-export high-water mark | 67.19 MiB | 2.70 MiB |

Baseline/optimized artifact and source fingerprints match in every pair.
This isolates staging improvement in a local CPU process. The source tensor,
Python/PyTorch runtime and native export buffers still use memory. Disk/cache
space is outside RSS; cache is warm. Larger tables, noncontiguous large tensors,
GPU transfer, concurrent serving and end-to-end model memory remain unmeasured.

## Integration validation

31 tests pass with PyTorch. Tests bound the bytes passed to NumPy to one block
per call, verify noncontiguous float32/float16/bfloat16 weights, streamed source
hashes, original source preservation and packed-file equality for both backends.
Codec tests include leading/trailing zeros, odd row widths and the existing
NaN payload/signed-zero/infinity cases. System Python skips 17 optional tests.

The pinned official Engram demo and proposed factory each pass all nine cases
across Python/native/native-XOR and three dtypes. Every source row and full
gated/ShortConv output is byte-exact. Random weights in these harnesses are
interface fixtures only. No trained Engram checkpoint, tokenizer equivalence,
GPU, backbone logits or held-out task quality was tested.

## Reproduce

```bash
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
python tools/benchmark_export.py \
  --training-dir artifacts/native-ngram-20261018/trained-reference \
  --native-library build/libmeft_rows.dylib \
  --output-dir artifacts/export-reproduction --repeat 3 --memory-rows 131072
python -m unittest discover -s tests
```

Linux uses `make native` and `build/libmeft_rows.so`. The benchmark requires the
original hashed trained files, the baseline Git commit and a fresh output
directory. Exact executed commands and source/library/artifact hashes are in
the reports. Environment: Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4, Apple M5,
10 logical CPUs, 32 GiB, macOS 26.6.2 arm64, Apple clang 21.0.0. CPU only, one
torch/interop thread in memory workers. The raw-table timing path itself is
single-threaded. Local compute, no paid API/rented compute; monetary/energy cost
was not measured. Original public-text source license was not established;
corpus/weights are not redistributed. These are SGNS storage/export measurements,
not the original paper's Chinese corpus results or Engram model benchmarks.
