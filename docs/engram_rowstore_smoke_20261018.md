# Byte-exact row-store smoke check — 2026-10-18

This is **synthetic engineering evidence**, not a trained embedding/Engram
benchmark. Each table contains 2,048 opaque 256-byte rows (524,288 input bytes),
seed 7. The repeated-row case cycles through 128 unique rows; the second case
has 2,048 independent random-byte rows. No market/text/model data is involved.

Code commit: `fc137a49f022f8e113180cb36f3165fa355ddd83`.
Both cases verify all original row IDs byte-for-byte and a 4,096-ID batch with
repeated IDs in its original order. Neither approximates or quantizes rows.

| Synthetic case | Artifact bytes | Artifact / raw | Compressed p50 / p95, µs | Raw mmap p50 / p95, µs |
| --- | ---: | ---: | ---: | ---: |
| 128 exact rows repeated | 49,566 | 0.0945 | 0.500 / 0.583 | 0.084 / 0.250 |
| 2,048 unique random rows | 544,620 | 1.0388 | 1.542 / 1.667 | 0.083 / 0.208 |

**The unique random table grows, and compressed lookup is slower in both
cases.** The first case's reduction comes from manufactured exact duplicates;
it cannot predict redundancy in trained weights. The 16,384-byte forwarding
map, block metadata and compression framing all count in artifact size.

Environment: Apple M5, 10 logical CPUs, 32 GiB RAM, macOS 26.6.2 arm64,
Python 3.9.6, standard-library zlib. Single Python process on CPU; PyTorch and
GPU are not used. Rows are compressed in 32-row blocks with a 4-block cache.
Timing is one warm Python microbenchmark and includes Python dispatch/copies;
it excludes tensor reconstruction, CPU-to-GPU transfer, real concurrent load,
cold page-cache control and RSS. It is not a production latency comparison.
There are no monetary/trading cost assumptions.

Exact commands, from the repository root:

```bash
make test
python3 -m py_compile tools/lossless_row_store.py tools/demo_lossless_rows.py
python3 tools/demo_lossless_rows.py --output-dir artifacts/engram-20261018/rowstore
git diff --check
```

Unit tests: **7 passed** across the row store and existing public-benchmark
tools. Python syntax and diff checks passed. Raw result, input/artifact hashes,
hardware and test log: [engram_rowstore_smoke_20261018.json](engram_rowstore_smoke_20261018.json).
Re-run [demo_lossless_rows.py](../tools/demo_lossless_rows.py) to regenerate
the synthetic input and artifacts; timings will vary with machine/load.

The [storage guide](engram_storage.md) defines the row-level contract, source
audit and a future immutable-checkpoint adapter experiment. No trained Engram
table, logits, downstream task or actual memory-offload path has been tested.
