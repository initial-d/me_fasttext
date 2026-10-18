# Official-demo gather interface verification — 2026-10-18

**The CPU adapter preserves original head IDs and embedding bytes on the
verified random-weight inputs. It does not save storage or beat resident
PyTorch gather on this small experiment.** This is interface/engineering
evidence; no trained Engram checkpoint is available or evaluated.

## Provenance and exact scope

- Implementation and harness commit: `701dcb813909bc5746121276e9ba34e8c60ff56b`.
- Official demo commit: `fb7f84a21f91223715394a33a1dc24bbfb7f788e`;
  [source](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py).
- Official source file SHA-256:
  `9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044`.
  The harness verifies this fingerprint before executing the unchanged
  `NgramHashMapping._get_ngram_hashes` and `MultiHeadEmbedding` definitions.
  It does not run the tokenizer/downloads, initialization of the full hash
  mapper, gates, convolution, backbone or training.
- Fixed synthetic normalized IDs: seed 7, 2 batches x 32 positions,
  n-gram orders 2/3, 2 heads per order, head sizes `[257, 263, 269, 271]`,
  original offset convention, multipliers `[17, 31, 47]`, pad ID 0.
- Random initialized `nn.Embedding` tables: 1,060 rows x 16 components,
  float32/float16/bfloat16. These are tiny synthetic sizes and random weights,
  not the official architecture's trained weights. All rows are unique in
  each case; no duplicate rows were deliberately introduced.
- CPU, torch/interop threads 1; zlib, 32-row blocks, 4 cached blocks.

The [raw result](engram_adapter_verification_20261018.json) includes input
hashes, unchanged weight/table hashes, exported artifact hashes, query output
hashes, shapes, commands, hardware, software, tests and timings.

## Acceptance checks

All source rows recover byte-for-byte in all three dtypes. Original-hash
queries, repeated batches, noncontiguous input, head-boundary IDs and empty
outputs all match original gather shape, dtype and bytes. The original
weight and hash outputs remain unchanged after export and queries.

Unit tests additionally cover repeated rows across head boundaries, signed
zero and NaN payloads, int32/int64 IDs, output ownership, incorrect head IDs,
metadata/offset mismatch, modified artifact checksums and inference-only mode.
With a disabled block cache, grouped batch reads decode each requested block
once and restore query order exactly.

## Footprint and gather latency

All reported sizes include the row-store file and JSON adapter metadata.
Raw size is just the original dense weight bytes, without an application
container. Every artifact includes an 8,480-byte original-row-ID map.

| Random table dtype | Raw bytes | Total artifact / raw | Official gather p50 / p95, µs | Adapter gather p50 / p95, µs |
| --- | ---: | ---: | ---: | ---: |
| float32 | 67,840 | 1.1112 | 3.083 / 3.292 | 243.375 / 245.833 |
| float16 | 33,920 | 1.3121 | 3.250 / 3.458 | 186.583 / 192.625 |
| bfloat16 | 33,920 | 1.1757 | 3.250 / 3.541 | 188.625 / 192.208 |

**The artifact grows for each random table, and the adapter is slower than
ordinary resident gather.** Full-file checksum verification is included in
adapter opening (about 2.6–3.5 ms here); it is not lazy startup. Exporter memory
and runtime RSS are not measured. There is no real-model memory saving claim.

## Block coalescing comparison

For the same 256 requested rows with cache disabled, grouping uses 33 block
decodes versus 256 for per-row reads. This is an internal row-store comparison,
not a speedup over PyTorch or an end-to-end Engram result.

| Dtype | Coalesced byte reads p50 / p95, µs | Per-row byte reads p50 / p95, µs |
| --- | ---: | ---: |
| float32 | 232.083 / 238.333 | 1,338.021 / 1,403.333 |
| float16 | 175.250 / 177.666 | 904.042 / 919.917 |
| bfloat16 | 177.792 / 186.958 | 914.230 / 939.584 |

These measurements exclude tensor reconstruction in the byte-read paths.
All timings use 5 warm-up calls and 30 measured calls, one CPU process and
warm page cache, with Python dispatch/copies included. No cold-cache, GPU
transfer, concurrent workload, batch-size sweep or model quality is measured.

## Reproduce

Environment: Apple M5, 10 logical CPUs, 32 GiB RAM, macOS 26.6.2 arm64,
Python 3.12.14, PyTorch 2.14.1, NumPy recorded in the JSON. CPU only;
CUDA unavailable and MPS unused. No monetary/trading cost assumption applies.

With PyTorch/NumPy installed, save the pinned source and run from repository root:

```bash
curl -fL \
  https://raw.githubusercontent.com/deepseek-ai/Engram/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py \
  -o /tmp/engram_demo_v1.py
python3 tools/demo_engram_adapter.py \
  --official-source /tmp/engram_demo_v1.py \
  --output-dir /tmp/meft-engram-gather
python3 -m unittest discover -s tests
```

The recorded run used the exact interpreter/path in the raw JSON command;
the command above is a portable equivalent. Use a fresh output directory.
Tests: **12 passed** with PyTorch. Standard-library-only tests discover 12,
pass 8 and skip 4 optional adapter tests. A separate CI job installs CPU
PyTorch so those tests cannot be silently skipped in adapter validation.
Ruff (`--select E9,F,B`), Python syntax and `git diff --check` pass.

No model files are committed. The harness regenerates random inputs; its
timings and random initialized weights can vary with environment/version.
The [adapter guide](engram_gather_adapter.md) documents its immutable,
CPU-only interface and scaling limits. A trained, licensed memory table is
still needed for meaningful compression, RSS and end-to-end model evaluation.
