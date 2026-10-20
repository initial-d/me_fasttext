# Native lossless XOR blocks and trained-row queries — 2026-10-20

The earlier offline XOR estimate is now an executable storage path: the C
reader reconstructs adaptive XOR blocks and returns each original row's exact
bytes. On the same nine trained SGNS tables, actual files save **6.9% at maximum
order 4, 8.9% at 8, and 9.0–9.1% at 12** when grouped by identical training
update signatures. These are small English character-ngram results, not trained
Engram compression or model-quality evidence. Grouping requires training metadata.

Raw evidence: [all trained file/query measurements](native_xor_evidence_20261020/trained-native.json),
[earlier byte-decoder pilot](native_xor_evidence_20261020/byte-decoder-pilot.json),
[official module](native_xor_evidence_20261020/official-forward.json),
[proposed factory](native_xor_evidence_20261020/factory-forward.json),
[tests](native_xor_evidence_20261020/tests.txt),
[sanitizers](native_xor_evidence_20261020/sanitizers.txt), and
[commands/environment/hashes](native_xor_evidence_20261020/provenance.json).

The [2026-10-21 export follow-up](streaming_export_20261021.md) reduces XOR
packing/staging costs while preserving these artifact bytes. Timings below
remain the original 2026-10-20 results.

## What changed

`pack_native(..., predictor="xor")` writes `MEFTNAT1` version 2, adding one
codec to the existing 32-byte block descriptor. Every block compares raw,
ordinary zlib and previous-row XOR followed by zlib, then stores the smallest
candidate. The first row is unchanged and prediction resets at every block.
Ties retain the simpler encoding. Existing calls still write version 1, and the
new reader accepts both versions. An old reader rejects version 2 explicitly.
The C ABI, per-original-row 8-byte map and descriptor widths stay unchanged.

XOR operates on bits rather than float subtraction. C decompresses one block,
reconstructs successive rows in its existing scratch allocation, then copies
requested rows in original query order. Word-sized operations use `memcpy` so
unaligned rows and widths with byte tails work without alignment/aliasing UB.
No persistent block cache, extra decoded table, GPU path or training support
was added. Each handle remains single-worker.

`row_order` optionally supplies a full permutation of original IDs for packing.
The builder writes forwarding entries at their original positions, preserving
all original IDs across reordering and exact deduplication. It validates full
coverage rather than dropping unqueried rows. Permutation validation uses O(n)
memory and construction reads become nonsequential; it is not a huge-table
streaming solution. Without a permutation/dedup dictionary, source rows still
stream in bounded blocks, with descriptors growing with block count.

## Actual file sizes

All three seeds (3/7/11), maxima 4/8/12, three layouts and two codec settings
were retained: **54 actual files**. Every file was read through C and every
original row was compared byte-for-byte. The denominator is the raw float32
ngram table; the numerator includes the native header, full forwarding map,
descriptors and payloads. JSON manifests and training/key metadata are separate.
No exact deduplication was used; the previously trained rows are byte-distinct.
Each block holds at most 64 rows of 128 float32 components (32 KiB).

Ranges below cover the three seeds; they are not confidence intervals.

| Max order | Layout | Ordinary file / raw | Adaptive XOR file / raw | XOR saving vs raw |
| --- | --- | ---: | ---: | ---: |
| 4 | Lexical | 0.94994–0.95009 | 0.94980–0.94993 | 5.01–5.02% |
| 4 | Shuffled | 0.94999–0.95023 | 0.94999–0.95023 | 4.98–5.00% |
| 4 | Identical-update group | 0.94613–0.94631 | 0.93064–0.93082 | 6.92–6.94% |
| 8 | Lexical | 0.94846–0.94878 | 0.93219–0.93258 | 6.74–6.78% |
| 8 | Shuffled | 0.94877–0.94910 | 0.94877–0.94910 | 5.09–5.12% |
| 8 | Identical-update group | 0.94530–0.94561 | 0.91081–0.91146 | 8.85–8.92% |
| 12 | Lexical | 0.94822–0.94841 | 0.93045–0.93067 | 6.93–6.96% |
| 12 | Shuffled | 0.94861–0.94881 | 0.94861–0.94881 | 5.12–5.14% |
| 12 | Identical-update group | 0.94513–0.94542 | 0.90948–0.90983 | 9.02–9.05% |

Shuffled layouts selected no XOR blocks. At order 12, all 221 lexical blocks
selected XOR, and 218–219 update-group blocks did. This supports a locality
effect in these learned rows, with stronger structural opportunities at higher
orders. It does not establish a universal order/compression law. The fixed
small word vocabulary largely saturates by order 12. Hashed Engram buckets do
not automatically supply lexical keys or identical-update signatures.

Adaptive selection ensures files do not exceed the ordinary codec **with the
same layout/map settings**. Layout changes can independently help or hurt.
Whole artifacts may still exceed raw tables on other inputs because of metadata.

## Query and construction cost

Queries use fixed original-ID batches with replacement; 32/256-row batches
force repeated IDs in one quarter of positions. All variants receive identical
IDs. `ctypes` calls and C gather are timed, with preallocated IDs and output;
there is no Python list creation in timed sections. Full row verification warms
the payload before timing. Each batch has three warmups and 40 samples; each
run records p50/p95 and block/decompression counts in JSON.

The table shows the median of per-run p50 across seeds for **update-group**
layouts, in microseconds. It does not pool the samples across seeds.

| Max order | Batch rows | Ordinary native p50 µs | Adaptive XOR p50 µs |
| --- | ---: | ---: | ---: |
| 4 | 1 | 53.62 | 54.75 |
| 4 | 32 | 1,223.02 | 1,257.10 |
| 4 | 256 | 3,971.31 | 4,089.94 |
| 8 | 1 | 53.62 | 54.44 |
| 8 | 32 | 1,276.67 | 1,312.00 |
| 8 | 256 | 6,405.77 | 6,639.79 |
| 12 | 1 | 53.54 | 54.44 |
| 12 | 32 | 1,224.02 | 1,259.25 |
| 12 | 256 | 6,852.12 | 7,098.33 |

The additional warm p50 cost is about **1.5–3.7%** here. There is no resident
tensor or direct raw-mmap latency baseline in this experiment; compressed
gather is not claimed faster than either. Larger random batches decode many
blocks, and a one-row query still decompresses its full block.

The earlier byte-at-a-time decoder pilot incurred about 15–17% extra batch
latency. Word-sized reconstruction reduced that cost while preserving identical
artifact sizes. The pilot is retained, not mixed into the final comparison.
There is no formal significance analysis; these are local warm measurements.

XOR export is slower: median update-group export times are ordinary/XOR
**0.050/0.173 s**, **0.124/0.446 s**, and **0.131/0.468 s** at orders 4/8/12.
Trying a second codec and Python bytewise prediction costs roughly 3.5× in
these small runs. Open timing follows export on a warm filesystem and excludes
the PyTorch full-file checksum; it is not cold startup. Peak builder/serving RSS,
disk-cache reset, concurrency and CPU-to-GPU transfer are unmeasured.

## Engram and dtype verification

The pinned official Engram demo and the proposed factory each run nine cases:
float32/float16/bfloat16 × Python/native/native-XOR. All source rows and full
official gates/ShortConv outputs remain byte-exact. Factory construction also
avoids allocating the original table. Tiny random weights verify compatibility,
not compression potential; these fixtures generally fall back to raw/zlib.
Separate tests force XOR blocks in all three dtypes, with NaN payloads, signed
zero and infinities, plus odd widths, reordered IDs, duplicate queries, block
tails, corrupt payloads and version rejection. **29 tests pass** with PyTorch;
the system-Python run skips 16 optional-dependency cases. ASan/UBSan check valid
versions 1/2 and 128 deterministic mutated/truncated files.

The upstream [draft PR #27](https://github.com/deepseek-ai/Engram/pull/27) is
still open/unmerged as checked on 2026-10-20. This codec is in `me_fasttext`;
it does not add a dependency or codec implementation to the upstream proposal.
No trained Engram checkpoint, tokenizer equivalence, backbone logits, GPU or
held-out model-quality result is established.

## Run it

```python
from tools.native_row_store import pack_native
from tools.engram_row_adapter import export_demo_embedding

# Preserve raw bytes and original global IDs, without a dedup dictionary.
pack_native("table.raw", "table.men", row_bytes=512, block_rows=64,
            deduplicate=False, predictor="xor")

# Keep Engram's local-head IDs and offsets; existing loader needs no change.
export_demo_embedding(original, head_sizes, "/tmp/layer-xor", block_rows=64,
                      backend="native", deduplicate=False, predictor="xor")
```

To reproduce the trained comparison after the
[original training run](trained_ngram_similarity_20261018.md#reproduce):

```bash
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
python tools/benchmark_native_residuals.py \
  --training-dir artifacts/native-ngram-20261018/trained-reference \
  --corpus artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt \
  --native-library build/libmeft_rows.dylib \
  --output-dir artifacts/native-xor-reproduction --block-rows 64 --repeat 40
python -m unittest discover -s tests
python tools/check_native_sanitizers.py
```

Linux uses `make native` and `build/libmeft_rows.so`. The source is pinned by
training-report/corpus/row hashes; the script downloads no external data and
retains every run. Code commit: `a69dbe3103d0fb53b9a434b016d9107fbedd8b29`.
Training commit: `ba564539489c44ff2582a7d83039a57c7dde7b9e`.
Environment: Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4, Apple M5, 10 logical
CPUs, 32 GiB, macOS 26.6.2 arm64, Apple clang 21.0.0; one PyTorch/interop thread,
CPU only. Local compute, no paid API/rented compute; monetary/energy cost was
not measured. The public-text source license was not established in the original
run, so corpus/weights are not redistributed. Exact commands and source/library/
artifact fingerprints are in the linked reports and provenance.
