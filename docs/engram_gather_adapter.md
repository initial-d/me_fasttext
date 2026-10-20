# Byte-exact PyTorch gather adapter for the Engram demo interface

`tools/engram_row_adapter.py` implements an immutable CPU gather at the official
demo's `MultiHeadEmbedding` boundary. Input `[..., heads]` contains the same
head-local IDs as before. The adapter adds the original per-head offsets and
returns `[..., heads, embedding_dim]` with identical row bytes and dtype.
It changes storage, leaving tokenization, hash addressing and downstream gates
to the caller.

## Export and replace an immutable embedding

PyTorch and NumPy are optional dependencies for this adapter. The basic row
store still needs only Python's standard library.

```bash
python3 -m pip install torch numpy
python3 -m unittest discover -s tests
```

Given an existing official-demo-style module named `original` and its actual
per-head vocabulary sizes:

```python
from tools.engram_row_adapter import export_demo_embedding, MappedMultiHeadEmbedding

# Export every trained row without quantization or approximate sharing.
export_demo_embedding(original, head_sizes, "/tmp/frozen-engram-table", block_rows=256)

with MappedMultiHeadEmbedding("/tmp/frozen-engram-table", cache_blocks=4) as mapped:
    # CPU integer IDs, shape [..., heads], unchanged from the original hasher.
    embeddings = mapped(head_local_ids)
```

For a CPU demo, replace `engram.multi_head_embedding` with this adapter after
exporting that layer's table. Preserve each layer's own head sizes/offsets and
use a separate output directory. This is the gather interface only; a full
Engram model/checkpoint loader is not implemented. Always close the adapter
when serving ends. `.train()` is rejected because no optimizer updates or
embedding gradients are supported.

`export_multihead(weight, head_sizes, output_dir)` also accepts a dense table
directly. Export preserves float32, float16 and bfloat16 bytes, detaches a
CPU row slices and writes `table.mer` plus `embedding.json`. Slices contain at
most `block_rows` rows, and source SHA-256 is updated incrementally. Freeze the
source table for the entire export; concurrent updates are unsupported. It records
shape, dtype, byte order, head offsets and raw/packed SHA-256 fingerprints.
It refuses to overwrite an existing adapter. Reader startup validates the full
artifact checksum and shape, so startup is O(file size), not lazy mmap loading.
The metadata/file format is for locally produced trusted artifacts.

## Batch query optimization

The row store groups requested row IDs by their compressed block. Each block is
decoded once per batch, even with cache disabled, then rows are copied back to
their original positions. Repeated IDs, arbitrary input order and tensor shape
remain intact. Persistent cache size is bounded; working memory additionally
includes the batch's locations/output and one decoded block.

Tests verify per-head boundaries, repeated and noncontiguous ID batches,
int32/int64 inputs, empty batches, all source rows and output bytes for all
three dtypes (including signed zero and NaN bits). They reject wrong offsets,
out-of-head IDs, invalid shapes/dtypes, modified artifacts and training mode.
A disabled-cache test confirms one decode per requested block.

The copied output tensor owns its data; modifying it does not change stored
weights. There are no embedding parameters or whole-table tensor in the adapter.
This alone does not establish an RSS saving: builder memory, forwarding IDs,
decompressed cache, batch copies, Python objects and filesystem page cache must
all be measured for a real workload.

## Offline official-interface verification

Save the pinned official demo locally (read-only download):

```bash
curl -fL \
  https://raw.githubusercontent.com/deepseek-ai/Engram/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py \
  -o /tmp/engram_demo_v1.py
python3 tools/demo_engram_adapter.py \
  --official-source /tmp/engram_demo_v1.py \
  --output-dir /tmp/meft-engram-gather
```

The harness checks the source SHA-256 before executing only the original
`NgramHashMapping._get_ngram_hashes` and `MultiHeadEmbedding` definitions.
It bypasses tokenizer construction/downloads and supplies tiny synthetic
normalized IDs, hash multipliers/head sizes and random initialized weights.
It checks hash batches, head boundaries, repeated/noncontiguous queries, empty
outputs and all stored source rows, with each supported dtype. The original
hash and embedding implementation remain unchanged.

The report also compares warm official tensor gather, adapter gather, uncached
coalesced block reads and uncached per-row reads. Timing includes CPU Python
dispatch and copying. Opening includes the full-file checksum. Use a fresh
output directory for each run. This verifies an interface, not training,
Engram logits, model quality, or production offload performance.

## Current limits

The [native C backend](native_ngram_memory.md) adds `backend="native"` export
and `native_library=...` loading, writes directly into the output tensor, and
supports ordinary immutable `nn.Embedding` calls through `MappedEmbedding`.
Its no-dedup construction streams rows instead of retaining a unique-row
dictionary; descriptors still grow with block count. The limits below describe
the original Python backend where applicable.

- CPU inference only; no GPU transfer, asynchronous prefetch, distributed
  sharding, concurrent-thread contract or embedding gradients.
- Offline builder still keeps all unique rows and compressed payloads in RAM.
  It is unsuitable for exporting huge tables without a streaming builder.
- Random weights may have little byte redundancy; file size can exceed raw
  weights, and compressed gathers can be much slower than resident tensor
  gathers. Measure storage and latency separately.
- The legacy C++ exporter remains approximate with the
  [documented source mismatch](engram_storage.md#legacy-export-audit).

See [the recorded interface result](engram_adapter_verification_20261018.md)
and [launch copy](engram_promotion.md). The new
[trained SGNS reference](trained_ngram_similarity_20261018.md) measures shared
training and byte-exact storage on real text. A trained Engram checkpoint and
serving workload are still needed for Engram footprint/model-quality claims.
