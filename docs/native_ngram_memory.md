# Native n-gram memory: C ABI and model integration

The native backend is a CPU storage/gather component for immutable embedding
tables. It accepts original numeric row IDs and writes identical bytes into
caller-owned memory. It is a place to optimize storage and query behavior;
it does not replace a model's tokenizer, n-gram hasher, contextual gates,
attention, language-model head or training.

The older `src/ftindex.h` interface uses `std::string` and `std::vector`, so it
is a **C++ interface**, despite its procedural style. The new
[`native/meft_rows.h`](../native/meft_rows.h) is a true C ABI, usable from C,
C++ and language FFI. It depends on POSIX mmap and zlib; Linux and macOS are
targets. Windows, CUDA and distributed serving are not implemented.

## Build and call from C

```bash
# Linux: zlib development headers and a C compiler are required.
make native
# macOS:
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
```

This builds a standalone shared library and C example independently of the
bundled legacy FastText/trie binaries. From a raw fixed-width byte table:

```python
from tools.native_row_store import pack_native

# Example: 128 float32 values per row. No numeric/dtype conversion.
pack_native("table.raw", "table.men", row_bytes=512, block_rows=256,
            deduplicate=False)
```

```bash
build/native_gather table.men 0 42 0
```

The header exports opaque handles, row count/width, `meft_rows_gather`, close,
and last-query block/decompression counters. IDs, pointers and capacities use
fixed-width integers or `size_t`; output must have `count * row_bytes` capacity.
Return zero on success and a readable error on failure. Failed output must be
discarded. One handle is single-threaded because it owns a scratch buffer; use
one per serving worker and close it explicitly. The C example is an ABI check,
not an ANN/nearest-neighbor engine.

`meft_rows_gather_chunked` limits the number of query-location records sorted
at once and writes each chunk directly into the caller's complete output. The
native PyTorch adapters use `query_chunk_rows=65536` by default; set a smaller
positive value to lower this temporary allocation or `None` for the previous
single-query gather. It counts global row requests, including each head's ID,
not tokens. Ordinary embeddings, the Engram factory and unique embedding bags
accept this setting. Outputs retain all original IDs, order, dtype and row bits;
last-query block/decode/cache-hit counters aggregate the entire call. Small calls
still use the original ABI, while larger chunked calls require the rebuilt
library. Chunk boundaries can repeat block decoding, so throughput can decrease.
This bounds C query-location/sort scratch only: the full output, Python/Torch ID
and staged-row tensors, decoded-block scratch, cache and mmap remain separate.
The Python reference backend is unchanged; this is not a total-RSS bound.

The PyTorch mapped modules support `copy.deepcopy` and serialization into
`multiprocessing` spawn workers. Only configuration, metadata and small buffers
are copied; the mmap, C pointer and decoded cache are excluded. Each restored
module verifies the expected table checksum and opens its own handle with a cold
cache. Checksum verification scans the full artifact during restore, adding startup
I/O without materializing its rows. `MappedEmbedding` and `MappedEmbeddingBag` inherit this through their
gather module. Close every copy explicitly. The artifact directory and native
library must remain available at the captured absolute paths; this is a local
worker handoff, not a portable self-contained checkpoint. A replaced/changed
table fails checksum verification rather than silently changing the snapshot.
Closed modules cannot be copied/serialized. Constructor cache settings are
restored; later direct cache changes through the low-level handle are not copied.
Use spawn or open modules independently inside fork workers; this does not make
one live handle safe to share across threads or coordinate distributed serving.

## Sum/mean bags without a full gathered tensor

For immutable CPU `nn.EmbeddingBag` inference, export the learned rows and
reduction configuration, then reopen them:

```python
from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag

export_embedding_bag(model.memory, "memory-bag", predictor="xor")
with open_embedding_bag("memory-bag") as memory:
    output = memory(ids)  # 2-D fixed bags, or memory(flat_ids, offsets)
```

This requires an actual sum/mean `nn.EmbeddingBag`, not an arbitrary model memory
module. To use an existing single-head native embedding artifact, construct
`MappedEmbeddingBag(path, mode="mean", padding_idx=0)` explicitly. A model must
expect the reduced `[bags, dim]` output; the adapter does not preserve an ordinary
embedding's unreduced shape. The included character-GRU supports both paths.

The default `strategy="unique"` gathers each distinct original ID once and uses
PyTorch's bag reduction; staging grows with query diversity, with at most one
unused row to retain the configured padding-aware kernel. `strategy="chunked"`
calls the C reducer with `chunk_rows=1024` (configurable), capping staged rows but
potentially decoding blocks repeatedly. C accepts complete nondecreasing offsets
of length `bags+1`, LE FP32/FP16/BF16 rows, sum/mean mode, padding ID or UINT64_MAX,
and caller-owned FP32 output. C accumulates in FP64; Python casts to source dtype.
Stored row bytes stay exact, while reduction precision can change model logits.

Both adapters preserve original IDs, exclude padding from sums/counts and return
zeros for empty/all-padding bags. Ragged offsets must start at zero and cover
all IDs; `include_last_offset` selects whether the caller supplies the terminal
count. No training, max mode, `.weight`/tied heads or CUDA.
Build the current library for the new C ABI. See the
[trained-query timings, RSS, exact-byte checks and model deltas](embedding_bag_evidence_20261025/README.md)
before choosing a strategy; dense resident lookup is faster in these tests.

Sum mode also accepts `memory(ids, offsets, per_sample_weights=weights)` for
inference-time weighted memory aggregation. Omit `offsets` for fixed 2-D bags.
Weights must be CPU tensors with the same shape as IDs and the stored row dtype;
detach learned gates before using this inference-only adapter. Repeated IDs keep
distinct weights for each occurrence; padding ignores both row and weight.
Negative, zero and nonfinite weights follow arithmetic semantics. There is no
implicit normalization or weighted-mean mode. The default unique path delegates
weighted reduction to PyTorch without materializing all gathered query rows.
The optional chunked path uses the additive C ABI
`meft_rows_embedding_bag_weighted`, with a byte capacity for scalar weights and
the same row staging bound. It multiplies and accumulates in FP64 before FP32
output/source-dtype casting, so reduction rounding can differ from PyTorch.
Rebuild the library for weighted C calls; the unweighted ABI stays unchanged.
This adds an aggregation primitive; it does not replace a model's gating network
or establish trained Engram quality/performance evidence.

Large repeated batches now use `unique_method="auto"` to compact IDs with a
histogram and lookup table when the addressable table has at most four rows per
queried ID and those two index buffers fit `unique_index_bytes` (default 8 MiB).
Otherwise the adapter falls back to sorted unique IDs. Set `unique_method="sort"`
to retain the previous method, or `unique_index_bytes=0` to disable dense
compaction. Both paths preserve sorted original IDs, inverse bag membership and
the same reduction kernel; this index budget does not cap staged embedding rows
or total RSS. The retained table still supports arbitrary future queries.

## Native layout and construction

`MEFTNAT1` is a binary little-endian format: versioned fixed-size header,
8-byte original-row-ID forwarding map, 32-byte block descriptors and block
payloads. The schema lives beside the reader in `native/meft_rows.c`.
Rows are opaque source bytes; dtype/endian reconstruction belongs to the caller.
Version 1 uses raw/zlib blocks. Optional version 2 adds previous-row bitwise XOR
followed by zlib. The new reader accepts both; old readers reject version 2.
The C ABI and map/descriptor widths are unchanged.

The exporter shares only byte-identical rows when `deduplicate=True`. With
`deduplicate=False`, it streams the source using a block buffer and temporary
map/payload files, avoiding a dictionary of every unique row. Descriptor memory
still grows with the number of blocks. It retains all rows for arbitrary future
queries; it does not prune according to a sample query trace.

`pack_native` writes and flushes/fsyncs the complete table into a sibling
temporary file, closes it, then atomically replaces the destination path. A
failed write, sync or replacement preserves the previous destination and cleans
up the temporary file. Existing C mmap handles retain the old inode and its
bytes; new handles open the published table. Replacing a regular file preserves
its access permissions; new tables use owner-only tempfile permissions. This
publishes one table file: adapter metadata/checksums require a consistent bundle,
and concurrent writers are not coordinated. The binary format/codec is unchanged.

Each block chooses zlib only when smaller than the raw block. With
`predictor="xor"`, the exporter also tries zlib-compressed XOR residuals and
keeps the smallest of all three candidates. The first row resets prediction
inside each block; C reconstructs the full block before copying queried rows.
This preserves all bits and adds decoding work. Raw fallback
avoids codec expansion, but maps/descriptors/metadata can still make the whole
artifact larger. This is a distinct format, not a reinterpretation of legacy
`.z` or Python `MEFTROW1` files.

The reader bounds-checks file structure, descriptors and queried row IDs. It
maps bytes lazily and groups a batch's requests by block, decodes once per block
and restores original order. An optional [byte-budgeted decoded-block cache](native_cache_20261022.md)
reuses compressed blocks between calls and is disabled by default. Cached payload
bytes are bounded by the chosen budget, with a separate cap of 64 entries.
Scratch retains up to one maximum decoded block plus per-batch request
locations. Direct C opening does not verify a full-file cryptographic checksum;
the PyTorch adapter does. Use trusted, immutable artifacts and external checksums
for archival provenance. Structural checks and sanitizers are not a production
artifact-security guarantee.

To enable the optional codec, pass `predictor="xor"` to `pack_native`,
`export_demo_embedding`, `export_multihead` or `export_embedding` with the native
backend. Existing calls still write version 1. `pack_native(..., row_order=...)`
accepts a full permutation of original IDs for locality experiments; the stored
forwarding map restores original lookup IDs even across reordered/deduplicated
rows. Permutation validation uses O(n) memory and export reads become
nonsequential. Default sequential no-dedup export remains block-streamed.
See [actual files and query measurements](native_xor_compression_20261020.md).
The [streaming export follow-up](streaming_export_20261021.md) accelerates XOR
packing and stages embedding bytes in bounded row slices. The caller must freeze
source weights throughout export; deduplication and Python-builder memory limits
still apply. Artifact formats and native queries are unchanged.

## PyTorch: two integration boundaries

```python
from tools.engram_row_adapter import (
    export_demo_embedding, MappedMultiHeadEmbedding,
    export_embedding, MappedEmbedding,
)

# Engram-like multi-head memory: retain existing hash IDs and head offsets.
export_demo_embedding(original, head_sizes, "/tmp/memory-native",
                      backend="native", deduplicate=False)
memory = MappedMultiHeadEmbedding("/tmp/memory-native",
                                 native_library="build/libmeft_rows.so")
vectors = memory(head_local_ids_cpu)

# Ordinary nn.Embedding: retain global vocabulary IDs.
export_embedding(original_embedding, "/tmp/token-native", backend="native")
embedding = MappedEmbedding("/tmp/token-native",
                            native_library="build/libmeft_rows.so")
vectors = embedding(token_ids_cpu)
# Close memory/embedding when no longer used.
```

The native adapter passes contiguous ID tensor addresses and the destination
tensor's storage directly to C. It avoids `.tolist()`, Python per-row decoding
and intermediate Python byte-output reconstruction. Returned float32/float16/
bfloat16 tensors preserve shape and source bit patterns. This removes one
FFI/copying bottleneck; CPU decompression, sorting and I/O still have costs.

IDs and output are CPU tensors. A GPU caller would need explicit host ID
transfer, pinned output buffers and asynchronous transfer/scheduling work;
this backend has not implemented or measured that path. No `.weight` parameter,
embedding gradient, tied output projection or full-model serialization is
provided. `nn.Embedding(max_norm=...)` mutates weights during forward and is
rejected. Full artifact checksums scan the file during PyTorch opening, so that
path does not have lazy startup even though the C reader does.

## Where it fits across model families

| Model or runtime | Candidate integration point | Current evidence | Additional work |
| --- | --- | --- | --- |
| Engram-style conditional memory | Multi-head embedding gather after the existing token/ngram hash | Official full CPU module output matches in 6 random-weight cases, including gates/ShortConv | Trained checkpoint, backbone/LM logits, model quality, GPU transfer |
| PyTorch decoder with ordinary embeddings | Global-ID input embedding lookup | Byte-exact gather; small synthetic untied causal Transformer logits match | Model-specific wrappers, tied heads, device/distributed placement |
| Llama/Qwen/Mistral/Gemma-style HF models | A compatible `nn.Embedding` call or a separately trained n-gram branch | No downloaded model validated here | Inspect `.weight` use, tying, `set_input_embeddings`, device map, export and generation |
| C++ inference / llama.cpp-style runtime | CPU row gather behind a runtime-specific tensor operator | C ABI and example compile/test | Loader, tensor layout and runtime scheduling integration; no plugin yet |
| vLLM / TensorRT-LLM / GPU serving | Host-memory gather and prefetch/transfer operator | No integration or GPU result | Custom operators, pinned memory, async queues, batching and quality checks |
| Plain Transformer without n-gram memory | Add a trained n-gram memory branch | No architectural/training result | Learn/gate the branch and evaluate against strong baselines |

A storage adapter can preserve an existing table. Adding high-order n-gram
memory to a model is a separate architecture/training change. Models that do
not query such a table cannot gain Engram capability from linking a C library.

## N-gram order: the testable claim

More n-gram orders can increase key length, vocabulary diversity, table size,
lookup count and the opportunity for shared-prefix indexing. They do not imply
that trained vectors have identical bytes or directions. In fixed-size hashed
tables, order/key count can grow while each table's capacity stays capped;
collisions and lookup fan-out must be measured separately.

The [order sweep](ngram_order_native_20261018.md) tests maxima 2/3/4/6/8/12,
two controlled key workloads and unchanged official hash/gather code. It
separates key prefix encoding, weight compression and native query latency.
Its narrow result: shared-prefix key encoding benefits at higher orders;
random unique weight tables still grow after packing and gather work increases.
It is not evidence that higher order universally improves compression or model
quality. Front-coded key sizes are not an implemented trie/query result.

Random rows are only interface/low-redundancy controls. The primary
[learned-structure experiment](trained_ngram_similarity_20261018.md) trains
real-text SGNS across maxima 4/8/12 and three seeds. Identical-update structural
pairs have median cosine about 0.996; their share increases at higher orders.
Native ordinary block storage preserves all bytes and saves about 5%. A separate
offline XOR/layout experiment estimated 6.9–9.1% savings. The
[native follow-up](native_xor_compression_20261020.md) now writes and reads actual
adaptive XOR files with comparable savings and disclosed query costs. Similarity and exact-byte equality are measured
separately, with shuffled controls and all runs disclosed.

## Verification and next milestones

The [full official Engram module verification](engram_full_adapter_verification_20261018.md)
runs unchanged hash initialization, addressing, gates/projections and ShortConv
with a local normalized-token fixture. Python/native × three dtypes preserve
every row and full module output. It does not validate real tokenizer/weights,
backbone logits or GPU use.

Native tests compile pure C, check byte equality and repeated batch order, raw
fallback/compressed blocks, output capacity, corrupt forwarding/descriptors,
invalid IDs and malformed/truncated files. ASan/UBSan exercise valid and 64
deterministically mutated files. PyTorch tests cover all supported dtypes and
a small causal decoder with an untied head. See the linked raw report for exact
commands, compiler, environment and negative results.

The practical sequence is: trained immutable table and row/gather equality;
larger footprint/cold/warm/RSS/batch tests; real model gates/logits and generation;
then native streaming/async/GPU integration. Claims should name the tested
boundary and distinguish storage savings, latency, indexed key compression,
and task quality.
