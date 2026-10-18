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

## Native layout and construction

`MEFTNAT1` is a binary little-endian format: versioned fixed-size header,
8-byte original-row-ID forwarding map, 32-byte block descriptors and block
payloads. The schema lives beside the reader in `native/meft_rows.c`.
Rows are opaque source bytes; dtype/endian reconstruction belongs to the caller.

The exporter shares only byte-identical rows when `deduplicate=True`. With
`deduplicate=False`, it streams the source using a block buffer and temporary
map/payload files, avoiding a dictionary of every unique row. Descriptor memory
still grows with the number of blocks. It retains all rows for arbitrary future
queries; it does not prune according to a sample query trace.

Each block chooses zlib only when smaller than the raw block. Raw fallback
avoids codec expansion, but maps/descriptors/metadata can still make the whole
artifact larger. This is a distinct format, not a reinterpretation of legacy
`.z` or Python `MEFTROW1` files.

The reader bounds-checks file structure, descriptors and queried row IDs. It
maps bytes lazily and groups a batch's requests by block, decodes once per block
and restores original order. There is no persistent decoded-block cache in this
version; scratch retains up to one maximum decoded block plus per-batch request
locations. Direct C opening does not verify a full-file cryptographic checksum;
the PyTorch adapter does. Use trusted, immutable artifacts and external checksums
for archival provenance. Structural checks and sanitizers are not a production
artifact-security guarantee.

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
offline XOR/layout experiment estimates 6.9–9.1% savings; that predictor is not
part of the native reader. Similarity and exact-byte equality are measured
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
