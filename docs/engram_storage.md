# Compact storage and lookup for Engram-style n-gram memory

Engram puts a systems problem back in focus: a model can retrieve learned
n-gram memory from a large table instead of reconstructing every static pattern
through dense computation. `me_fasttext` is useful to this community as an
inspectable storage/indexing project and a place to test row-level compression.

## What connects the projects

DeepSeek's [official README](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/README.md)
describes deterministic addressing and host-memory offload. Its
[demo](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py)
maps normalized token IDs to multi-head n-gram hash IDs, gathers an embedding
table, then applies hidden-state-dependent gates and a short convolution.
The demo mocks backbone components; it is not a full training/serving stack.
Sources checked on 2026-10-18 at the pinned commit above.

| Layer | Engram official demo | `me_fasttext` and the new reference |
| --- | --- | --- |
| Input keys | Normalized token n-grams, multiple hash heads | C++: UTF-8 character n-grams with trie IDs |
| Addressing | Hash IDs plus per-head offsets | C++: exact trie IDs; row-store reference: original numeric row IDs |
| Storage | Learned multi-head embedding matrix | Legacy compact `.z`; new opaque-byte compressed row store |
| Access | Embedding gather, then gates/projections | C++ lexical vectors; Python byte rows via mmap and bounded block cache |
| Context fusion | Hidden-state gates and short convolution | No transformer/gating implementation |

Character n-grams and token/hash IDs are different interfaces. Replacing Engram's
hash function with a trie changes model addressing and is an architecture
experiment. A storage adapter can instead keep Engram's existing row IDs and
recover the same bytes. That is the narrower experiment supported by the new
reference.

## Byte-preservation contract

`tools/lossless_row_store.py` accepts a raw, fixed-width byte table. Its steps are:

1. Share storage only for rows with identical bytes, including dtype bit
   patterns. Similar vectors are never merged.
2. Keep an 8-byte forwarding ID for every original row, preserving original
   ID order, repetitions and all rows; no observed-query pruning.
3. Compress consecutive unique rows in independent zlib blocks.
4. mmap the file, retrieve the original ID's block and row, and use a bounded
   cache of decompressed blocks.

`row(original_id)` must equal the original bytes. `rows(ids)` must preserve
batch ordering and duplicates. Tests include signed zero, NaN payloads,
opposite vectors of equal norm, block boundaries, cache eviction and malformed
inputs. This proves the row-level contract on test inputs, not end-to-end
Engram logits or task quality.

The format is experimental and intended for locally produced trusted artifacts.
Its Python builder holds unique rows, the ID map and compressed payloads in RAM;
it is not ready to build multi-hundred-GB tables. Compression can increase
storage when rows are mostly unique or incompressible. The reference omits
GPU transfer, distributed sharding, asynchronous prefetch, concurrent serving,
online training/updates and production artifact validation. It does not consume
FastText `.bin`/`.z` or full Engram checkpoints directly. The new
[PyTorch gather adapter](engram_gather_adapter.md) can export the official
demo's `MultiHeadEmbedding` interface and serve identical rows on CPU.

## Run the byte-exact demo

Only Python's standard library is needed:

```bash
python3 tools/demo_lossless_rows.py --output-dir /tmp/meft-lossless-demo
make test
```

The demo compares raw mmap slicing with the compressed store on two fixed
synthetic byte tables: repeated identical rows and unique random rows. It
verifies every row and a repeated-ID query batch, and records artifact sizes,
input/artifact SHA-256 hashes and warm p50/p95 timings. These are synthetic
engineering checks, not trained embeddings or a speedup claim. Compression and
decompression can be slower than direct mmap reads. See the recorded
[smoke result](engram_rowstore_smoke_20261018.md).

To pack an existing raw table with 128 float32 components per row (512 bytes):

```bash
python3 tools/lossless_row_store.py \
  --input /path/to/table.f32 \
  --output /path/to/table.mer \
  --row-bytes 512 --block-rows 256
```

No dtype conversion is performed. The caller owns dtype, shape, byte order,
checkpoint export and tensor reconstruction. Block compression uses CPU time;
mmap maps compressed bytes rather than directly usable float tensors.

## Engram adapter experiment

For an immutable inference checkpoint, the next integration should operate at
`MultiHeadEmbedding.forward`'s gather boundary:

1. Freeze a checkpoint and record its hash, table shape/dtype and tokenizer/hash
   configuration. Preserve all rows for all heads and layers.
2. Export each row's exact stored bytes, retaining the global IDs obtained from
   the original per-head offsets. Do not replace hash addressing.
3. Compare ordinary gathers and reconstructed rows for identical ID batches,
   including repeated/colliding keys, boundaries and arbitrary unseen queries.
4. Verify byte equality, then identical embedding outputs before testing
   gated outputs/logits under controlled deterministic execution.
5. Measure artifact size, forwarding/index overhead, cold load, RSS, warm and
   cold p50/p95, batch throughput and CPU-to-GPU transfer. Compare with direct
   mmap and ordinary tensor gathers; disclose losses in latency or footprint.

An optional [CPU gather adapter](engram_gather_adapter.md) now supports this
interface; its verification uses random weights and the official hash/gather
definitions, without tokenizer downloads. No trained-checkpoint comparison is
completed here. Exact identity,
lossless storage and efficient queries are separate claims: proving one does
not establish the others. The paper's FastText numbers cannot establish Engram
memory savings.

## Legacy export audit

At `me_fasttext` base commit `db9b83d`,
[`datrie_foreach_cal`](../src/fasttext.cc) computes
`min(||a||, ||b||) / max(||a||, ||b||)` and merges when that value is at least
`0.99`. It computes no dot product. For `[1,0]` and `[-1,0]` the norm ratio is
1, while cosine similarity is -1. Similar norm is not similar direction.
The callback does not explicitly verify a prefix/suffix relationship either;
trie traversal and reversed keys alone are not that check.

The callback's row strides and loops use literal `128`, and export loops also
copy 128 components. This is not dimension-independent. These observations
come from source inspection, not a new large-corpus run. The paper describes
structurally constrained cosine merging; that is not implemented by the
published callback as inspected.

Accordingly, the legacy `.z` export is approximate and requires correction and
separate reproduction before claims about preservation can be made. The new
byte-exact reference is independent and does not repair or silently reinterpret
legacy exports. Pure row renumbering/layout can preserve retained bytes;
approximate row merging cannot be called lossless.

## Next reading

- [Native C ABI and model integration](native_ngram_memory.md): batched CPU
  gather directly into PyTorch output and model/runtime compatibility boundaries.
- [Trained structural similarity](trained_ngram_similarity_20261018.md): primary
  learned n-gram evidence; random rows are interface controls only.
- [Engram promotion copy](engram_promotion.md): English and Chinese posts.
- [PyTorch gather adapter](engram_gather_adapter.md): unchanged head IDs and tensor bytes.
- [Compression bridge](model_compression_engram_bridge.md): measurement questions.
- [Benchmark protocol](benchmark_protocol.md): public-corpus reporting.
- [Paper](https://arxiv.org/abs/2506.01254): original FastText research context.
