# Native storage through the original FastText model and CLI

Dense **me_fasttext** `.bin` models can be converted into immutable native
serving bundles. `loadModel(path)` and existing query commands then use the
original dictionary and row IDs, with input vectors stored in compressed native
blocks. The production DATrie prefix/suffix planner and mark-compact forwarding
are part of this conversion. [Actual trained-model results and limits](fasttext_native_evidence_20261031/README.md)
include a historical-main comparison and a matched identity-layout control.

## Export and query

The optional facade and bundled DATrie archives require Linux x86-64. Build from
this repository; the standalone memory wheel does not bundle the legacy trainer.

```bash
make fasttext-native native structural-native

# Use a NEW output directory. Keep the original .bin for training/quantization.
python tools/export_fasttext_native_cli.py \
  --model model.bin --output-dir model.native \
  --fasttext-binary build/fasttext-native \
  --structural-library build/libmeft_structural.so \
  --native-library build/libmeft_rows.so --mode exact

# The same command and arguments accept either a .bin or a bundle directory.
printf 'example\nexamples\n' | build/fasttext-native print-word-vectors model.native
printf 'an example sentence\n' | build/fasttext-native print-sentence-vectors model.native
```

For a supervised model, `predict`, `predict-prob` and `test` also use the native
input backend. `nn`/`analogies` still precompute resident word vectors and are not
a bounded-memory nearest-neighbor service. The original `make` targets remain
available. The original `.z`/FTIndex path retains its own format and reader.

After installing `meft-memory` from this checkout, the same converter is exposed
as `meft-fasttext-export` and `meft_memory.export_fasttext_model`. The binary and
structural-library paths remain explicit. No Torch dependency is required for
FastText export/query.

In an optional C++ source build, query calls keep their signatures:

```cpp
fasttext::FastText model;
model.loadModel("model.native");
fasttext::Vector word(model.getDimension());
model.getWordVector(word, "example");
```

Link `native/fasttext_bundle.cc` and the C row reader with
`-DMEFT_NATIVE_FASTTEXT`, as the Make target does. Recompile C++ callers: retained
source signatures are not an old binary ABI guarantee.

## Fidelity, compatibility and memory

- `layout` retains every row; `exact` additionally shares byte-identical
  structurally related rows when present. Both verify the full reconstructed
  input SHA-256, including untrained rows. The source `.bin` is never rewritten.
- `joint` is opt-in approximate sharing with cosine and norm-ratio guards.
  Its manifest separates source bytes from reconstructed representative bytes.
  It does not inherit lossless output guarantees.
- The dictionary looks up the **original registered IDs** in `keys.das`;
  `table.men` resolves their physical rows through forwarding. Aggregation
  preserves the original request order/multiplicity and float32 adds, across
  gather chunks. The FP64 native bag reducer is deliberately not used here.
- Loaded dictionaries now skip unknown subwords instead of registering rows
  beyond the trained input matrix. Direct subword helpers use real DATrie IDs;
  missing keys return `-1`/zero. Empty or zero-vector string sentence queries
  return zero instead of NaN. These fix prior undefined/invalid behavior;
  equivalence is claimed for valid historical queries, not those bugs.
- The default export caps decoded blocks at 8,192 bytes, uses a 4 MiB decoded
  block cache and gathers at most 256 rows before ordered accumulation. These
  are configurable via `--max-block-bytes`, `--cache-bytes` and
  `--query-chunk-rows`. A block's row count also depends on vector dimension.
- Input is frozen and each handle is single-worker. `saveModel`, legacy
  `saveIndex`, `quantize` and `getInputMatrix` reject native input; reload a dense
  model for those operations. Starting a new `train(args)` clears the backend.
- The dictionary, its legacy 200 MB unused word hash allocation, exact key
  index and dense output matrix remain resident. Neither loader builds an
  inference negative-sampling table. Cache budget is not a total RSS limit;
  touched mmap pages remain file-backed memory.

The bundle contains `table.men`, `keys.das`, original args/dictionary/output in
`model.meta`, reader settings and audit manifests. Open only trusted,
self-produced bundles on little-endian IEEE float32 Linux x86-64: the legacy
metadata/index loader is not an untrusted-input parser. The exporter records
checksums, but the C++ facade does not authenticate the entire bundle at open.
Conversion holds full source rows and planning maps offline.

This converter supports this repository's dense, unquantized, unpruned models.
Official fastText `.bin`, quantized `.ftz`, and old `.z` files are separate
formats. Engram hash collisions do not provide this unique string-key registry;
the numeric-row adapter retains its separate integration boundary.
