# NGramStore engine quickstart

Start with the storage engine, then choose FastText or Engram as the application.
Use an existing learned table; deterministic fixtures establish correctness only.

## Install and pack numeric rows

From a Linux/macOS source checkout with a C compiler and zlib headers:

```bash
python -m pip install .
ngram-store pack --input vectors.f32 --dim 256 --output memory.native
ngram-store inspect --store memory.native
ngram-store query --store memory.native --ids 0 42 0
```

`vectors.f32` must be a nonempty, contiguous little-endian float32 table.
`--dim` is its row width in floats. Output must be a fresh directory. No dtype
conversion or rounding occurs. All reconstructed row bytes are verified before
publication. `--deduplicate` optionally shares identical row bytes with an
index that grows with the number of unique rows. Default packing omits it.

Use `query --hex` for signed-zero/nonfinite payload inspection. Numeric JSON
output requires finite values. The query keeps original IDs, order and duplicates.
`--max-block-bytes` controls decoded block width; one indivisible row can exceed
it. `--cache-bytes` controls query cache payload, not total process memory.

The wheel distribution remains `meft-memory`; new code imports `ngram_store`.
Existing imports are aliases of the same API, not independent implementations.
`python -m ngram_store` also runs directly from a checkout; pass
`--native-library build/libmeft_rows.so` after `make native` if not installed.
On macOS use `make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib`.

## Registered string n-grams

On Linux x86-64, the optional original DATrie bridge is built with:

```bash
make structural-native
ngram-store pack --input vectors.f32 --dim 256 --output keyed.native \
  --keys keys.json --structural-library build/libmeft_structural.so --mode exact
ngram-store query --store keyed.native --keys '<memory>' '<vector>' \
  --structural-library build/libmeft_structural.so
```

`keys.json` maps canonical original row IDs to unique registered strings, for
example `{"0":"<memory>","42":"<vector>"}`. Unkeyed rows retain their identities.
`layout` keeps all rows; `exact` shares only byte-identical structural pairs;
`joint` changes rows under cosine/norm-ratio guards. Generic byte dedup and
structural representative selection are separate options.

A many-to-one hash bucket registry does not provide this key-to-row contract.
Engram's numeric/hash path works through the same row engine without assuming
prefix/suffix relationships. Open only trusted self-produced DATrie files.

## Choose an application

- [FastText prepare/train/export/query](fasttext_native_bundle.md): includes a
  real tiny text-training smoke path; the current trainer requires positional/source
  pairs rather than ordinary upstream FastText training input.
- [Engram demonstration](engram_quickstart.md): preserve existing hashes and
  head offsets, source-pin the demo and verify full module output.
- [TinyEngram/Qwen checkpoint](tinyengram_qwen_evidence_20261030/README.md):
  convert and verify actual trained memory weights with the existing backbone.

## Report a useful result

Record the exact commit, command, data and split, source/artifact hashes, dtype,
row count, compiler/Python/PyTorch versions, CPU/GPU and storage environment.
Compare with raw mmap/dense lookup using the same rows and trace. Report whole
artifact bytes, p50/p95 latency, load time, anonymous/file-backed RSS, cache and
block settings, output fidelity and task quality separately. Name approximate
thresholds and denominators. Include slow, failed and no-benefit cases.

[Current results](index.md) include strong negative controls. Legacy lexical
retrieval protocols remain in [the compatibility guide](legacy_fasttext.md).
