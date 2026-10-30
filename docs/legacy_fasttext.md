# Historical me_fasttext compatibility

NGramStore is the renamed repository and shared n-gram vector backend.
FastText remains the original word-embedding application. Repository history,
issues, releases and original paper attribution are retained. Old GitHub URLs
redirect after rename; update local remotes to the canonical address:

```bash
git remote set-url origin https://github.com/initial-d/ngram-store.git
```

## Retained interfaces

- `meft-memory` installation and `meft_memory` imports continue; new callers
  use `ngram_store`. Both names expose the same runtime objects.
- Existing `meft-*` commands remain; `ngram-store` adds a common dispatcher.
- C symbols and MEFTNAT1 row bytes are unchanged. Rebuild C++ consumers after
  source updates; preserving source calls is not an old C++ binary ABI promise.
- FastText `.bin` and optional native bundle loading preserve valid historical
  queries. `.z` retains the separate FTIndex reader and its approximate contract.
- Existing `make opt` and `make debug` targets remain available on Linux.

## Train current source correctly

This repository's custom SGNS reader recognizes `p#` positions and
`query\x01source` pairs. Ordinary whitespace text and standard supervised-label
examples from upstream FastText are not a verified direct training path here.
Use [the tested preparation/training workflow](fasttext_native_bundle.md), then
retain `.bin` for training/quantization and export native for frozen queries.
Existing supervised models can be queried through the native facade.

The CLI trainer still also writes its legacy .z and trie dumps in the working
directory. Run in your own output directory. Re-exporting through the current
.z exporter uses explicit structure, cosine and norm-ratio guards; old .z files
keep their original contents. [Legacy contract and checks](legacy_merge_contract.md).

## Historical research pages

[Lexical retrieval comparisons](retrieval_comparison_matrix.md),
[legacy FTIndex benchmarks](retrieval_oov_benchmark.md), and older outreach and
release notes document the FastText application and historical positioning.
Their old project names are intentional history, not a second active engine.
For new reports, start from the [current engine guide](evaluator_quickstart.md)
and [evidence index](index.md). The paper title and BibTeX remain unchanged.
