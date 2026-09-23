# Compression comparison checklist

This checklist is for readers who want to compare `me_fasttext` with model
compression, sparse lookup, quantized embedding, or lexical-memory baselines.
It keeps the comparison narrow: `me_fasttext` is a compact FastText-style
lexical embedding artifact, not a general LLM compression method.

## What counts as a useful comparison

A useful comparison should answer one concrete question:

> Does exact n-gram identity plus compact mmap serving buy enough memory,
> startup, or OOV-serving value on this corpus to justify the extra indexing
> machinery?

Good reports can be positive, negative, or mixed. A result that says the compact
index saves memory but loses too much retrieval quality is still useful.

## Minimum evidence

Include these fields so another evaluator can tell what was actually compared:

| Area | Required detail |
| --- | --- |
| Commit | `me_fasttext` commit SHA and whether the tree was clean. |
| Corpus | Name, language, license, document count, token count, preprocessing, and split. |
| Model settings | Training mode, dimension, `minn`, `maxn`, `minCount`, epochs, and thread count. |
| Compression settings | Compact export path, row-sharing threshold if changed, and generated `.z` size. |
| Baselines | Original FastText, quantized FastText, hash-free uncompressed, BM25, dense embeddings, or an explicit reason for omission. |
| Hardware | CPU, memory, storage type, OS, compiler, and filesystem. |
| Serving metrics | Cold load time, resident memory, p50/p95 latency, and OOV subword coverage. |
| Quality guardrail | Retrieval, ranking, classification, or feature-serving metric; otherwise state `serving-only`. |

## Recommended table

```markdown
| Method | Artifact size | Cold load | RSS | Query p50 | Query p95 | OOV coverage | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FastText | | | | | | | |
| Quantized FastText | | | | | | | |
| me_fasttext .z | | | | | | | |
```

If a baseline cannot be run because it is too large, fails to build, or does not
support the same corpus, report that explicitly instead of dropping it silently.

## Comparison boundaries

Use careful wording:

- compare compact lexical embedding serving, not full transformer inference;
- keep memory, load time, latency, and quality in the same report;
- separate exact identity before compression from compressed serving after
  export;
- report OOV-heavy and entity-heavy query slices separately when possible;
- avoid treating the paper's large-corpus compression ratio as universal.

Avoid these claims:

- `me_fasttext` is an LLM compression framework;
- trie lookup replaces pruning, quantization, or learned conditional memory;
- smaller artifacts are better without a latency and quality check;
- one corpus is enough to claim general model-compression behavior.

## Suggested issue path

Use the
[compression comparison issue template](https://github.com/initial-d/me_fasttext/issues/new?template=compression-comparison.yml)
when the report is specifically about comparing `me_fasttext` with sparse,
quantized, or compact-serving baselines.

Use the general
[benchmark result template](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml)
when the report is a broader retrieval, classification, or public-corpus run.
