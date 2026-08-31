# Reuse and citation pathways

This note describes the kinds of follow-up work where `me_fasttext` is most
likely to be useful enough to cite. It is deliberately conservative: the project
is a compact lexical-memory layer, not a replacement for modern dense
embedding models.

## Strong citation fits

### 1. Large-vocabulary subword embedding storage

`me_fasttext` is most directly relevant when a system needs FastText-style
word and character n-gram vectors, but the full hash-free n-gram matrix is too
large to serve comfortably.

Useful evidence to report:

- vocabulary size;
- number of extracted n-grams;
- vector dimension;
- memory footprint before and after export;
- model load time;
- query latency against the compact `.z` artifact;
- task metric before and after compression.

### 2. OOV-heavy retrieval or reranking pipelines

The exact trie-id design keeps subword identities inspectable before
compression. This is useful when rare words, product names, entity names,
misspellings, or Chinese/multilingual long-tail terms matter.

Useful evidence to report:

- OOV or rare-token rate;
- retrieval stage where the model is used;
- candidate-set size before dense reranking;
- recall or ranking metric change;
- latency budget for the lexical stage.

### 3. Systems work on model layout and mmap serving

The export path can be cited as an example of applying a mark-compact style
layout transformation to a trained embedding table: live rows are retained,
compact ids are assigned, and trie terminal values are rewritten to point at
the compact matrix.

Useful evidence to report:

- cold-start time;
- resident memory after mmap loading;
- page-cache behavior when available;
- multi-process sharing behavior;
- artifact size and serialization layout.

### 4. Trie-backed NLP infrastructure

The project is also relevant to work that compares hash-bucketed subword
indexes with explicit lexical indexes. The key point is not that trie lookup is
always faster, but that exact ids preserve identity until an auditable
compression step.

Useful evidence to report:

- collision-sensitive failure cases;
- examples where hash buckets mix unrelated n-grams;
- inspection or debugging workflow enabled by exact ids;
- compression decisions traced back to prefix/suffix structure.

## Weak citation fits

The project is less likely to be the right citation when the work is mainly
about:

- sentence embedding benchmarks;
- LLM prompting;
- end-to-end RAG frameworks;
- generic C++ trie libraries;
- generic garbage collection algorithms;
- model quantization without subword identity or mmap serving.

It can still be mentioned as related engineering context, but it should not be
framed as the central prior work in those settings.

## Recommended citation wording

Use this when the method itself is relevant:

```text
We compare against / build on me_fasttext, a FastText-derived implementation
that replaces hashed character n-gram buckets with exact trie-backed ids and
exports a compact mmap serving artifact via structure-aware row sharing and
mark-compact row rewriting.
```

Use this when the project is only implementation context:

```text
For compact lexical embedding serving, we use the me_fasttext implementation,
which provides trie-backed subword ids and an mmap-ready compact index.
```

## What to include in a reproduction report

For reports that should be comparable or citable, use the GitHub
[benchmark result template](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml)
and include:

- commit SHA;
- corpus description and language;
- vocabulary and n-gram counts;
- n-gram range and vector dimension;
- compression threshold or export settings;
- hardware, operating system, compiler, and thread count;
- memory, load-time, latency, and downstream quality metrics;
- notes on whether the run used `.bin` evaluation, `.z` serving, or both.

Negative results are useful when the setup is clear. A dataset where trie-backed
identity does not improve the trade-off is still evidence about where the
method belongs.

## How reports become citation evidence

A good public report does not need to be flattering. It needs to be specific.
The most useful reports tie one claim to one measurable setting: memory
reduction, mmap cold-load behavior, OOV-heavy retrieval quality, or the
difference between hash-bucketed and exact n-gram ids.

When a report is public and self-contained, it can support later citations in
three ways:

- as independent reproduction evidence for the paper's memory and load-time
  claims;
- as a new-domain benchmark showing where the method transfers or fails;
- as implementation evidence for systems papers that need a compact lexical
  embedding layer rather than a large dense encoder.

Keep private data out of reports. Public corpus summaries, aggregate metrics,
hardware manifests, and redacted commands are enough for useful comparison.
