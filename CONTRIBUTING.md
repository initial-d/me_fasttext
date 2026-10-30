# Contributing

Thanks for taking a look at NGramStore. This is a research prototype, so the
most valuable contributions are the ones that make its evidence easier to
understand, reproduce, or falsify.

Please follow the [code of conduct](CODE_OF_CONDUCT.md). In short: keep
discussion respectful, evidence-oriented, and careful with private data.

## Best first contributions

The highest-impact contribution is a benchmark report on a public or clearly
described corpus. Positive, negative, and partial results are all useful when
they include enough context for another reader to compare them fairly.

Good first contributions include:

- a public-corpus benchmark report using the
  [evaluator quickstart](docs/evaluator_quickstart.md);
- an
  [evaluation question](https://github.com/initial-d/ngram-store/issues/new?template=evaluation-question.yml)
  about build, benchmark setup, corpus preparation, baselines, or result
  interpretation;
- a
  [showcase or use-case note](https://github.com/initial-d/ngram-store/issues/new?template=showcase.yml)
  for citations, list entries, comparisons, integrations, or deployment notes;
- a retrieval, ranking, classification, or OOV-heavy serving result;
- a negative result showing where the compact `.z` path does not help;
- a build, portability, or documentation fix;
- a small code patch with a clear before/after behavior.

Please do not include private corpora, secrets, internal hostnames, proprietary
logs, or credentials. Aggregate metrics and corpus manifests are enough.

## Benchmark reports

Use the
[benchmark result issue template](https://github.com/initial-d/ngram-store/issues/new?template=benchmark-result.yml)
when possible. A useful report should include:

- exact NGramStore commit SHA;
- corpus name, language, license status, split, and preprocessing;
- hardware, OS, compiler, filesystem, and storage type;
- build, training, export, and benchmark commands;
- full input/table/index/metadata bytes and the exact lossless/approximate mode;
- dense/raw-mmap controls, native query latency and anonymous/file-backed RSS;
- one quality metric, or an explicit serving-only caveat;
- failures, omitted baselines, and anything unfavorable to the method.

Start with [`docs/evaluator_quickstart.md`](docs/evaluator_quickstart.md) for
the shortest path. Use
[`docs/first_public_corpus_benchmark.md`](docs/first_public_corpus_benchmark.md)
for the fuller report shape.

## Showcase and visibility notes

Use the
[showcase issue template](https://github.com/initial-d/ngram-store/issues/new?template=showcase.yml)
for lighter-weight adoption evidence: citations, public list entries,
integration notes, talks, blog posts, retrieval experiments, or compact-serving
deployment notes.

These notes help future readers decide whether the project is relevant before
they run a full benchmark. Please keep them factual: say what was tried, what
fit, what did not fit, and whether the evidence is public.

## Code and documentation changes

Before opening a pull request:

1. Keep the change narrow and explain the user-facing or research-facing value.
2. Run the relevant command locally.
3. Include the command output or a short verification note in the PR.
4. Avoid unrelated formatting, generated artifacts, or binary churn.

For Python helper changes, run:

```bash
python3 -m unittest discover -s tests
```

For C++ changes, build on Linux when possible:

```bash
make opt
```

If a full build is not possible on your machine, say so and include the compiler
or platform limitation.

## Scope and positioning

NGramStore is a compression, storage and exact-query backend for learned n-gram vectors. FastText and Engram/TinyEngram are applications with different training and addressing contracts. Numeric/hash queries use the native row engine; structural sharing requires registered string keys. Lossless bytes, approximate distortion, query cost and task quality need separate evidence.
See [`docs/artifact_card.md`](docs/artifact_card.md) and
[`docs/citation_guide.md`](docs/citation_guide.md) for wording that is easier
to cite without overclaiming.
