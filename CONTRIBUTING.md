# Contributing

Thanks for taking a look at `me_fasttext`. This is a research prototype, so the
most valuable contributions are the ones that make its evidence easier to
understand, reproduce, or falsify.

## Best first contributions

The highest-impact contribution is a benchmark report on a public or clearly
described corpus. Positive, negative, and partial results are all useful when
they include enough context for another reader to compare them fairly.

Good first contributions include:

- a public-corpus benchmark report using the
  [evaluator quickstart](docs/evaluator_quickstart.md);
- a retrieval, ranking, classification, or OOV-heavy serving result;
- a negative result showing where the compact `.z` path does not help;
- a build, portability, or documentation fix;
- a small code patch with a clear before/after behavior.

Please do not include private corpora, secrets, internal hostnames, proprietary
logs, or credentials. Aggregate metrics and corpus manifests are enough.

## Benchmark reports

Use the
[benchmark result issue template](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml)
when possible. A useful report should include:

- exact `me_fasttext` commit SHA;
- corpus name, language, license status, split, and preprocessing;
- hardware, OS, compiler, filesystem, and storage type;
- build, training, export, and benchmark commands;
- `.bin`, `.vec`, and `.z` artifact sizes;
- `bench_ftindex` output for compact `.z` serving;
- one quality metric, or an explicit serving-only caveat;
- failures, omitted baselines, and anything unfavorable to the method.

Start with [`docs/evaluator_quickstart.md`](docs/evaluator_quickstart.md) for
the shortest path. Use
[`docs/first_public_corpus_benchmark.md`](docs/first_public_corpus_benchmark.md)
for the fuller report shape.

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

Keep claims narrow. `me_fasttext` is best framed as a FastText-derived compact
lexical embedding artifact for memory-sensitive serving, OOV-heavy text, and
first-stage retrieval or ranking. It is not a replacement for transformer
embeddings, a vector database, or a full RAG system.

See [`docs/artifact_card.md`](docs/artifact_card.md) and
[`docs/citation_guide.md`](docs/citation_guide.md) for wording that is easier
to cite without overclaiming.
