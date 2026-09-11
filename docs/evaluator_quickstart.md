# Evaluator quickstart

This page is for readers who have cloned `me_fasttext` and want to produce a
useful first signal quickly. The goal is not a perfect benchmark. The goal is a
clear public run that another researcher or engineer can understand, compare,
or reproduce.

## What helps most

A useful report answers one narrow question:

> On this corpus and machine, does the compact `.z` mmap-serving artifact reduce
> memory or startup cost while preserving useful lexical behavior?

Report one of these outcomes:

| Outcome | Why it helps |
| --- | --- |
| Positive result | Shows where exact-then-compact lexical embeddings transfer. |
| Negative result | Shows where the method does not help or is too costly. |
| Partial result | Records build, corpus, or serving friction for the next evaluator. |

## Fifteen-minute orientation

1. Read the [artifact card](artifact_card.md) to confirm the project fits your
   use case.
2. Build the tools on Linux:

   ```bash
   make opt
   ```

3. Pick a public or redistributable plain-text corpus. Product titles, news
   text, encyclopedia titles, entity-heavy logs, and Chinese or multilingual
   text are especially useful.
4. Generate a manifest and query slices:

   ```bash
   python3 tools/prepare_public_benchmark.py \
     --input data/public_train.txt \
     --output-dir runs/public-benchmark \
     --name "example-public-corpus" \
     --language en \
     --license "public / CC / unknown"
   ```

5. Train a small first model:

   ```bash
   /usr/bin/time -v ./fasttext skipgram \
     -input data/public_train.txt \
     -output runs/public-meft \
     -dim 128 \
     -minn 2 \
     -maxn 6 \
     -minCount 5 \
     -thread 8
   ```

6. Measure the compact serving artifact:

   ```bash
   ./bench_ftindex runs/public-meft.z runs/public-benchmark/queries.txt 1 3
   ```

## Minimum report

Open a [benchmark result issue](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml)
with:

- exact commit SHA;
- corpus name, language, license status, and preprocessing;
- build and training commands;
- hardware, OS, compiler, filesystem, and storage type;
- `.bin`, `.vec`, and `.z` artifact sizes;
- `bench_ftindex` output;
- one quality metric, or an explicit serving-only caveat;
- anything that failed, surprised you, or made the method look bad.

If you cannot share the corpus, share aggregate metrics and the generated
`corpus_manifest.json` fields instead of private text.

## What not to claim

Keep the comparison narrow. `me_fasttext` is not a replacement for transformer
embeddings, a vector database, or a full RAG system. It is most useful as a
compact lexical embedding layer for memory-sensitive serving, OOV-heavy text,
and first-stage retrieval or ranking.

For a fuller protocol, continue with
[`first_public_corpus_benchmark.md`](first_public_corpus_benchmark.md) and
[`retrieval_oov_benchmark.md`](retrieval_oov_benchmark.md).
