# First public corpus benchmark

This page turns the current benchmark request into one concrete external run.
The goal is to produce the first public, citable `me_fasttext` report outside
the original paper setting.

The benchmark should be small enough for a contributor to finish, but specific
enough to be useful as evidence for memory-efficient FastText-style inference.

## Target question

Does exact trie-backed subword identity plus compact `.z` export reduce serving
cost on a public corpus while preserving useful OOV behavior?

This is intentionally narrower than "is `me_fasttext` better than every modern
embedding model?" The useful comparison is:

1. original FastText-style `.bin` model;
2. `me_fasttext` compact `.z` serving artifact;
3. task or retrieval quality before and after compact export.

## Good first corpus choices

Pick one corpus that is public, redistributable, and easy to describe. Good
first choices are:

| Corpus type | Why it fits | Useful slice |
| --- | --- | --- |
| News classification corpus | FastText-style supervised text classification is a natural baseline. | Rare names, organizations, locations, and misspellings in held-out text. |
| Wikipedia or encyclopedia text | Large enough to create many character n-grams. | Entity-heavy query/title lines. |
| Product, title, or catalog text | OOV and long-tail names are common. | Brand names, model numbers, multilingual titles. |
| Chinese news or encyclopedia text | Matches the paper's long-tail Chinese motivation. | New words, names, and domain entities. |

Before publishing a report, verify that the corpus license allows redistribution
or publish only aggregate metrics and a corpus manifest.

## Minimal run

Build the binaries:

```bash
make opt
```

Train a model with explicit n-gram ids:

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

Record:

- commit SHA;
- corpus name, license, language, document count, and token count;
- generated `runs/public-meft.bin`, `.vec`, and `.z` sizes;
- `Maximum resident set size` and elapsed training time;
- any build or portability patches.

## Query slice

Create `queries.txt` with at least 10,000 short lines when possible. Include
three named slices:

| Slice | Examples | Why it matters |
| --- | --- | --- |
| `in_vocab` | common words, common titles | Baseline lookup latency. |
| `oov_heavy` | misspellings, rare names, new compounds | Tests whether character n-grams still cover unseen words. |
| `entity_heavy` | products, organizations, biomedical or code names | Tests long-tail lexical retrieval behavior. |

Then run:

```bash
./bench_ftindex runs/public-meft.z queries.txt 1 3
```

Paste the printed Markdown table into the benchmark issue template.

## Quality metric

Pair the serving metrics with at least one quality metric:

| Task | Acceptable first metric |
| --- | --- |
| Classification | Accuracy, macro-F1, or per-class F1 before and after compact export. |
| Retrieval | Recall@k, MRR, NDCG, or candidate recall before dense reranking. |
| Embedding sanity check | Nearest-neighbor examples plus OOV coverage, if no labeled task exists. |

If no quality metric is available, mark the report as a serving-only benchmark.
Serving-only reports are useful, but they should not be used to claim task
quality is preserved.

## Report template

Use the GitHub
[benchmark result template](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml)
and include this compact table:

```markdown
| Method | Artifact | Cold load | RSS | Query p50 | Query p95 | OOV coverage | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FastText `.bin` | | | | | | | |
| me_fasttext `.z` | | | | | | | |
```

Negative results are welcome. A report showing that a corpus has too few
repeated n-grams, too little OOV pressure, or no quality-preserving compression
is still useful evidence about where the method belongs.
