# Benchmark protocol

This document describes a repeatable way to evaluate `me_fasttext` against
FastText-style baselines. The goal is to make reported numbers comparable
across machines and corpora, not to claim a universal speedup.

## What to measure

Report all of the following for each run:

| Metric | Why it matters |
| --- | --- |
| Peak resident memory | Captures the deployment cost of the full model or serving index. |
| Serialized model size | Captures artifact storage and transfer cost. |
| Cold load time | Captures restart, autoscaling, and batch job startup cost. |
| Warm query latency | Captures steady-state serving behavior after the OS page cache is warm. |
| OOV coverage | Captures whether rare and unseen words still receive useful subword vectors. |
| Task quality | Prevents compression-only wins that damage downstream accuracy. |

For reporting, prefer median plus p95 latency over a single average. For model
quality, include the original task metric and the dataset split.

## Baselines

Use at least two baselines when possible:

| Baseline | Purpose |
| --- | --- |
| Original FastText | Measures the cost and quality of the upstream hash-bucket design. |
| Hash-free uncompressed model | Separates the cost of exact n-gram identity from compression. |
| Quantized FastText | Compares against an established memory reduction method. |
| `me_fasttext` compact index | Measures the exact-then-compress serving path. |

If a baseline is omitted, state why. For example, a hash-free uncompressed model
may be too large for a single machine on very large corpora.

## Corpus manifest

Include a small manifest with every benchmark:

```yaml
corpus:
  name: example-corpus
  language: zh
  documents: 1000000
  tokens: 250000000
  train_split: train.txt
  eval_split: eval.txt
model:
  mode: skipgram
  dim: 300
  minn: 2
  maxn: 6
  min_count: 5
  epoch: 5
  thread: 24
hardware:
  cpu: "model name"
  memory_gb: 256
  storage: "NVMe SSD"
  os: "Linux kernel version"
```

The manifest matters because trie size, n-gram count, and compression ratio are
strongly corpus-dependent.

## Suggested command shape

```bash
make opt

/usr/bin/time -v ./fasttext skipgram \
  -input data/train.txt \
  -output runs/meft \
  -dim 300 \
  -minn 2 \
  -maxn 6 \
  -minCount 5 \
  -thread 24
```

Use the same training corpus and hyperparameters for every comparable run.
Capture `Maximum resident set size`, elapsed time, generated artifact sizes,
and the exact git commit.

## Serving benchmark

For serving, measure the compact `.z` index separately from the full `.bin`
model:

1. Drop or account for OS page cache before cold-load measurements.
2. Open the index with `ftindex_mmap`.
3. Run a fixed query file with at least 10,000 words or sentences.
4. Repeat after warmup and report median and p95 latency.
5. Record whether vectors are word-level or sentence-level.

For RAG or search prefiltering use cases, report recall-at-k or downstream
reranking quality in addition to raw latency.

## Paper reference point

The paper reports a Chinese large-vocabulary setting with 30.1M words and
287.4M extracted n-grams:

| Method | Memory | Load time | Storage |
| --- | ---: | ---: | ---: |
| Original FastText | 145.2 GB | 12.3 min | 89.4 GB |
| HashFree FastText | 287.4 GB | 28.7 min | 201.8 GB |
| Quantized FastText | 72.6 GB | 8.9 min | 44.7 GB |
| SVD Compression | 89.3 GB | 15.4 min | 62.1 GB |
| me_fasttext | 28.9 GB | 3.2 min | 18.6 GB |

Treat these as a reference point. New benchmark reports should include their
own corpus manifest, hardware manifest, and downstream quality metric.

## Minimal report template

```markdown
## Benchmark report

- Commit:
- Corpus:
- Language:
- Training command:
- Hardware:
- Baselines:

| Method | Memory | Load time | Storage | Query p50 | Query p95 | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FastText | | | | | | |
| me_fasttext | | | | | | |

Notes:
-
```
