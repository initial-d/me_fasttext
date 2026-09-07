# Retrieval and OOV serving benchmark

This benchmark is for readers who want to evaluate `me_fasttext` as a compact
lexical inference layer rather than as a general vector database. It measures
whether the mmap `.z` artifact can provide low-cost sentence or query vectors
for OOV-heavy retrieval, entity search, and first-stage ranking.

## Build the benchmark tool

```bash
make opt
```

This builds `bench_ftindex` together with the regular `fasttext`, `index`,
`cut_ngram`, and `print_ngrams` binaries.

## Prepare query slices

Use a plain text file with one query, title, product name, entity mention, or
short document per line:

```text
rare brand spelling
long tail entity name
中文 新词 实体
misspelled retrival query
```

For useful reports, include at least two slices:

| Slice | Purpose |
| --- | --- |
| In-vocabulary queries | Baseline lookup and sentence-vector latency. |
| OOV-heavy queries | Measures whether character n-grams still cover unseen words. |
| Entity-heavy queries | Tests long-tail product, company, biomedical, or code names. |
| Multilingual queries | Tests UTF-8 n-gram behavior. |

## Run

```bash
./bench_ftindex models/meft.z queries.txt 1 3
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `models/meft.z` | Compact mmap serving artifact exported by `me_fasttext`. |
| `queries.txt` | Query file, one query per line. |
| `1` | Warmup rounds. |
| `3` | Measured rounds. |

The tool prints a Markdown table:

```markdown
| metric | value |
| --- | ---: |
| cold_mmap_load_ms | 12.345 |
| mmap_bytes | 123456789 |
| dim | 128 |
| nwords | 1000000 |
| nngrams | 5000000 |
| queries | 10000 |
| total_tokens | 42000 |
| oov_tokens | 2100 |
| oov_rate | 0.050 |
| oov_subword_coverage | 0.982 |
| query_mean_us | 18.400 |
| query_p50_us | 15.200 |
| query_p95_us | 31.700 |
```

## What to compare

For a citable retrieval or inference report, compare at least:

1. the full `.bin` FastText-style model load path;
2. the compact `.z` mmap serving path;
3. the downstream retrieval or ranking metric for both paths.

Useful downstream metrics include recall@k, MRR, NDCG, classification accuracy,
or candidate-set recall before dense reranking. The benchmark table alone is
not enough: pair latency and memory with a quality metric.

## Reporting checklist

- Commit SHA.
- Corpus language and preprocessing.
- Training command and n-gram range.
- Query-slice manifest and OOV rate.
- Hardware, OS, compiler, filesystem, and storage type.
- Cold load time, mmap bytes, p50/p95 query latency, and OOV coverage.
- Retrieval, ranking, or classification quality before and after compact export.
- Any negative result or failure mode.

This evidence is the strongest bridge to vector-search and inference
optimization communities because it makes the project comparable to other
first-stage retrieval and serving systems.
