# Model-compression and Engram-style lookup bridge

This page gives model-compression and conditional-memory readers a precise way
to evaluate `me_fasttext` without turning it into an LLM memory claim.

`me_fasttext` is not Engram, not an MoE component, and not a transformer
architecture. It is a smaller lexical-memory compression baseline: keep word
and character n-gram identity exact with trie-backed ids, then export a compact
mmap-friendly artifact after conservative row sharing and mark-compact style id
rewriting.

## Why model-compression readers may care

Most compression writeups focus on reducing weights after the model has already
chosen its parameterization: quantization, pruning, distillation, low-rank
factorization, or table compression. `me_fasttext` is narrower but useful as a
counterpoint because it separates three questions:

1. **identity:** which word or character n-gram does this row represent?
2. **sharing:** which trained rows can safely share storage?
3. **serving:** what compact layout should be mapped at inference time?

That makes it a good benchmark target for compression reports that want to keep
identity and serving cost visible at the same time.

## Why Engram-style readers may care

Recent Engram-style conditional-memory work reopens an old systems question:
which linguistic patterns should be stored in indexed memory rather than
recomputed through dense neural paths?

`me_fasttext` answers that question only at the lexical-embedding layer:

| Question | `me_fasttext` scope |
| --- | --- |
| Memory unit | word rows and UTF-8 character n-gram rows |
| Addressing | exact trie-backed ids before compact export |
| Compression | prefix/suffix candidate sharing plus vector-similarity checks |
| Serving artifact | compact `.z` file with mmap-friendly vectors and tries |
| Quality guardrail | OOV coverage plus retrieval, ranking, or classification metric |

This is useful as an inspectable control point: if a larger conditional-memory
system claims that indexed n-gram-like memory helps, `me_fasttext` provides a
small non-transformer baseline where memory, identity, load time, lookup
latency, and task quality can all be measured directly.

## Benchmark questions

For model-compression or Engram-adjacent reports, prefer one of these questions:

- How much memory is paid for exact n-gram identity before compression?
- How much of that memory is recovered by compact export?
- Does the compact `.z` artifact reduce cold load time or RSS?
- What are p50 and p95 lookup latencies for in-vocabulary and OOV-heavy queries?
- Does compression preserve a retrieval, ranking, or classification metric?
- Which n-gram rows were shared, and can the decision be traced to lexical
  structure plus vector similarity?

## Minimal report shape

A useful report should include:

```yaml
corpus:
  name:
  language:
  line_count:
  token_count:
model:
  dim:
  minn:
  maxn:
  min_count:
  cutoff:
artifacts:
  fasttext_bin_size:
  compact_z_size:
  nwords:
  nngrams_before:
  nngrams_after:
serving:
  cold_load_seconds:
  warm_rss_mb:
  query_latency_p50_ms:
  query_latency_p95_ms:
quality:
  task:
  metric:
  baseline_value:
  compact_value:
```

Negative results are useful. A report that says "the compact artifact reduced
RSS but hurt OOV recall" is more valuable than an unqualified compression ratio.

## Safe wording

Use this:

```text
me_fasttext is a compact lexical-memory baseline for FastText-style subword
embeddings: it preserves explicit word and character n-gram identities before
compression, then serves the remaining rows from an mmap-friendly artifact.
```

For Engram-adjacent contexts:

```text
me_fasttext is not an LLM conditional-memory module, but it is a small,
inspectable example of the same systems pressure behind Engram-style lookup:
keep reusable n-gram-like memory indexed, compact, and separately measurable.
```

Avoid this:

- `me_fasttext` implements Engram.
- Engram is based on `me_fasttext`.
- trie lookup is a general replacement for learned conditional memory.
- the paper's compression ratio is universal.
- memory reduction alone is enough without latency and quality measurements.

## Where to continue

- [`conditional_memory_context.md`](conditional_memory_context.md): cautious
  Engram-adjacent positioning and source links.
- [`benchmark_protocol.md`](benchmark_protocol.md): memory, load-time, latency,
  and task-quality reporting.
- [`retrieval_oov_benchmark.md`](retrieval_oov_benchmark.md): OOV-heavy serving
  benchmark shape.
- [`retrieval_comparison_matrix.md`](retrieval_comparison_matrix.md): compare
  with FastText, BM25, dense embeddings, and ANN/vector-search systems.
- [`compression_comparison_checklist.md`](compression_comparison_checklist.md):
  focused evidence fields for sparse, quantized, and compact-serving
  comparisons.
- [`citation_guide.md`](citation_guide.md): cite the paper without overclaiming.
