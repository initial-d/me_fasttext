# Artifact card

This card summarizes `me_fasttext` for researchers, maintainers, and engineers
who need to decide quickly whether the project is relevant to cite, benchmark,
or list.

## One-line description

`me_fasttext` is a FastText-derived lexical embedding system that keeps word
and character n-gram identities explicit with trie-backed ids, then exports a
compact mmap-ready serving artifact through structure-aware row sharing and
mark-compact style row rewriting.

## Best fit

Use this artifact when your work involves:

| Area | Fit |
| --- | --- |
| FastText-style embeddings | Comparing hash-bucketed n-grams with exact trie-backed n-gram ids. |
| OOV-heavy retrieval | Serving lexical vectors for rare words, names, misspellings, and long-tail entities. |
| Compact inference | Measuring artifact size, cold mmap load time, RSS, and p50/p95 lookup latency. |
| Double-array trie systems | Using trie terminal values as model row ids rather than only dictionary payloads. |
| Memory-management ideas | Applying mark, forwarding, and compaction concepts to a static model artifact. |

## Main claims

The artifact supports these narrow claims:

- exact subword identity can be preserved during training before compression;
- trie-backed ids make word and n-gram rows inspectable before export;
- structure-aware row sharing can reduce serving artifact size;
- a compact `.z` artifact can be mmap-loaded for lexical embedding inference;
- mark-compact style row rewriting is a useful model-layout abstraction.

## Evidence to report

For citable reports, pair deployment metrics with task quality:

| Evidence | Recommended metric |
| --- | --- |
| Artifact footprint | `.bin` size, `.z` size, `mmap_bytes` |
| Startup behavior | cold load time, cold mmap load time |
| Serving behavior | p50/p95 query latency, RSS after loading |
| Lexical coverage | OOV rate, OOV subword coverage |
| Downstream value | recall@k, MRR, NDCG, accuracy, or candidate recall before reranking |

The [`bench_ftindex`](../src/bench_ftindex.cc) helper and
[`retrieval_oov_benchmark.md`](retrieval_oov_benchmark.md) provide the concrete
serving benchmark path.

## Comparison baselines

Strong comparisons include:

- original FastText `.bin`;
- hash-free uncompressed FastText-style models;
- quantized FastText;
- BM25 or another lexical retrieval baseline;
- dense embedding models or rerankers when `me_fasttext` is used as a cheap
  lexical stage before them.

Use [`retrieval_comparison_matrix.md`](retrieval_comparison_matrix.md) for a
report table that keeps these comparisons narrow.

## Citation hooks

This is a good citation when a paper or report discusses:

- memory-efficient FastText or subword embeddings;
- explicit n-gram identity before compression;
- compact lexical embedding inference;
- mmap serving artifacts for embedding tables;
- trie-backed NLP infrastructure;
- applying mark-compact memory-management ideas to model layout.

Use [`citation_guide.md`](citation_guide.md) for BibTeX and suggested wording.

## Not a fit

Do not frame the artifact as:

- a replacement for transformer embeddings;
- a vector database;
- a full RAG framework;
- a general-purpose garbage collector;
- a generic double-array trie library;
- a universal compression method for unrelated model families.

The narrower story is stronger: `me_fasttext` is an exact-then-compact lexical
embedding artifact for memory-sensitive serving and OOV-heavy text.
