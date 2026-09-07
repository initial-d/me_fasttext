# Retrieval comparison matrix

Use this matrix when comparing `me_fasttext` with FastText, dense embedding
models, vector indexes, or first-stage retrieval systems. It keeps the project
in its strongest lane: compact lexical embedding inference for OOV-heavy text.

## Positioning matrix

| System type | Main strength | Useful comparison question | What to report |
| --- | --- | --- | --- |
| Original FastText `.bin` | Mature subword embedding baseline | What deployment cost comes from loading the full training artifact? | Model size, peak RSS, cold load time, downstream metric |
| Hash-free FastText variant | Exact n-gram identity without compression | What does collision-free identity cost before export? | N-gram count, matrix size, memory, quality delta |
| `me_fasttext` `.z` | Compact mmap lexical serving artifact | How much cost is recovered after row sharing and mark-compact rewriting? | `mmap_bytes`, cold mmap load, p50/p95 query latency, OOV coverage |
| Quantized FastText | Established compact FastText baseline | Does quantization beat or complement structure-aware row sharing? | Artifact size, quality delta, latency, implementation constraints |
| BM25 or lexical inverted index | Strong exact-token retrieval baseline | Does subword embedding help rare, misspelled, or unseen terms? | Recall@k, MRR, OOV slice metrics, candidate-set size |
| Dense embedding model | Strong semantic retrieval and reranking | Is a small lexical layer useful before dense reranking? | Candidate recall before reranking, dense-stage latency saved |
| ANN or vector database | Large-scale nearest-neighbor infrastructure | Is the lexical vector layer small enough to feed or complement ANN search? | Index build cost, query latency, recall/quality, memory |

## Recommended report shape

For a report that others can cite, pair one deployment metric with one quality
metric. A compact artifact is only interesting if it preserves enough task
quality for the use case.

```markdown
## Retrieval comparison

- Commit:
- Corpus / query slice:
- OOV rate:
- Baselines:
- Hardware:

| Method | Artifact | Cold load | Query p50 | Query p95 | OOV coverage | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FastText `.bin` | | | | | | |
| me_fasttext `.z` | | | | | | |
| BM25 / lexical baseline | | | | | | |
| Dense reranker or embedding model | | | | | | |

Notes:
-
```

## Conservative claims

Good claims for this project:

- exact subword identities are useful before compression;
- a compact mmap artifact can reduce lexical embedding serving cost;
- OOV-heavy retrieval should be measured on rare-token query slices;
- mark-compact row rewriting is a useful model-layout analogy.

Claims to avoid:

- replacing dense transformer embeddings;
- replacing vector databases;
- serving as a complete RAG framework;
- being a general-purpose garbage collector or trie library.
