# Cross-community positioning

`me_fasttext` is easiest to spread when each community sees the part of the
project that belongs to its own vocabulary. The same repository can be relevant
to FastText users, word-embedding researchers, double-array-trie readers,
memory-management people, and inference-optimization engineers, but the pitch
should change.

## Positioning map

| Community | Lead with | Useful citation hook | Avoid saying |
| --- | --- | --- | --- |
| FastText users | Exact character n-gram ids and compact serving artifacts. | A FastText-derived implementation that separates subword identity from post-training compression. | It replaces FastText in all settings. |
| Word embeddings | OOV-heavy lexical embeddings for rare words, entities, and multilingual text. | A compact lexical embedding layer with inspectable subword rows. | It competes with sentence-transformer quality benchmarks. |
| Double-array trie / datrie | Trie terminal values become model row ids, not only dictionary payloads. | DA-trie as part of embedding-table identity and serving layout. | It is a general-purpose trie library. |
| GC / dynamic memory management | Mark-compact as an offline model-layout transformation. | A practical case where mark, forwarding ids, and compaction reorganize trained embedding rows. | It implements a runtime garbage collector. |
| Inference optimization | Smaller artifact, mmap loading, lower cold-start cost, and p50/p95 lookup latency. | A compact-inference prototype for lexical embeddings before dense reranking. | It is an LLM inference engine. |
| Retrieval / RAG | Cheap lexical first-stage recall before dense reranking. | A low-memory OOV-aware candidate generation layer. | It is a complete RAG framework. |

## Strongest one-sentence version

`me_fasttext` is a FastText-derived compact-inference prototype that keeps word
and character n-gram identities explicit with double-array tries, then applies
a mark-compact-style export step to produce a smaller mmap-ready embedding
artifact.

## Community-specific blurbs

### FastText / word embeddings

```text
me_fasttext revisits FastText's hash-bucketed subword design: train with exact
trie-backed word and character n-gram ids, then compact the learned rows for
mmap serving. It is most useful for OOV-heavy lexical embeddings, entity search,
and long-tail multilingual text.
```

### Double-array trie / datrie

```text
me_fasttext uses double-array tries as model-index structures: trie terminal
values point to word and n-gram embedding rows, and those ids are rewritten
during compact export. The trie is not only a dictionary; it is part of the
embedding serving layout.
```

### GC / dynamic memory management

```text
me_fasttext applies a mark-compact-style idea outside a runtime GC: trained
embedding rows are treated as layout objects, retained rows receive forwarding
ids, and trie terminal values are rewritten to a dense compact matrix for
serving.
```

### Inference optimization

```text
me_fasttext is a compact lexical embedding inference layer: exact subword ids
for training and inspection, conservative row sharing after training, and a
single mmap-ready `.z` artifact for lower-memory serving and faster cold starts.
```

## Where to submit

Prefer narrow, high-fit submissions:

- word embedding and FastText lists;
- efficient NLP / model-compression lists;
- retrieval lists that accept lexical first-stage tools;
- GC or memory-management lists only when the entry is framed as an applied
  mark-compact model-layout case;
- data-structure communities only when the entry explains DA-trie terminal ids
  inside the model artifact.

Do not submit the same generic paragraph everywhere. A maintainer should be
able to see why the project belongs to that list in one sentence.
