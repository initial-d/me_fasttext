# Outreach notes

`me_fasttext` is easiest to explain when it is positioned as practical NLP
infrastructure, not as a replacement for transformer embeddings.

## Short positioning

Memory-efficient FastText for large-vocabulary, OOV-heavy NLP: exact trie-based
subword ids, conservative n-gram row sharing, and mmap serving for lower-memory
lexical retrieval and feature generation.

## When to mention it

- Large vocabularies where FastText-style OOV behavior is still useful.
- Chinese, multilingual, entity-heavy, or noisy text where long-tail words are
  common.
- Retrieval pipelines that need a cheap lexical signal before a dense reranker.
- Latency-sensitive services where loading a large embedding model is expensive.
- Systems papers or engineering discussions about model compression,
  memory-mapped serving, or trie-based indexes.

## When not to oversell it

- Do not call it a general replacement for modern embedding models.
- Do not claim universal speedups without a corpus and hardware manifest.
- Do not pitch it as a complete RAG framework.
- Do not submit to unrelated agent or LLM lists unless they accept retrieval
  infrastructure.

## GitHub issue or pull request blurb

```markdown
Hi, thanks for maintaining this list.

Would you consider adding `me_fasttext`?

It is a FastText-derived C++ research prototype for memory-sensitive
large-vocabulary NLP. Instead of hashing character n-grams into fixed buckets,
it assigns exact trie ids, then exports a compact mmap serving index through
conservative n-gram row sharing and mark-compact layout.

Repo: https://github.com/initial-d/me_fasttext
Paper: https://arxiv.org/abs/2506.01254

I think it fits best under word embeddings, retrieval infrastructure, or
efficient NLP systems. Happy to adjust the wording if another section is more
appropriate.
```

## Short social post

```text
I open-sourced me_fasttext, a memory-efficient FastText-derived prototype for
large-vocabulary NLP.

The idea is simple: replace hash-bucketed subword ids with exact trie ids, then
compress learned n-gram rows into a compact mmap serving index.

Useful for OOV-heavy retrieval, entity-heavy search, and low-latency lexical
features when transformer embeddings are too expensive for the first stage.

Code: https://github.com/initial-d/me_fasttext
Paper: https://arxiv.org/abs/2506.01254
```

## Suggested targets

Submit only where the project clearly fits:

| Target type | Angle |
| --- | --- |
| Awesome NLP lists | Word embeddings and efficient representation learning. |
| Embedding lists | Subword embedding infrastructure, not sentence-level SOTA. |
| RAG lists | Cheap lexical prefilter or fallback retrieval layer. |
| Systems forums | mmap serving, model footprint, trie indexing, cold-start cost. |

Start with one or two high-fit submissions. If maintainers respond well, reuse
their preferred wording elsewhere.

## Distribution map

| Community | Fit | Suggested action | Notes |
| --- | --- | --- | --- |
| FastText / word embedding lists | High | Submit concise project entry. | Lead with OOV behavior and subword ids. |
| NLP and ML library lists | High | Submit to C++ or text embedding sections. | Keep wording short and disclose authorship. |
| Trie / double-array trie communities | Medium | Share a technical note, not a generic listing. | The repo uses trie indexes inside an NLP system; it is not a standalone DA-trie library. |
| Memory management / systems communities | Medium | Write an engineering note about mark-compact export and mmap layout. | Frame it as compact model layout, not as a garbage collector. |
| RAG / retrieval communities | Medium | Mention as a lexical prefilter or OOV fallback layer. | Avoid positioning it as a complete RAG framework. |
| Functional programming communities | Low | Do not submit as-is. | The implementation is imperative C++; only discuss deterministic pipeline design if asked. |
| General C++ awesome lists | Conditional | Submit only if project age and list rules allow. | Some lists reject repositories created after a cutoff date. |

## Technical note outline

Use [engineering_note.md](engineering_note.md) when sharing with systems,
data-structure, or memory-management readers. If adapting it into a shorter
post, keep this outline:

1. Problem: FastText hash buckets bound memory but make subword identity opaque.
2. Data structure: exact word and n-gram ids through trie-backed dictionaries.
3. Compression: merge only structurally related n-gram rows with close vectors.
4. Layout: mark live rows, assign dense ids, and serialize an mmap-friendly
   `.z` file.
5. Serving: avoid reconstructing heavyweight training-time structures at
   startup.
6. Caveat: compression ratio and latency depend on corpus, n-gram range,
   dimension, and OS page-cache behavior.
