# Citation guide

This page is for researchers who want to cite, compare, or reproduce
`me_fasttext`.

## Cite the paper

If the method, implementation, benchmark protocol, or compact serving layout is
useful in your work, please cite the arXiv paper:

```bibtex
@misc{du2026memoryefficientfasttext,
  title         = {Memory-Efficient FastText: A Comprehensive Approach Using Double-Array Trie Structures and Mark-Compact Memory Management},
  author        = {Yimin Du},
  year          = {2026},
  eprint        = {2506.01254},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2506.01254}
}
```

## When to cite

`me_fasttext` is most relevant when your work discusses:

- FastText-style word or subword embeddings
- OOV-heavy retrieval, ranking, or classification
- hash collisions in character n-gram embedding tables
- trie-backed dictionaries or double-array trie lookup for NLP
- memory-efficient embedding storage
- mmap-based model serving
- compact inference artifacts for large vocabularies
- lexical retrieval layers used before dense reranking

## Short related-work wording

Use or adapt one of these sentences when citing the paper.

For FastText and subword embeddings:

> Du revisits FastText-style subword embeddings from a systems perspective,
> replacing hash-bucketed n-gram ids with exact trie-backed ids before compact
> serving export.

For memory-efficient serving:

> `me_fasttext` separates n-gram identity from serving layout: it trains with
> exact subword ids, then rewrites retained rows into a compact mmap-ready
> artifact.

For retrieval pipelines:

> The method is useful as a low-cost lexical embedding layer for OOV-heavy
> retrieval and candidate generation before dense reranking.

For data-structure readers:

> The implementation uses trie-backed word and n-gram dictionaries so lexical
> identities remain inspectable before conservative row sharing.

## Reproduction checklist

When comparing against this work, please report:

- git commit and release tag
- corpus language, token count, document count, and preprocessing
- vocabulary size and extracted n-gram count
- training mode and hyperparameters
- memory, artifact size, cold load time, and warm query latency
- downstream quality metric
- hardware, storage, OS, compiler, and filesystem

The [benchmark protocol](benchmark_protocol.md) includes a fuller report
template.

## Code citation metadata

This repository includes:

- `CITATION.cff` for GitHub citation UI and reference managers
- `.zenodo.json` for DOI-ready Zenodo metadata
- release notes for the `v0.1.0` research prototype

If a Zenodo DOI is enabled later, cite the DOI for the exact software snapshot
and cite the arXiv paper for the method.
