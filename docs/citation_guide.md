# Citation guide

This page is for researchers who want to cite, compare, or reproduce
NGramStore (formerly me_fasttext).

For a one-page summary of the artifact, including fit, claims, evidence, and
non-goals, see [`artifact_card.md`](artifact_card.md).

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

## Cite the native n-gram artifact

NGramStore is the renamed repository, with the original paper title and method
attribution retained. The C gather ABI, immutable PyTorch adapters and Engram-related experiments are
later software work. Cite their exact repository commit/report alongside the
FastText paper when relevant; do not attribute the new results to that paper.
Replace the bracketed commit below with the full SHA used in your experiment:

```bibtex
@misc{du2026mefasttextngramstorage,
  author       = {Yimin Du},
  title        = {NGramStore: Compression, Storage and Exact Lookup for Learned N-Gram Vectors},
  year         = {2026},
  howpublished = {Software and reproducibility artifact},
  url          = {https://github.com/initial-d/ngram-store},
  note         = {Commit [full SHA]; report [report path and date]}
}
```

This is a repository/software citation, not a new publication or minted DOI.
The [Engram quickstart](engram_quickstart.md) connects code, source-pinned
verification, trained results and comparison requirements.

Suggested wording for the tested boundary:

> The NGramStore engine provides immutable storage/gather adapters that
> preserve original hash IDs, head offsets and row bytes. Small source-pinned
> Engram demo checks also preserve gated module output. Separate trained
> TinyEngram/Qwen CPU/CUDA checks and timing controls do not establish official
> trained DeepSeek Engram or production serving results.

For approximate compression, name the actual trained table/corpus, threshold,
norm guard, byte denominator and distortion metric. Do not describe the
character-ngram reference's compression factor as an Engram model result.

## When to cite the method

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
template. The [citation evidence ladder](citation_evidence_ladder.md) explains
which reports are strong enough to link from README-level docs or reuse as
paper/report evidence.

## Code citation metadata

This repository includes:

- `CITATION.cff` for GitHub citation UI and reference managers
- `.zenodo.json` for DOI-ready Zenodo metadata
- release notes for the `v0.1.0` research prototype

If a Zenodo DOI is enabled later, cite the DOI for the exact software snapshot
and cite the arXiv paper for the method.
