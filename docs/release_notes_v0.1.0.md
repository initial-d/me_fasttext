# v0.1.0 research prototype

This release packages the current `me_fasttext` prototype as a citable research
artifact for readers of the accompanying arXiv paper.

## Included

- FastText-compatible C++ training and inference commands.
- Exact trie-based word and character n-gram ids.
- Prefix/suffix n-gram row sharing with cosine-similarity checks.
- Mark-compact export into a single mmap serving index (`<output>.z`).
- Helper tools for compact-index generation and n-gram inspection.
- Benchmark protocol and benchmark issue template for external reports.
- Citation metadata for GitHub and reference managers.

## Intended use

Use this release as a starting point for memory-sensitive FastText-style
embedding experiments, large-vocabulary lexical retrieval, OOV-heavy text
features, and mmap serving investigations.

This is not presented as a drop-in replacement for transformer embedding models
or a complete RAG framework. It is a compact lexical memory layer that can be
used alongside modern retrieval and reranking systems.

## Notes

- Linux is the tested target.
- Rebuild binaries on the target machine before production use.
- Compression and latency depend on corpus, vocabulary, n-gram settings,
  hardware, filesystem, and OS page-cache behavior.
