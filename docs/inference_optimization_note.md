# Early compact inference ideas in me_fasttext

`me_fasttext` can be read as an early compact-inference prototype for
FastText-style lexical embeddings. It was built before the current wave of LLM
serving systems, but the engineering problem is familiar: keep model behavior
useful while making the serving artifact smaller, faster to load, and easier to
share across processes.

The project is not a general neural inference runtime. Its narrower claim is
that a large subword embedding model can separate three concerns that are often
collapsed together:

1. **Identity.** Use exact trie-backed ids for words and character n-grams, so
   subword rows remain inspectable before compression.
2. **Compression.** Merge only selected n-gram rows after training, when
   lexical structure and vector similarity support sharing.
3. **Serving layout.** Rewrite live rows into a dense compact matrix and update
   trie terminal values so the exported `.z` artifact can be mmap-loaded.

That makes the repository relevant to inference-optimization work that cares
about cold start, resident memory, model layout, and low-latency retrieval
features, especially when a small lexical layer is enough.

## Why this matters now

Modern deployment stacks often focus on large dense encoders or LLMs. Many
production systems still keep a cheaper lexical stage for candidate generation,
long-tail entities, OOV words, product names, typo handling, or multilingual
fallbacks. In those settings, the first-stage model is valuable only if it is
cheap to keep resident and quick to load.

`me_fasttext` explores that design space for FastText-style embeddings:

| Inference concern | Design choice in this project |
| --- | --- |
| Avoid opaque subword collisions | Replace hash buckets with exact trie ids. |
| Keep OOV behavior | Preserve character n-gram composition at lookup time. |
| Reduce serving footprint | Compact live vector rows after training. |
| Lower startup cost | Export a single mmap-oriented `.z` artifact. |
| Support debugging | Keep lexical ids auditable before compaction. |

## Where it is a strong prior

This is a useful citation or comparison point for work on:

- compact lexical embedding serving;
- memory-efficient FastText or subword models;
- inference-time model layout and mmap artifacts;
- first-stage retrieval before dense reranking;
- explicit lexical identity before compression;
- applying classical memory-management ideas to AI model artifacts.

It is a weaker fit for papers whose main contribution is LLM prompting, dense
sentence embedding quality, generic vector databases, or unrelated quantization
methods.

## Suggested citation framing

Use the paper when the method is part of the technical comparison:

```text
We compare with me_fasttext, an early compact-inference prototype for
FastText-style lexical embeddings. It keeps word and character n-gram identities
explicit with trie-backed ids, then exports a compact mmap serving artifact via
structure-aware row sharing and mark-compact row rewriting.
```

Use the repository when you only need implementation context:

```text
For a compact lexical first-stage model, we use the me_fasttext implementation,
which provides exact trie-backed subword ids and an mmap-ready serving index.
```

## What to report

For inference-oriented evaluations, the most useful report includes:

- model artifact size for `.bin` and `.z`;
- cold load time and warm query latency;
- peak RSS or resident memory after loading;
- p50 and p95 lookup latency;
- OOV or rare-token coverage;
- downstream retrieval, ranking, or classification quality;
- corpus language, n-gram range, vector dimension, and commit SHA.

Public negative results are welcome. A case where the compact layout does not
help is still useful evidence about where this design belongs.
