# Memory management for AI embeddings

`me_fasttext` is easiest to describe as a memory-efficient FastText variant.
That description is accurate, but it hides the main design idea.

The deeper idea is this: a large FastText-style subword model is not only an
NLP model. It is also a memory-layout problem. Once a corpus produces hundreds
of millions of character n-grams, the system has to decide which lexical
objects deserve stable identities, which vector rows can be shared, and how the
remaining rows should be rewritten for serving.

That is the same shape as an old systems problem: identify live objects,
compute forwarding addresses, compact storage, and rewrite references.

## The missed framing

The original FastText design made a good engineering choice: hash character
n-grams into a fixed bucket table so memory stays bounded. This made the model
simple, fast, and practical.

But the hash bucket is also a lossy memory manager. It decides that many
distinct n-grams must share storage before the model has learned anything about
them. The collision is cheap, but it is blind. Two unrelated subwords can occupy
the same row because the hash function says so, not because the language data or
the learned vectors justify it.

`me_fasttext` asks a different question:

> What if subword identity and serving layout are separated?

The answer is an exact-first, compact-later pipeline:

1. Give every retained word and observed character n-gram an exact identity.
2. Store those identities in trie-backed dictionaries.
3. Train FastText-style embeddings without hash-bucket collisions.
4. Compress the trained n-gram matrix only after structure and vector similarity
   are available.
5. Rewrite the surviving rows and trie terminal values into a compact,
   memory-mappable serving artifact.

This reframes the project from "a FastText optimization" to "classical memory
management applied to AI model storage."

## Why tries matter

The trie is not just a lookup acceleration trick. It is the identity layer.

FastText's hash buckets make subwords anonymous once they enter the table.
Trie-backed ids keep lexical forms inspectable:

- the word has a stable id
- the n-gram has a stable id
- prefix and suffix structure remains visible
- compression can be reasoned about after training

This matters most for Chinese, multilingual, and entity-heavy corpora, where
long-tail forms are common and OOV behavior is not a corner case. Character
n-grams carry useful signal, but the number of possible n-grams grows quickly.
The trie gives the system a precise map of that space before any compaction
happens.

## The language-statistics opportunity

The reason this design is useful is not only that n-grams are numerous. It is
that n-grams are numerous in a structured way.

FastText-style character n-grams expose repeated lexical fragments:

- shared prefixes
- shared suffixes
- morphology-like fragments
- transliteration fragments
- entity-name fragments
- rare forms that still share local character evidence

After training, many related n-gram rows become similar enough that keeping
separate serving rows is wasteful. But that decision should be made with both
structure and learned vectors in view.

`me_fasttext` therefore treats compression as a late binding decision. It does
not merge rows merely because two strings landed in the same hash bucket. It
considers structurally related n-grams and then checks whether their learned
vectors are close enough to share a compact row.

## Mark-compact as an embedding export algorithm

The export path follows the shape of a mark-compact collector:

| Classical memory-management idea | Embedding-system analogue |
| --- | --- |
| Object | Word or n-gram vector row |
| Root/reference | Trie terminal value pointing to a row |
| Live object | Row retained after conservative sharing |
| Mark phase | Decide which rows still need distinct storage |
| Forwarding address | Map original row ids to compact row ids |
| Compaction | Rewrite rows into dense serving order |
| Reference update | Rewrite trie terminal values to compact ids |

This is not a garbage collector implementation. It is a reuse of the same
conceptual machinery for a static model artifact.

That reuse is the core taste of the project: old systems ideas become useful
again when AI models are viewed as large, structured memory objects rather than
only as parameter tensors.

## Why mmap is part of the idea

Compression alone is not the full serving story. A smaller matrix helps, but a
production service also cares about startup behavior, process sharing, and
reload cost.

The compact `.z` artifact is designed around mmap serving:

- load model metadata without rebuilding training structures
- map contiguous vector rows directly
- keep trie lookup tables in the serving artifact
- allow operating-system page-cache behavior to work for repeated loads
- reduce cold-start cost for memory-sensitive deployments

This makes the project closer to a model-storage system than a pure training
algorithm.

## Why the idea still matters

Modern NLP is dominated by transformer encoders and LLM embeddings. That does
not make small lexical memory layers obsolete.

There are still many places where a cheap, deterministic, OOV-aware layer is
valuable:

- first-stage retrieval before dense reranking
- entity-heavy search
- long-tail Chinese or multilingual features
- low-latency classification
- embedded or memory-constrained inference
- feature generation where transformer calls are too expensive

The point is not to compete with large semantic encoders. The point is to keep
a compact lexical memory layer available when the workload needs exact local
character evidence.

## How to evaluate the idea

The right comparison is not only "does it get the best benchmark score?"

A fair evaluation should ask:

- How much memory is used by the full training representation?
- How much memory is used by the compact serving artifact?
- How long does cold loading take?
- What is warm query latency?
- How does quality change on OOV-heavy or entity-heavy slices?
- How much quality is lost relative to exact, hash-free rows?
- How much quality is lost relative to the original FastText hash-bucket model?
- Can another corpus reproduce the same memory/quality tradeoff?

This is why the repository includes a benchmark protocol. The strongest future
work is not another slogan. It is independent reports on different corpora,
languages, and retrieval/classification tasks.

## A better one-sentence summary

If you cite or describe this work, the most useful framing is:

> `me_fasttext` applies classical mark-compact memory-management ideas to
> FastText-style subword embeddings: exact trie-backed n-gram identities are
> trained first, then rewritten into a compact mmap serving layout.

That sentence is the heart of the project.
