# Related work map

`me_fasttext` sits between several communities that often talk past each other:
subword embeddings, compact lexical data structures, model-serving systems, and
memory-management ideas. This page gives readers a conservative map of where
the project belongs and where it should not be overclaimed.

## Core framing

The project is best read as an exact-then-compact FastText variant:

1. keep word and character n-gram identities explicit with trie-backed ids;
2. train with inspectable rows instead of opaque hash buckets;
3. merge only selected rows after training when lexical structure and vector
   similarity support sharing;
4. rewrite live rows into a dense mmap-ready serving artifact.

The main contribution is not a new neural architecture. It is a storage and
serving layout for FastText-style lexical embeddings where identity,
compression, and deployment are handled as separate steps.

## Where to place it

| Area | Natural connection | What `me_fasttext` adds |
| --- | --- | --- |
| FastText and subword embeddings | Character n-gram vectors for rare and unseen words | Exact n-gram ids before compression, instead of fixed hash buckets |
| Trie and double-array trie systems | Compact lexical lookup for large dictionaries | Word and n-gram lookup become part of the embedding table design |
| Memory-efficient model serving | Smaller artifacts, lower cold-load cost, mmap-friendly layout | A compact `.z` artifact produced by row rewriting and trie terminal remapping |
| Retrieval and RAG pipelines | Cheap lexical recall before dense reranking | OOV-aware lexical vectors that can be served without loading a large training model |
| Memory management | Mark, forwarding, and compaction as a layout transformation | A classical mark-compact idea applied to trained embedding rows |

## What it is not

The project should not be described as:

- a replacement for dense transformer embeddings;
- a complete RAG framework;
- a generic trie library;
- a general garbage collector;
- a universal compression method for all embedding models;
- a claim that trie lookup is always faster than hashing.

Those are adjacent topics. The narrower claim is stronger: explicit subword
identity can be useful during training and analysis, then compacted into a
serving layout when deployment cost matters.

## Related-work hooks

### FastText-style embeddings

The most direct comparison is the standard hash-bucketed FastText design. A
paper or benchmark can use `me_fasttext` when it needs to discuss the tradeoff
between fixed-size hashed n-gram tables and exact n-gram identities.

Useful questions:

- How many n-grams collide under a fixed bucket budget?
- Does removing collisions improve a downstream metric?
- How much memory is added by exact ids before compression?
- How much of that memory can be recovered during export?

### Lexical retrieval and reranking

For search systems, `me_fasttext` is relevant as a small lexical embedding
layer. It is most plausible when rare terms, product names, domain entities,
misspellings, or Chinese/multilingual long-tail tokens matter.

Useful questions:

- Can it improve first-stage recall for OOV-heavy queries?
- Can it reduce candidate sets before dense reranking?
- Does the compact index lower startup cost for batch retrieval jobs?
- Does exact n-gram inspection make failures easier to debug?

### Data structures

The data-structure angle is not just "uses a trie". The trie participates in
the model's id assignment and serving layout. This is where the implementation
is relevant to double-array trie readers: lexical keys remain recoverable until
the explicit compression step.

Useful questions:

- What is the storage cost of exact lexical ids?
- How does trie serialization affect cold load and lookup?
- Which debugging workflows become possible when n-gram ids are not opaque?

### Systems and mmap serving

The compact `.z` file is designed for serving. The interesting systems question
is whether the exported artifact improves cold-start, memory footprint, or
multi-process sharing compared with loading the full training model.

Useful questions:

- What is the serialized artifact size?
- How long does cold mmap load take?
- What RSS is observed before and after warmup?
- How stable are p50 and p95 lookup latencies?

### Classical memory-management ideas

The mark-compact analogy is precise but limited. Original rows are treated like
objects, live rows are marked, old ids receive forwarding ids, and trie terminal
values are rewritten to point into a dense compact matrix. This is not runtime
garbage collection; it is an offline model-layout transformation.

Useful questions:

- Which rows remain live after conservative row sharing?
- How are old row ids mapped to compact ids?
- Can the layout be audited after export?
- Does compacting rows improve serving behavior without hiding identity too
  early in the pipeline?

## Citation positioning

Use the paper as prior work when your project depends on one of these claims:

- exact subword identity is useful before compression;
- a FastText-style model can be served from a compact mmap artifact;
- row compaction can be framed as a model-layout problem;
- trie-backed n-gram ids make lexical embedding storage more inspectable;
- OOV-heavy retrieval or classification benefits from a small lexical layer.

Mention the repository as implementation context when you only need an example
of a compact FastText-derived system and are not evaluating the method itself.

## Suggested one-sentence description

`me_fasttext` is a FastText-derived lexical embedding system that replaces
hash-bucketed character n-gram ids with trie-backed exact ids, then exports a
compact mmap-ready serving artifact through conservative row sharing and
mark-compact style row rewriting.
