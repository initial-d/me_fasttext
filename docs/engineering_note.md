# Engineering note: exact subword ids, compact rows, and mmap serving

`me_fasttext` is an NLP system, but its core tradeoff is a systems problem:
how to keep FastText-style OOV behavior without paying for a very large,
collision-free n-gram matrix at serving time.

## The baseline problem

FastText avoids storing every observed character n-gram by hashing n-grams into
a fixed bucket table. This is compact and fast, but it makes subword identity
opaque: unrelated n-grams can share the same row because of hash collisions.

A hash-free variant can assign every n-gram an exact id, but the embedding
matrix grows with the observed n-gram vocabulary. That can be expensive for
large Chinese, multilingual, or entity-heavy corpora.

## Exact first, compact later

`me_fasttext` separates identity from storage:

1. During dictionary construction, words and UTF-8 character n-grams receive
   exact trie-backed ids.
2. During training, each retained word or n-gram has a distinct row.
3. During export, structurally related n-gram rows are considered for sharing
   only when their learned vectors are very close.
4. The remaining live rows are assigned dense ids and serialized into a compact
   serving artifact.

This preserves inspectability before compression and makes row sharing an
explicit export-time decision rather than an arbitrary hash collision.

## Why double-array tries

A trie gives the system a direct mapping from lexical forms to row ids. A
double-array trie keeps this mapping compact enough for large vocabularies and
can be serialized with predictable lookup behavior.

In this repository, tries are used for:

- word to word-row id lookup
- character n-gram to n-gram-row id lookup
- compact serving index lookup after row ids have been rewritten

This is why the project is relevant to trie and DA-trie communities even though
it is not a general-purpose trie package.

## Mark-compact as model layout

The export path borrows the shape of a traditional mark-compact collector:

| GC term | `me_fasttext` export analogue |
| --- | --- |
| object | word or n-gram vector row |
| live set | rows retained after conservative sharing |
| mark | record which original rows are still needed |
| forwarding address | map old row ids to dense compact row ids |
| compact | rewrite rows and trie terminal values into a dense layout |

The goal is not automatic memory reclamation inside a runtime. The goal is to
produce a smaller, cache-friendlier, mmap-ready model artifact for inference.

## Why mmap matters

The compact `.z` file is designed so serving code can map the model into memory
instead of rebuilding training-time structures. This matters for:

- service restart time
- autoscaling cold starts
- memory sharing between worker processes
- batch jobs that load many models or reload frequently

Cold load time and RSS still depend on OS page-cache behavior, filesystem,
hardware, and access pattern, so benchmark reports should include a hardware
manifest.

## Where this is useful

`me_fasttext` is most useful as a cheap lexical layer:

- first-stage retrieval before dense reranking
- OOV fallback for entity-heavy search
- long-tail Chinese or multilingual text features
- low-latency classification or ranking features
- deployments where transformer embeddings are too expensive for every query

It is not a complete RAG framework, a general garbage collector, or a
standalone functional-programming example.

## Related communities

- FastText and static embedding users can evaluate OOV and quality tradeoffs.
- Retrieval and RAG users can test lexical prefiltering and fallback recall.
- Trie/data-structure readers can inspect the DA-trie indexing role.
- Systems and memory-management readers can evaluate the compact export layout.
- C++ readers can study the mmap serving boundary and artifact format.
