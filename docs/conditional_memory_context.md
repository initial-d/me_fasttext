# Indexed lexical memory and conditional memory

This note positions `me_fasttext` next to a newer design pattern that is now
appearing in large-model systems: explicit, indexed, and sparsely accessed
memory outside the most expensive dense path.

The comparison is intentionally conservative. This is not a claim that
DeepSeek's Engram uses `me_fasttext`, or that `me_fasttext` is an LLM memory
module. The useful connection is narrower: both designs respond to the same
pressure to separate reusable lexical or token memory from dense computation.

## The shared design pressure

Dense neural computation is expensive when every bit of linguistic memory has
to flow through the same path. Long-tail words, entity names, repeated local
patterns, and n-gram-like signals often need a different treatment:

1. keep an explicit key or identity;
2. look up only the relevant memory;
3. keep the serving representation compact;
4. report memory, latency, coverage, and quality together.

`me_fasttext` explores this pressure for FastText-style lexical embeddings.
DeepSeek's Engram explores a related pressure for large language models.

## How `me_fasttext` fits

`me_fasttext` is a small, inspectable lexical memory system:

- word and character n-gram identities are assigned with trie-backed exact ids;
- rows remain inspectable before compression;
- related rows can be shared after training;
- live rows are rewritten into a compact mmap-ready `.z` artifact.

The important point is the separation of concerns. Identity, compression, and
serving layout are separate steps instead of one fixed hash-bucket table.

## How Engram is adjacent

DeepSeek describes Engram as conditional memory through scalable lookup, a
complementary sparsity axis for large language models. The official
DeepSeek-V4.1-Flash model card lists Engram conditional memory alongside the
model's MoE and long-context architecture, and the official Engram repository
frames the method as modernizing classic n-gram embedding for lookup-oriented
memory.

That makes Engram an adjacent signal for the same broad direction:

| Question | `me_fasttext` answer | Engram-style answer |
| --- | --- | --- |
| What is indexed? | Words and character n-grams | Token or n-gram-like memory entries |
| Where does it live? | A compact lexical embedding artifact | A conditional memory component in an LLM stack |
| How is it accessed? | Trie lookup and mmap serving | Sparse lookup conditioned on tokens |
| What is optimized? | Artifact size, cold load, OOV coverage, lexical serving | Large-model capacity, memory allocation, and inference cost |
| What should be reported? | Memory, load time, p50/p95 latency, coverage, quality | Memory allocation, latency/cost, downstream model quality |

## Safe citation framing

Use cautious wording:

```text
me_fasttext is not an LLM memory module, but it is a compact lexical example of
the same broader systems idea behind conditional-memory work: keep reusable
word or n-gram memory indexed, inspectable, and cheaper to serve than dense
neural computation.
```

Avoid stronger claims:

- do not say Engram is based on `me_fasttext`;
- do not say `me_fasttext` predicts or explains DeepSeek-V4.1-Flash;
- do not compare quality without a shared benchmark;
- do not frame trie lookup as a universal replacement for learned memory.

## Sources for the adjacent context

- DeepSeek-V4.1-Flash model card:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>
- DeepSeek Engram repository:
  <https://github.com/deepseek-ai/Engram>
- DeepSeek-V4.1-Flash announcement:
  <https://www.deepseek.com/en/news/deepseek-v4-1-flash/>

For `me_fasttext`, continue with the
[`artifact_card.md`](artifact_card.md),
[`inference_optimization_note.md`](inference_optimization_note.md), and
[`retrieval_oov_benchmark.md`](retrieval_oov_benchmark.md).
