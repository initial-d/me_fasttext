# NGramStore artifact card

**A general-purpose backend for learned n-gram embeddings: compression, storage and exact lookup.**
Formerly me_fasttext; the original Memory-Efficient FastText paper is the method
origin, while native row storage and LLM integrations are later software work.

The historical thread runs from n-gram embeddings for word representations to
learned n-gram memory modules in LLMs. FastText and Engram illustrate these
application generations: both query trained vector tables, with different keys,
addressing and model computation. This table-based form of parametric memory is
the scope of the shared engine.

| Layer / component | What it provides | Boundary |
| --- | --- | --- |
| 1. Shared native C engine | Lossless compressed blocks, original-ID forwarding, mmap and gather | Frozen single-worker handles; caches do not bound RSS |
| 2. Optional DATrie / mark-compact | Exact registered keys and structural representatives | Linux x86-64 trusted metadata; joint sharing can change weights |
| 3. FastText application | Dense model export and existing vector/classification queries | This repository's custom unquantized models, not official .bin conversion |
| 3. Engram / TinyEngram application | Storage replacement behind existing hashes and memory gathers | Demo and trained TinyEngram/Qwen evidence are distinct from official trained Engram |

The shared contract is a frozen table with original IDs, shape, dtype and row
bytes. Models supply IDs and retain training, hashing, aggregation and downstream
computation. Registered unique strings enable the optional structural layer;
the core also accepts generic numeric rows without making n-gram-structure claims.

Hash buckets trade independent identities for bounded capacity; full distinct-key
tables incur storage/working-set costs; quantization or approximate sharing
changes vectors. Exact storage introduces no new vector error, but it cannot
recover identities already combined during bucket-based training. Exact key
indexing requires distinct source rows and registered unique keys.

## Start and evaluate

Use the [engine quickstart](evaluator_quickstart.md), then select an application.
[Trained evidence](index.md) separates identity, storage, query costs and task
quality. Small real-model native artifacts can save little and be slower than
dense/raw mmap; structural layout is not universally better. Numeric hash buckets
can use the row engine but do not certify structural string sharing.

A comparison should preserve rows and query traces, state exact vs approximate
mode and include forwarding/index/metadata bytes. Report complete latency and
RSS alongside compression. Generic vectors are supported by the row engine;
stronger structural claims require a real unique n-gram registry.

The [engineering roadmap](ngram_memory_research_plan.md) makes independent
integration, measured serving/storage tradeoffs and construction resources the
adoption criteria. Industry-wide adoption remains a goal, not a measured result.

## Cite and reuse

Use [the paper](https://arxiv.org/abs/2506.01254) for its method and
[the exact code/report](citation_guide.md) for later engine results.
This is a storage/query backend, with FastText and Engram application examples;
it is not a universal task-quality or production-performance guarantee.
