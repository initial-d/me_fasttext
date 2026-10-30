# NGramStore artifact card

**Compression, storage and exact lookup for learned n-gram vectors.**
Formerly me_fasttext; the original Memory-Efficient FastText paper is the method
origin, while native row storage and LLM integrations are later software work.

| Component | What it provides | Boundary |
| --- | --- | --- |
| Native C engine | Lossless compressed blocks, original-ID forwarding, mmap and gather | Frozen single-worker handles; caches do not bound RSS |
| DATrie / mark-compact | Exact registered keys and structural representatives | Linux x86-64 trusted metadata; joint sharing can change weights |
| FastText application | Dense model export and existing vector/classification queries | This repository's custom unquantized models, not official .bin conversion |
| Engram / TinyEngram application | Storage replacement behind existing hashes and memory gathers | Demo and trained TinyEngram/Qwen evidence are distinct from official trained Engram |

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

## Cite and reuse

Use [the paper](https://arxiv.org/abs/2506.01254) for its method and
[the exact code/report](citation_guide.md) for later engine results.
This is a storage/query backend, with FastText and Engram application examples;
it is not a universal task-quality or production-performance guarantee.
