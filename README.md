# NGramStore

**A general-purpose backend for learned n-gram embeddings: compression, storage and exact lookup.**

**面向 learned n-gram embedding 的通用后端引擎：压缩、存储与精确查询。**

[![CI](https://github.com/initial-d/ngram-store/actions/workflows/ci.yml/badge.svg)](https://github.com/initial-d/ngram-store/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/arXiv-2506.01254-b31b1b.svg)](https://arxiv.org/abs/2506.01254)
[![Release](https://img.shields.io/github/v/release/initial-d/ngram-store?label=release)](https://github.com/initial-d/ngram-store/releases/latest)

**从早期 n-gram embedding，到大模型中的 learned n-gram 参数化记忆。**

FastText uses learned subword vectors to build word representations;
Engram-style LLM memory retrieves learned token-n-gram features and fuses them
with hidden states. The application changes, while the need to index, store and
query trained vectors continues. NGramStore follows this thread through a
shared backend. Learned n-gram tables are one form of parametric memory.

Formerly **me_fasttext**. The engine grew from
[Memory-Efficient FastText: A Comprehensive Approach Using Double-Array Trie Structures and Mark-Compact Memory Management](https://arxiv.org/abs/2506.01254).
The paper keeps its original title; later Engram integrations are software extensions.

NGramStore separates the learned n-gram table from the model that uses it.
It stores frozen vectors in compressed native blocks and queries original row
IDs through mmap, decoded caches and batched gathers. The common boundary is
the embedding table: models retain their own training, addressing and vector
aggregation. Registered string keys add DATrie indexing and structural
mark-compact planning; numeric/hash IDs can use the engine directly.

| Layer | Responsibility |
| --- | --- |
| 1. Shared table engine | Lossless block encoding, original-ID forwarding, mmap, bounded decoded caches and gather |
| 2. Optional n-gram structure | Registered DATrie keys, prefix/suffix representative selection and mark-compact export |
| 3. Model applications | FastText word embeddings and Engram/TinyEngram LLM memory, each retaining its model semantics |

FastText is the historical word-embedding application; Engram is the LLM-memory
application. Both exercise the same engine. Generality is at the frozen-table
boundary; it does not require a new model architecture or retraining.

**Lossless and approximate modes are explicit.** Ordinary packing and
structural `layout`/`exact` preserve source bytes. Opt-in `joint` sharing uses
cosine and norm-ratio guards and can change vectors. Engram hash collisions do
not certify string-prefix/suffix relations. Storage, latency and model quality
are measured separately; gains depend on the trained table and query workload.

The motivation is the tradeoff between n-gram identity, capacity and fidelity:
hash buckets can make distinct n-grams share a learned row; full distinct-key
tables increase storage and working-set pressure; quantization or approximate
sharing changes vectors. NGramStore offers byte-preserving storage and optional
exact key indexing. It cannot undo already-trained hash collisions, and exact
compression can save little or query more slowly. [Tradeoffs and roadmap](docs/ngram_memory_research_plan.md).

For a stable vocabulary, fix unique key-to-ID mappings before training, train
mutable embeddings, then freeze and export the same IDs and rows. Cover the
chosen vocabulary within resource limits; the native backend does not currently
perform gradient updates. Existing frozen tables can be packed directly.

## Quick start

Linux/macOS require a C compiler and zlib development headers. From this checkout:

```bash
python -m pip install .
ngram-store --help
```

The distribution remains `meft-memory` for install compatibility; the engine API
is `ngram_store` and the CLI is `ngram-store`. No Torch is needed for packing or
numeric lookup. Existing `meft_memory` imports and `meft-*` commands still work.

Use an **existing trained table** exported as row-contiguous little-endian float32:

```bash
ngram-store pack --input vectors.f32 --dim 256 --output memory.native
ngram-store inspect --store memory.native
ngram-store query --store memory.native --ids 0 42 0
```

Packing verifies every reconstructed row before publishing a new directory.
IDs retain order and duplicates. `query --hex` returns the exact row bits.
For registered keys, add `--keys keys.json --structural-library /path/to/libmeft_structural.so`
to `pack`; see the [engine guide](docs/evaluator_quickstart.md).

```python
from ngram_store import NativeRows

with NativeRows("memory.native/table.men", cache_bytes=4 * 1024**2) as memory:
    original_bytes = memory.rows([0, 42, 0])
```

## Application examples

| Application | Runnable entry | What stays with the model |
| --- | --- | --- |
| FastText / word embeddings | [Prepare text → train → export → query](docs/fasttext_native_bundle.md) | Dictionary, original float32 aggregation, output weights |
| Engram / LLM memory | [Official demo factory and verified boundaries](docs/engram_quickstart.md) | Hash IDs, head layout, projections, gates and convolution |
| TinyEngram / Qwen | [Checkpoint export, trained outputs and decode](docs/tinyengram_qwen_evidence_20261030/README.md) | Backbone, tokenizer, learned gates and checkpoint state |

`ngram-store fasttext-export`, `tiny-export` and `tiny-verify` route to the
existing converters. FastText's custom trainer first needs
`ngram-store prepare-fasttext`; ordinary text is not its native input format.
The [included sample workflow](docs/fasttext_native_bundle.md) requires no
external data or GPU. It checks integration, not representative compression.

## Trained evidence and limits

| Evidence | Observed result and boundary |
| --- | --- |
| [Actual FastText model](docs/fasttext_native_evidence_20261031/README.md) | 90,380 rows and word/sentence outputs byte-identical; whole bundle saves 0.46%, hot queries slower than dense; structural order has no added benefit here |
| [Structural contribution controls](docs/structural_native_evidence_20261030/README.md) | Trained SGNS/GRU references measure DATrie and approximate joint sharing separately; exact mode finds no identical rows to merge |
| [FP16/INT8 + structural controls](docs/quantized_structural_evidence_20261031/README.md) | Same trained tables retain additional approximate savings after quantization; 4-order GRU INT8 regresses; quality failures and fixed-default audit are separate |
| [1.85M trained-row capacity](docs/cpu_capacity_evidence_20261030/README.md) / [sparse blocks](docs/sparse_blocks_evidence_20261030/README.md) | Smaller blocks lower native miss latency 16–27× relative to the previous native layout; raw mmap remains faster |
| [Large FP16/INT8 capacity and CPU controls](docs/quantized_capacity_evidence_20261031/README.md) | 54 fresh query workers; FP16 native saves 52.29% backend bytes versus raw FP32; INT8 native expands versus raw INT8, prompting an optional raw fallback; quantization is lossy |
| [TinyEngram/Qwen trained integration](docs/tinyengram_qwen_evidence_20261030/README.md) / [CUDA controls](docs/qwen_cuda_performance_20261030/README.md) | Preserved trained outputs and measured CPU/device costs; native lookup remains slower than dense offload on these traces |

Official DeepSeek Engram verification uses its small demonstration, not a
released trained Engram checkpoint. Small-model results are not production
serving or universal compression claims. [Evidence index](docs/index.md).

The standalone C engine/wheel supports Linux and macOS. The bundled legacy
DATrie bridge and FastText facade require Linux x86-64. Handles are frozen and
single-worker; cache budgets do not bound RSS. Structural planning holds full
rows/maps offline. Trusted legacy metadata/index formats retain their own
limits. [C ABI and runtime boundaries](docs/native_ngram_memory.md).

## History, citation and contribution

The [engineering roadmap](docs/ngram_memory_research_plan.md) targets a backend
that n-gram embedding users can routinely evaluate and reuse. Next priorities
are sparse-query cost, useful structural compression and bounded construction
resources, with stable integration and same-source baselines as acceptance gates.

Repository history, releases, issues and original paper attribution continue
under NGramStore. Old model formats, C ABI symbols, `meft_memory` imports and
FastText commands remain compatible. C++ clients must recompile; `.z` retains
its own reader and approximate contract. [Compatibility guide](docs/legacy_fasttext.md).

Cite the [original paper](https://arxiv.org/abs/2506.01254) for its method and the
exact software commit/report for later engine work. [Citation guide](docs/citation_guide.md).

Share source-linked measurements or integration issues through
[Contributing](CONTRIBUTING.md) and the
[benchmark template](https://github.com/initial-d/ngram-store/issues/new?template=benchmark-result.yml).
Negative results and omitted-baseline notes are useful evidence.

## License

Project code is MIT; bundled upstream FastText sources retain their original
notices. See [LICENSE](LICENSE).
