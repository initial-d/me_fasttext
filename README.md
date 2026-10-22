# me_fasttext

[![arXiv](https://img.shields.io/badge/arXiv-2506.01254-b31b1b.svg)](https://arxiv.org/abs/2506.01254)
[![CI](https://github.com/initial-d/me_fasttext/actions/workflows/ci.yml/badge.svg)](https://github.com/initial-d/me_fasttext/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/initial-d/me_fasttext?label=release)](https://github.com/initial-d/me_fasttext/releases/tag/v0.1.0)
[![Benchmark reports welcome](https://img.shields.io/badge/benchmark_reports-welcome-2ea44f)](https://github.com/initial-d/me_fasttext/issues/1)
[![Evaluator quickstart](https://img.shields.io/badge/evaluator-quickstart-blue)](docs/evaluator_quickstart.md)
[![Showcase](https://img.shields.io/badge/use_cases-showcase-orange)](https://github.com/initial-d/me_fasttext/issues/new?template=showcase.yml)

**Compact n-gram memory: exact indexing, storage compression, and mmap lookup.**
`me_fasttext` explores these systems problems through a FastText-derived C++
prototype, a native C row-gather backend, and PyTorch adapters for Engram-style
lookup experiments.

This repository accompanies the paper [Memory-Efficient FastText: A Comprehensive Approach Using Double-Array Trie Structures and Mark-Compact Memory Management](https://arxiv.org/abs/2506.01254).

> Status: research / production-prototype code. The command-line interface remains close to the original fastText workflow, while the storage and serving path has been extended for large-vocabulary, memory-sensitive deployments.

> Inference framing: `me_fasttext` can also be read as an early compact-inference prototype for lexical embeddings. It keeps subword identity exact during training, then rewrites the learned rows into a smaller mmap-friendly serving artifact for low-memory retrieval and feature-serving workloads.

## Engram readers: start here

**中文入门：[Engram 之后，高阶 n-gram 记忆还能压缩多少？](docs/zhihu_engram_article_20261022.md)**
— trained evidence, two figures, and a runnable integration example.

[DeepSeek's Engram](https://github.com/deepseek-ai/Engram) highlights a concrete
systems question: **how do we store and serve large n-gram memory tables?**
This project makes that question inspectable through exact trie indexing,
compact row layouts, and mmap lookup. **[Run, evaluate and cite the Engram
storage artifact](docs/engram_quickstart.md)** is the shortest path from the
interface proposal to executable code and source-linked evidence.

- [Engram storage guide](docs/engram_storage.md): architecture mapping,
  byte-preservation contract, and the adapter boundary for token/hash row IDs.
- [Run the lossless row-store demo](docs/engram_storage.md#run-the-byte-exact-demo):
  exact-byte deduplication, compressed blocks, original-row-ID forwarding, and
  bounded-cache mmap access. It runs with Python's standard library.
- [PyTorch gather adapter](docs/engram_gather_adapter.md): immutable CPU
  replacement for the official demo's multi-head embedding gather, preserving
  per-head offsets, dtype and tensor shape. Random weights verify the interface.
- [C ABI and model integration](docs/native_ngram_memory.md): standalone native
  gather, direct PyTorch output, ordinary embedding support, and integration
  boundaries for Engram, HF decoders and C++/GPU runtimes.
- [Full official Engram CPU module verification](docs/engram_full_adapter_verification_20261018.md):
  Python/native storage produces identical gates/convolution outputs across
  three dtypes on tiny random weights; trained/GPU results remain open.
- [Trained n-gram evidence](docs/trained_ngram_similarity_20261018.md): real-text
  SGNS across orders 4/8/12 and three seeds. Identical-update structural pairs
  have median cosine about 0.996; offline byte-exact residual prediction tests
  how to exploit that redundancy. This is a FastText-style reference, not Engram
  training or a model-quality result.
- [Native lossless XOR codec](docs/native_xor_compression_20261020.md): actual
  C-readable artifacts on nine trained tables. Adaptive block encoding and
  original-ID forwarding save 6.9–9.1% with training-signature grouping;
  storage gains and measured CPU query costs are reported separately.
- [Streaming export follow-up](docs/streaming_export_20261021.md): about 1.9×
  faster XOR packing on the same trained rows, identical files, and block-sized
  embedding staging with a separate process-memory measurement.
- [Bounded native query cache](docs/native_cache_20261022.md): optional decoded
  block reuse with a byte budget, original-byte output and measured hotspot,
  corpus-replay and no-benefit cases. Disabled by default.
- [High-order memory research plan](docs/ngram_memory_research_plan.md): trained
  structural redundancy, lossless codecs, indexing and serving experiments.
- [Approximate compression curves](docs/approximate_ngram_compression_20261018.md):
  cosine thresholds 0.999..0.90, norm-ratio controls, actual artifact sizes and
  vector error. At max order 12, joint cosine/norm ≥0.99 gives 2.14–2.16×
  n-gram storage reduction; these are vector-only results, not model task loss.
- [English and Chinese launch copy](docs/engram_promotion.md): a technical
  introduction centered on n-gram memory storage and lookup.

The new reference preserves each original row's bytes and ID lookup. The legacy
C++ `.z` exporter performs approximate row sharing and has known implementation
limits. The new gather adapter is independent of that exporter. No trained
Engram checkpoint or end-to-end model has been evaluated here. See the
[implementation audit](docs/engram_storage.md#legacy-export-audit).

## Help the project grow

If `me_fasttext` is useful or relevant to your work, the fastest ways to help
other people find it are:

- star or watch the repository so new benchmark notes surface to more readers;
- share a public-corpus benchmark, replication, or negative result with the
  [benchmark template](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml);
- open a
  [showcase / use-case note](https://github.com/initial-d/me_fasttext/issues/new?template=showcase.yml)
  if you cite it, compare it, add it to a list, or try it in retrieval,
  ranking, OOV-heavy text, Chinese text, or compact-serving work;
- use the [share kit](docs/share_kit.md) for copy-ready descriptions, posts,
  list-maintainer blurbs, and responsible sharing boundaries;
- send an
  [evaluation question](https://github.com/initial-d/me_fasttext/issues/new?template=evaluation-question.yml)
  if you are unsure how to build, benchmark, or interpret a result.

Short, honest reports are better than polished success stories. Reproducible
negative results, omitted-baseline notes, and corpus-specific limits all make
the project easier to trust and cite.

## If you are evaluating this project

The most valuable contribution right now is a reproducible public-corpus
benchmark: memory, load time, compact `.z` mmap-serving latency, OOV coverage,
and one retrieval, ranking, or classification quality metric. Negative results
are welcome if the corpus and hardware context are clear.

Fast paths:

- [Evaluator quickstart](docs/evaluator_quickstart.md): a short path from clone
  to a useful benchmark report.
- [Contributing guide](CONTRIBUTING.md): what benchmark reports, negative
  results, documentation fixes, and code patches should include.
- [Evaluation question](https://github.com/initial-d/me_fasttext/issues/new?template=evaluation-question.yml):
  ask about build, benchmark setup, corpus preparation, baselines, or result interpretation.
- [Showcase / use case](https://github.com/initial-d/me_fasttext/issues/new?template=showcase.yml):
  share citations, list entries, comparisons, integrations, or deployment notes.
- [Benchmark issue](https://github.com/initial-d/me_fasttext/issues/1): share a
  reproducible run, replication, or negative result.
- [Artifact card](docs/artifact_card.md): decide quickly whether the project is
  relevant to cite, benchmark, or list.
- [Compression comparison checklist](docs/compression_comparison_checklist.md):
  report sparse, quantized, or compact-serving comparisons without overclaiming.
- [Answer engine context](docs/answer_engine_context.md): concise source-of-truth
  wording for search, citation, and AI answer summaries.
- [Citation guide](docs/citation_guide.md): cite the paper and describe the
  comparison without overclaiming.

## Start here

- [Evaluator quickstart](docs/evaluator_quickstart.md): the shortest route from
  clone to a useful report.
- [Contributing guide](CONTRIBUTING.md): how to share benchmark evidence,
  negative results, docs fixes, and code patches.
- [Code of conduct](CODE_OF_CONDUCT.md): expectations for evidence-oriented
  benchmark and reuse discussions.
- [Benchmark protocol](docs/benchmark_protocol.md): how to compare memory, load time, latency, and task quality.
- [First public corpus benchmark](docs/first_public_corpus_benchmark.md): a concrete external run plan for the first citable public-corpus report.
- [Retrieval and OOV serving benchmark](docs/retrieval_oov_benchmark.md): how to measure mmap load time, p50/p95 query latency, and OOV subword coverage.
- [Retrieval comparison matrix](docs/retrieval_comparison_matrix.md): how to compare the compact lexical layer with FastText, BM25, dense embeddings, and ANN/vector-search systems.
- [Artifact card](docs/artifact_card.md): a one-page summary for citation, benchmark, and list-maintainer triage.
- [Answer engine context](docs/answer_engine_context.md): canonical short
  descriptions, citation wording, and boundaries for AI-generated summaries.
- [Citation guide](docs/citation_guide.md): how to cite the paper, position related work, and report comparisons.
- [Citation evidence ladder](docs/citation_evidence_ladder.md): how benchmark
  reports become README-level or paper-level citation evidence.
- [Inference optimization note](docs/inference_optimization_note.md): how to position the project as compact lexical embedding inference.
- [Conditional memory context](docs/conditional_memory_context.md): a cautious
  note connecting indexed lexical memory with newer Engram-style conditional memory.
- [Model-compression and Engram-style bridge](docs/model_compression_engram_bridge.md):
  how to evaluate `me_fasttext` as a compact lexical-memory baseline without
  overclaiming it as an LLM memory module.
- [Compression comparison checklist](docs/compression_comparison_checklist.md):
  a focused report shape for sparse, quantized, and compact-serving baselines.
- [Cross-community positioning](docs/cross_community_positioning.md): precise framing for FastText, word-embedding, DA-trie, GC, and inference-optimization readers.
- [Related work map](docs/related_work_map.md): how the project connects to FastText, tries, mmap serving, retrieval, and memory management.
- [Engineering note](docs/engineering_note.md): how trie ids, mark-compact style row rewriting, and mmap serving fit together.
- [Design essay](docs/memory_management_for_ai_embeddings.md): why this project reframes FastText storage as an AI memory-management problem.
- [Chinese retrospective](docs/technical_retrospective_zh.md): a long-form Chinese note for technical storytelling and outreach.
- [Outreach notes](docs/outreach.md): concise wording for adding the project to NLP, embedding, retrieval, or systems lists.
- [Share kit](docs/share_kit.md): copy-ready summaries, posts, maintainer notes, and responsible sharing boundaries.
- [Promotion posts](docs/promotion_posts.md): short posts and maintainer notes for careful outreach.
- [Listing follow-up](docs/listing_followup.md): accepted, open, and closed external list submissions with next actions.
- [Distribution log](docs/distribution_log.md): external list submissions, accepted entries, and follow-up status.
- [Visibility status](docs/visibility_status.md): traffic snapshots, live entry points, and evidence-oriented next actions.
- [Reuse and citation pathways](docs/reuse_and_citation_pathways.md): where follow-up work can naturally cite the method.
- [Citation metadata](CITATION.cff): machine-readable citation information for GitHub and reference managers.
- [Release notes](docs/release_notes_v0.1.0.md): scope and caveats for the first research-prototype release.

## External listings

- [Awesome Data Science](https://github.com/academic/awesome-datascience) lists `me_fasttext` under General Machine Learning Packages.

## Where it fits

Use `me_fasttext` when you need a small lexical memory layer rather than a
large semantic encoder:

| Use case | Why this project is relevant |
| --- | --- |
| OOV-heavy retrieval | Character n-grams preserve signals for rare and unseen words. |
| Entity-heavy search | Exact trie ids make subword identities inspectable before compression. |
| RAG prefiltering | A low-cost lexical stage can reduce candidates before dense reranking. |
| Compact inference | A dense compact matrix plus mmap loading lowers the cost of keeping lexical embeddings online. |
| Memory-sensitive serving | The compact `.z` artifact is designed for mmap-based loading. |
| Long-tail Chinese or multilingual text | UTF-8 n-gram enumeration avoids byte-level shortcuts. |

## Why this exists

Standard fastText represents a word by averaging its word vector and character n-gram vectors. To avoid storing every observed n-gram, the reference implementation hashes subwords into a fixed bucket table. That keeps memory bounded, but unrelated n-grams can collide into the same row.

`me_fasttext` follows an exact-then-compress design:

1. Give every retained word and every observed character n-gram an explicit identity.
2. Store word and n-gram ids in trie-based indexes instead of arbitrary hash buckets.
3. Explore offline row sharing followed by a compact layout; the paper's
   intended similarity criterion differs from the current legacy exporter
   ([audit](docs/engram_storage.md#legacy-export-audit)).
4. Rewrite the remaining live rows into a compact, memory-mappable serving layout.

The goal is not to replace modern LLM embedding models. The goal is to provide a small, deterministic, inspectable lexical memory layer for retrieval, ranking, entity-heavy search, long-tail Chinese text, and low-latency feature generation.

## Highlights

- FastText-style training and inference for `skipgram`, `cbow`, and `supervised` models.
- Collision-free word and character n-gram lookup using trie indexes.
- UTF-8 aware character n-gram enumeration.
- Legacy prefix/suffix traversal and approximate row sharing; see the export audit.
- Mark-compact style row reorganization for dense compact ids.
- Single-file mmap serving index generated as `<output>.z`.
- Static library output (`libfasttext.a`) for embedding the implementation in services.
- Utility binaries for index generation and n-gram inspection.
- Experimental Python byte-exact row deduplication and block-compressed mmap
  lookup, preserving original row IDs independently of `.z` export.

## Reported paper results

The paper reports a large Chinese vocabulary benchmark with 30.1M words and 287.4M extracted n-grams. In that setting, the compressed model reduced memory from 145.2 GB for the original FastText baseline to 28.9 GB, reduced cold load time from 12.3 minutes to 3.2 minutes, and preserved downstream quality within 0.3 points of the hash-free model.

| Method | Memory | Load time | Storage |
| --- | ---: | ---: | ---: |
| Original FastText | 145.2 GB | 12.3 min | 89.4 GB |
| HashFree FastText | 287.4 GB | 28.7 min | 201.8 GB |
| Quantized FastText | 72.6 GB | 8.9 min | 44.7 GB |
| SVD Compression | 89.3 GB | 15.4 min | 62.1 GB |
| me_fasttext | 28.9 GB | 3.2 min | 18.6 GB |

These numbers are from the paper and will vary with corpus, vocabulary size, hardware, operating-system page cache behavior, and compression threshold.

They are not Engram benchmarks and have not been reproduced by the new row
store. The legacy exporter also differs from the paper's cosine-based method;
inspect the implementation audit before treating it as a paper reproduction.

## Repository layout

```text
.
|-- src/
|   |-- main.cc             # fasttext-compatible CLI entry point
|   |-- args.*              # command-line arguments and defaults
|   |-- dictionary.*        # word/ngram dictionary and UTF-8 n-gram extraction
|   |-- fasttext.*          # training, model IO, vector export, compact index export
|   |-- ftindex.*           # mmap serving index API
|   |-- index.cc            # helper for generating serving artifacts
|   |-- cut_ngram.cc        # helper for cutting n-gram trie by vocabulary
|   `-- print_ngrams.cc     # helper for inspecting n-gram trie contents
|-- tools/
|   `-- prepare_public_benchmark.py
|-- mylib/
|   |-- include/            # trie, hash, pub, spell, xml headers
|   `-- lib/                # bundled static libraries used by the Makefile
|-- Makefile
|-- fasttext                # prebuilt binary from the original environment
|-- index                   # prebuilt helper binary
|-- cut_ngram               # prebuilt helper binary
|-- print_ngrams            # prebuilt helper binary
`-- libfasttext.a           # prebuilt static library
```

## Build

### Requirements

- Linux is the tested target.
- C++ compiler with C++11 support.
- `make`, `ar`, pthreads, and POSIX mmap support.
- The bundled trie/pub libraries under `mylib/`.

The serving implementation uses Linux-specific mmap flags such as `MAP_POPULATE`, so non-Linux platforms may need small portability changes.

### Build from source

```bash
make opt
```

This builds:

- `fasttext`
- `index`
- `cut_ngram`
- `print_ngrams`
- `libfasttext.a`

For debug builds:

```bash
make debug
```

### Test helper scripts

The Python benchmark helper uses only the standard library. Run its tests with:

```bash
python3 -m unittest discover -s tests
```

To clean generated objects and the main binary:

```bash
make clean
```

## Quick start

### Train a skipgram model

Input format is plain text, one sentence or document per line:

```bash
./fasttext skipgram \
  -input data/train.txt \
  -output models/meft \
  -minn 2 \
  -maxn 6 \
  -thread 24
```

The training path saves:

| File | Purpose |
| --- | --- |
| `models/meft.bin` | Full fastText-style model |
| `models/meft.vec` | Text word-vector export |
| `models/meft.z` | Compact mmap serving index |
| `worddat.bin` | Intermediate word trie dump, written in the current working directory |
| `ngramdat.bin` | Intermediate n-gram trie dump, written in the current working directory |

### Train a supervised classifier

Supervised training follows the standard fastText label convention:

```text
__label__sports the team won the final
__label__finance quarterly revenue increased
```

```bash
./fasttext supervised \
  -input data/train.txt \
  -output models/classifier \
  -epoch 5 \
  -lr 0.1 \
  -wordNgrams 2
```

Evaluate and predict:

```bash
./fasttext test models/classifier.bin data/test.txt 1
./fasttext predict-prob models/classifier.bin data/test.txt 3
```

## Common commands

```bash
# Print word vectors from a trained model.
echo "example" | ./fasttext print-word-vectors models/meft.bin

# Print sentence vectors.
echo "example sentence" | ./fasttext print-sentence-vectors models/meft.bin

# Inspect the n-grams used for a word.
./fasttext print-ngrams models/meft.bin example

# Interactive nearest-neighbor lookup.
./fasttext nn models/meft.bin 10

# Interactive analogy lookup.
./fasttext analogies models/meft.bin 10

# Quantize a supervised model.
./fasttext quantize \
  -input data/train.txt \
  -output models/classifier \
  -cutoff 100000 \
  -qnorm \
  -retrain
```

## Serving with the compact index

The `.z` file is designed for mmap-based serving. It stores:

1. Metadata: number of words, number of compact n-gram rows, dimension, `minn`, and `maxn`.
2. A contiguous float matrix: word rows followed by compact n-gram rows.
3. A double-array trie mapping words to row ids.
4. A double-array trie mapping character n-grams to compact row ids.

Minimal C++ usage, with `MODEL_DIM` set to the dimension used when training the model:

```cpp
#include "src/ftindex.h"

#include <iostream>

int main() {
  void* mmap_addr = nullptr;
  size_t mmap_len = 0;
  FTIndex* index = ftindex_mmap("models/meft.z", &mmap_addr, &mmap_len);
  if (!index) {
    std::cerr << "failed to mmap index\n";
    return 1;
  }

  real vec[MODEL_DIM] = {0};
  getSentenceVector(index, "hello world", vec);

  for (int i = 0; i < MODEL_DIM; ++i) {
    std::cout << vec[i] << (i + 1 == MODEL_DIM ? '\n' : ' ');
  }

  ftindex_delete(index);
  ftindex_munmap(mmap_addr, mmap_len);
  return 0;
}
```

Compile against the static library and bundled dependencies:

```bash
c++ -std=c++11 -pthread \
  -I. -I./mylib/include \
  examples/query.cc \
  libfasttext.a \
  -L./mylib/lib -ltrie -lpub \
  -o query
```

## How compression works

At training time, the dictionary inserts retained words and UTF-8 character n-grams into trie structures instead of relying on `hash(ngram) % bucket`.

At export time, `FastText::saveIndex()` performs the compact-index pipeline:

1. Walk the n-gram trie, selecting shared rows with the legacy norm-ratio
   heuristic (the paper describes a cosine-based criterion).
2. Build a reversed/suffix view and repeat the same conservative merge process.
3. Mark live row ids, assign dense compact ids, and rewrite trie terminal values.
4. Serialize metadata, compact vectors, word trie, and n-gram trie into `<output>.z`.

Row sharing happens after training and the index remapping can be inspected.
The current norm-ratio check does not establish directional similarity or
lossless reconstruction; see the [export audit](docs/engram_storage.md#legacy-export-audit).

## Differences from upstream fastText

| Area | Upstream fastText | me_fasttext |
| --- | --- | --- |
| Subword ids | `hash(ngram) % bucket` | Exact trie id before compression |
| N-gram collisions | Arbitrary bucket collisions | No collision unless explicitly merged |
| Large-model serving | Load model structures from `.bin` | mmap compact `.z` index |
| Compression policy | Bucket size, pruning, quantization | Prefix/suffix row sharing plus mark-compact layout |
| Debuggability | Bucket contents are opaque | N-gram identities remain inspectable |

## Practical notes

- Rebuild binaries on the target Linux machine instead of relying on the checked-in prebuilt executables.
- The compact `.z` index is the preferred artifact for low-latency serving.
- The full `.bin` model is still useful for evaluation, CLI inspection, and regenerating derived artifacts.
- The trie dumps `worddat.bin` and `ngramdat.bin` are written to the current working directory, so run training from a predictable location.
- Compression ratios depend heavily on corpus language, n-gram length, vocabulary filtering, vector dimension, and similarity threshold.

## Citation

If this repository or the accompanying method is useful in your work, please cite:

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

## License

This codebase is derived from the public fastText C++ implementation and is
distributed under the MIT License. See [LICENSE](LICENSE). Before redistributing
binary artifacts, review any third-party or platform-specific dependency terms
that may apply to the bundled support libraries.
