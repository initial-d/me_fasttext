# me_fasttext

Memory-Efficient FastText: a FastText-derived C++ prototype that replaces hashed character n-gram buckets with exact trie indexes, then compresses the learned n-gram matrix through structure-aware row sharing and mark-compact layout.

This repository accompanies the paper [Memory-Efficient FastText: A Comprehensive Approach Using Double-Array Trie Structures and Mark-Compact Memory Management](https://arxiv.org/abs/2506.01254).

> Status: research / production-prototype code. The command-line interface remains close to the original fastText workflow, while the storage and serving path has been extended for large-vocabulary, memory-sensitive deployments.

## Start here

- [Benchmark protocol](docs/benchmark_protocol.md): how to compare memory, load time, latency, and task quality.
- [Outreach notes](docs/outreach.md): concise wording for adding the project to NLP, embedding, retrieval, or systems lists.
- [Citation metadata](CITATION.cff): machine-readable citation information for GitHub and reference managers.
- [Release notes](docs/release_notes_v0.1.0.md): scope and caveats for the first research-prototype release.

## Where it fits

Use `me_fasttext` when you need a small lexical memory layer rather than a
large semantic encoder:

| Use case | Why this project is relevant |
| --- | --- |
| OOV-heavy retrieval | Character n-grams preserve signals for rare and unseen words. |
| Entity-heavy search | Exact trie ids make subword identities inspectable before compression. |
| RAG prefiltering | A low-cost lexical stage can reduce candidates before dense reranking. |
| Memory-sensitive serving | The compact `.z` artifact is designed for mmap-based loading. |
| Long-tail Chinese or multilingual text | UTF-8 n-gram enumeration avoids byte-level shortcuts. |

## Why this exists

Standard fastText represents a word by averaging its word vector and character n-gram vectors. To avoid storing every observed n-gram, the reference implementation hashes subwords into a fixed bucket table. That keeps memory bounded, but unrelated n-grams can collide into the same row.

`me_fasttext` follows an exact-then-compress design:

1. Give every retained word and every observed character n-gram an explicit identity.
2. Store word and n-gram ids in trie-based indexes instead of arbitrary hash buckets.
3. Merge only structurally related n-gram rows whose learned vectors are extremely close.
4. Rewrite the remaining live rows into a compact, memory-mappable serving layout.

The goal is not to replace modern LLM embedding models. The goal is to provide a small, deterministic, inspectable lexical memory layer for retrieval, ranking, entity-heavy search, long-tail Chinese text, and low-latency feature generation.

## Highlights

- FastText-style training and inference for `skipgram`, `cbow`, and `supervised` models.
- Collision-free word and character n-gram lookup using trie indexes.
- UTF-8 aware character n-gram enumeration.
- Prefix and suffix based n-gram row sharing with high cosine-similarity checks.
- Mark-compact style row reorganization for dense compact ids.
- Single-file mmap serving index generated as `<output>.z`.
- Static library output (`libfasttext.a`) for embedding the implementation in services.
- Utility binaries for index generation and n-gram inspection.

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

1. Walk the n-gram trie in prefix order and merge structurally related rows when cosine similarity is high enough.
2. Build a reversed/suffix view and repeat the same conservative merge process.
3. Mark live row ids, assign dense compact ids, and rewrite trie terminal values.
4. Serialize metadata, compact vectors, word trie, and n-gram trie into `<output>.z`.

This differs from hash buckets in one important way: row sharing happens after training and only between n-grams that can be audited through lexical structure and vector similarity.

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
