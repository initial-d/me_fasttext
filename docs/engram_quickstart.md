# Engram memory storage: run, evaluate, cite

`me_fasttext` provides immutable CPU n-gram table storage and gather adapters.
For an existing Engram memory table, keep its original hash IDs and head offsets;
replace the gather storage behind `MultiHeadEmbedding`. The default path recovers
the original bytes. This project also studies approximate structural row sharing
on trained character n-grams, with an explicit cosine/norm distortion budget.

## Choose a verified boundary

| What you want to do | Start here | Evidence |
| --- | --- | --- |
| Verify current official Engram module outputs | [Official CPU forward](engram_full_adapter_verification_20261018.md) | Six small random-weight cases preserve full gates/ShortConv output and all rows |
| Open a frozen table without allocating the default table first | Factory example below | Proposed fork API; six Python/native × dtype cases, default allocation prevented |
| Evaluate learned redundancy and compression | [Trained study](trained_ngram_similarity_20261018.md), [distortion curves](approximate_ngram_compression_20261018.md) | Nine trained character-ngram tables; 135 approximate cases, not trained Engram weights |
| Integrate another model/runtime | [C ABI and model guide](native_ngram_memory.md) | CPU gather boundary; GPU/checkpoint/runtime work remains |

The [upstream draft PR #27](https://github.com/deepseek-ai/Engram/pull/27)
proposes an optional `embedding_factory`. It is **open and unmerged** as checked
on 2026-10-19. Current official main can use post-construction replacement;
the factory example needs the pinned proposal source. This is a submitted
contribution, not official adoption or endorsement.

## Run the factory integration example

From a fresh checkout, install the demo dependencies and build the standalone C
library. Linux needs a C compiler, Make and zlib development headers:

```bash
python -m pip install torch numpy sympy transformers tokenizers
make native
curl -fL https://raw.githubusercontent.com/initial-d/Engram/8b4155c2f8fc886ef27973d1164b798126f90c0d/engram_demo_v1.py \
  -o /tmp/engram_factory_v1.py
python tools/demo_engram_factory.py \
  --proposal-source /tmp/engram_factory_v1.py \
  --native-library build/libmeft_rows.so \
  --output-dir artifacts/engram-factory
```

macOS uses `make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib` and
`build/libmeft_rows.dylib`. The script checks the proposal SHA256 before importing
it. It creates tiny random-weight fixtures locally, exports each table, then
constructs a second Engram module through the real mapped backend. Default table
construction is patched to fail, so a successful run proves it was avoided.
Original downstream weights are restored before comparing full module output.

The harness runs six cases: float32/float16/bfloat16 × Python/native. It executes
original hash initialization, addressing, projections, gates and ShortConv.
Tokenizer construction uses a tiny identity fixture; no tokenizer or checkpoint
download is made. Use a fresh output directory. Results are integration checks,
not representative random-weight compression or model-quality evidence.

Recorded 2026-10-19 run: [JSON](engram_factory_verification_20261019.json).
All six cases avoid default allocation and preserve full output bytes. Commit,
exact command, source/library/output hashes and environment are in the report:
Python3.12.14/PyTorch2.14.1/NumPy2.5.4, transformers5.20.1/tokenizers0.23.3/
sympy1.14.0, Apple M5/10 logical CPUs/32 GiB/macOS26.6.2 arm64, one torch/interop
thread, Apple clang21.0.0 native build. Local CPU only, no paid API/rented compute
or monetary/energy estimate. Avoided table allocation is a functional check;
total process memory and serving speed were not measured.

## Use your own immutable table

After exporting each actual layer's table with `export_demo_embedding`, bind the
checked loader to that layer's directory:

```python
from functools import partial
import torch
from tools.engram_row_adapter import open_demo_embedding

factory = partial(open_demo_embedding, "/path/to/exported/layer-1",
                  native_library="build/libmeft_rows.so")
# Engram imported from the proposal version above:
engram = Engram(layer_id=1, embedding_factory=factory)
try:
    # Restore this layer's non-embedding weights from its own checkpoint.
    # Verify missing/unexpected keys and outputs; supply compatible CPU tensors.
    engram.eval()
    with torch.inference_mode():
        output = engram(hidden_states, input_ids)
finally:
    engram.multi_head_embedding.close()
```

The loader checks the requested head layout and dimension, including different
head partitions with the same total row count. A mismatch closes the store and
raises an error. Each layer/worker needs its own handle. The factory does not
restore projections/gates, tokenizer settings or the rest of a checkpoint.
Calling `.to(dtype=...)` does not recode stored rows; use the exported dtype and
compatible CPU projections. Training and GPU/offload scheduling are unsupported.
The [current-main replacement example](engram_full_adapter_verification_20261018.md)
works without the proposal but initially allocates the original embedding table.

## Interpret the compression results

| Mode | Learned character-ngram result | What it preserves |
| --- | --- | --- |
| Native exact block storage | About 5% size saving | Every original row byte |
| Offline XOR/update-group prediction | Estimated 6.9–9.1% saving | Every original bit; predictor not in native reader |
| Approximate cosine + norm ≥0.99, max order12 | Actual 2.14–2.16× storage reduction | Original IDs; vectors change, mean ngram-component relative L2 about 0.48% |
| Approximate cosine ≥0.98 + norm ≥0.99, max order12 | About 2.35× | Original IDs; vectors change, mean component relative L2 about 0.52%, worst about 3.79% |

These are trained **character-ngram** reference results, not Engram checkpoint
numbers. Hash buckets in a learned Engram table do not automatically expose
character-prefix/suffix relationships or identical training schedules. Approximate
vector error is not accuracy/perplexity loss. Read all seeds, row-level tail
errors and byte denominators in the linked reports before comparing methods.

## Reproduce or build on this work

To compare an existing frozen memory table, record its source/license/hash,
layer/head layout, dtype, hash/tokenizer settings and original query IDs. Report
all rows and repeated/boundary queries, total artifact bytes, builder memory,
RSS and warm/cold gather latency against resident tensor and raw mmap baselines.
For approximation, add direct source-to-final-representative cosine/norm/error
distributions and held-out task quality on the complete model.

[Submit a benchmark report](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml),
[ask an integration question](https://github.com/initial-d/me_fasttext/issues/new?template=evaluation-question.yml),
or [record a reuse/citation](https://github.com/initial-d/me_fasttext/issues/new?template=showcase.yml).

For the original FastText method, cite the
[paper](https://arxiv.org/abs/2506.01254). For these later native adapters and
Engram experiments, also identify the exact software commit and linked result;
the old paper does not report these new experiments. The
[citation guide](citation_guide.md#cite-the-native-ngram-artifact) provides
software BibTeX and evidence-specific wording.
