# Full official Engram CPU forward: exact storage replacement — 2026-10-18

**Integration evidence only: random small weights, local normalized-token
fixture, CPU inference.** All six cases (Python/native × float32/float16/bfloat16)
preserve every original row and the complete official Engram module output
byte-for-byte. This extends the earlier hash/gather-only verification.

[Raw report](engram_full_adapter_verification_20261018.json) contains the exact
command, commit, source/library/input/output hashes and dependency versions.

The source is pinned to official commit
`fb7f84a21f91223715394a33a1dc24bbfb7f788e`, SHA256
`9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044`.
The full source is imported after hash verification; the main demo is not run.
Only tokenizer construction is substituted with a 32-ID local identity fixture.
Original hash initialization, prime allocation, addressing, Engram forward,
key/value projections, RMSNorm gates and ShortConv execute unchanged. Tiny
configs use 2 heads/order, orders2/3, 4 values/head, hidden8/hyperconnection2.

At the current official demo, a frozen CPU table can be exported and replaced:

```python
sizes = [n for group in engram.hash_mapping.vocab_size_across_layers[engram.layer_id]
         for n in group]
export_demo_embedding(engram.multi_head_embedding, sizes, directory,
                      backend="native", deduplicate=False)
mapped = MappedMultiHeadEmbedding(directory, native_library=library)
engram.multi_head_embedding = mapped
# Supply CPU input IDs/hidden states and call in inference mode.
# Close mapped when serving is finished.
```

This replacement path initially constructs the default table. A proposed
upstream `embedding_factory` hook would allow opening a previously exported
store during construction, avoiding that initial allocation. It is a narrow
extension point; it neither bundles this project's code nor supplies offload,
compression, checkpoint loading or a training implementation.

The [published upstream draft PR #27](https://github.com/deepseek-ai/Engram/pull/27)
now has a [runnable factory quickstart](engram_quickstart.md) and six-case
verification using real Python/native mapped backends. It remains unmerged as
checked on 2026-10-19; this report still executes unchanged official main.

## Reproduce

```bash
python -m pip install torch numpy sympy transformers tokenizers
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
curl -fL https://raw.githubusercontent.com/deepseek-ai/Engram/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py \
  -o /tmp/engram_demo_v1.py
python tools/demo_engram_full_adapter.py \
  --official-source /tmp/engram_demo_v1.py \
  --native-library build/libmeft_rows.dylib \
  --output-dir artifacts/native-ngram-20261018/full-engram
```

Use `.so` and `make native` on Linux. The run uses Python3.12.14,
PyTorch2.14.1, NumPy2.5.4, sympy1.14.0, transformers5.20.1,
tokenizers0.23.3, Apple M5/10 logical CPUs/32 GiB/macOS26.6.2 arm64,
one torch/interop thread, seed7, Apple clang21.0.0 native build. Local CPU only;
no paid API/rented compute or cost/energy estimate.

This is not tokenizer fidelity, learned Engram compression, full backbone/LM
logits, task quality, model-sized memory, serving latency or GPU validation.
Tiny random tables expand after packaging; their sizes in the report are
integration controls, not learned-weight compression evidence. Character-ngram
[approximate compression results](approximate_ngram_compression_20261018.md)
remain a different training experiment and cannot establish Engram savings.
