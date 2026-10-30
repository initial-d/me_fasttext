# Host memory to device gather and TinyEngram prefetch

`meft-memory` 0.3.0 adds `StagedMultiHeadEmbedding` and device placement for the
TinyEngram checkpoint loader. Tables, original hash IDs and head offsets stay
on CPU. Only the queried vectors are copied to the target device, without
changing their dtype or bytes. The backbone, gates, KV cache and convolution
history live on that device; raw-token history remains on CPU.

CPU and synchronous MPS checks pass locally. **CUDA correctness now passes on a
real A100 with PyTorch 2.11.0+cu130**, including pinned staging, copy events,
non-default consumer streams, allocation lifetime and partial-failure cleanup.
[The trained Qwen CUDA replay](cuda_a100_evidence_20261030/README.md) preserves
same-schedule full/cached logits and fixed greedy tokens byte for byte.
Shared-card correctness does not establish performance. Later
[three-baseline A100 gather measurements](cuda_gather_evidence_20261030/README.md)
and [controlled Qwen inference timings](qwen_cuda_performance_20261030/README.md)
record actual costs on a card that passed sampled competition checks. These
small experiments do not establish large-table memory gains or production
serving performance. Skips do not constitute validation.

The [trained Qwen replay reports](device_memory_evidence_20261030/README.md)
record MPS, installed-wheel CPU and the earlier local CUDA blocker, retained as
historical evidence. The later A100 result is linked above.

## Use a trained TinyEngram bundle

Install the Torch or TinyEngram extra from source or a compatible release wheel:

```sh
pip install 'meft-memory[tinyengram] @ git+https://github.com/initial-d/me_fasttext.git@v0.3.0'
```

The [checkpoint workflow](tinyengram_qwen_evidence_20261030/README.md) explains
conversion and pinned upstream source requirements. No model is downloaded by
the loader. Existing 0.2.0 bundles need no conversion or format change.

```python
from meft_memory import load_tinyengram_checkpoint

with load_tinyengram_checkpoint(
    "native-bundle", upstream_dir="pinned-tinyengram-source",
    device="cuda", staging_rows=65536, staging_slots=2,
    native_cache_bytes=8 * 1024**2,
) as model:
    # Tokenizer outputs are CPU integer [batch, tokens], without padding.
    result = model.generate(cpu_token_ids, max_new_tokens=32)
    # result stays on CPU; logits from forward() live on the model device.
```

Set `device="mps"` for synchronous MPS inference, or `device="cpu",
prefetch_memory=True` to check the scheduling logic without an accelerator.
The default loader retains the existing CPU path. Eager, unpadded inference
and fixed-length greedy decode restrictions still apply.

The wrapper hashes all layers before decoder execution. Each Engram layer
waits for its own prefetched vectors. It preserves original addressing and
checks that the layer's hash IDs match the prefetched query. It avoids the
upstream layer's GPU → CPU → GPU hash-ID round trip.

## Integrate another model's gather

```python
from meft_memory import StagedMultiHeadEmbedding

with StagedMultiHeadEmbedding("exported-memory", device="cuda") as memory:
    ticket = memory.prefetch(cpu_head_local_ids)
    # Run other backbone work on the consumer stream here.
    vectors = ticket.wait()  # wait event on current CUDA stream
    # Same shape/dtype as original multi-head embedding(ids).
```

`prefetch()` hashes nothing; callers supply the original CPU head-local IDs.
It synchronously queries the C reader into pinned CPU buffers and enqueues H2D
copies on a dedicated CUDA stream. `wait(stream)` makes that stream wait for
the copy event and records the returned tensor's allocator ownership. Call it
for **every** CUDA stream that consumes the output. A returned output belongs
to its query and survives later queries and `close()`.

The adapter is immutable and has a fixed device/dtype. Parent `.to(device)`
preserves the CPU metadata; dtype or target changes are rejected. Reopen from
the artifact in each worker. Concurrent gather/close and serialization of live
staged adapters are unsupported. CUDA graph capture is rejected.

## Memory and transfer limits

Host staging is bounded by `staging_slots × staging_rows × row_bytes`. CUDA
staging uses pinned memory; MPS uses pageable buffers with synchronous copies.
Buffers are allocated lazily and reused only after the previous copy completes.
More chunks than slots can block the CPU while waiting for a slot. Native C
scratch uses the separate `query_chunk_rows` setting.

The bound excludes query IDs, full per-query output tensors, prefetched outputs
for all layers, the native block cache, mmap/RSS, model weights and KV state.
All prefetched layer outputs are retained until that forward finishes. A large
prefill can therefore consume substantial device memory. Controlled Qwen
end-to-end timings are linked above; transfer overlap has not been isolated,
and large-table whole-model GPU-memory savings remain unmeasured.

## Reproduce on real hardware

```sh
MEFT_TINYENGRAM_SOURCE=/path/to/pinned/source \
  python -m unittest tests.test_device_row_adapter tests.test_tinyengram_checkpoint -v

python -m tools.verify_tinyengram_device \
  --checkpoint /path/to/trained/checkpoint --bundle /path/to/native-bundle \
  --corpus /path/to/local/corpus --upstream-dir /path/to/pinned/source \
  --device cuda --report-out artifacts/device-cuda.json
```

The installed command is `meft-tiny-verify-device`. An unavailable device writes
`status: blocked` and exits 2 before loading weights. Failures retain their
report; use a fresh output path for every run. The verifier checks every memory
row, full logits, same-schedule cached logits, tokenization and fixed greedy
tokens against the resident checkpoint on the **same device**. Cold latency
samples include readback and are diagnostic, not a speed benchmark. Both models
coexist, so the verifier does not compare peak memory.
