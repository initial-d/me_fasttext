# Real A100 CUDA correctness, 2026-10-30

**The native host-memory backend preserves trained Qwen3-0.6B outputs on real
CUDA hardware. GPU speed and large-table memory gains remain unmeasured.**
The card was shared with existing training jobs; those jobs were left running.
Performance work is deferred until the owner supplies an idle card.

| Check | Observed result |
| --- | --- |
| CUDA unit/integration tests | 16 run, 13 passed, 3 expected skips |
| Pinned buffers, non-default consumer streams, owned output lifetime | Actual CUDA tests passed |
| Partial-copy failure cleanup and slot reuse | Actual CUDA test passed |
| Reconstructed trained checkpoint | All 320 saved tensors match original SHA-256 fingerprints |
| Validation/test full logits vs resident CUDA reference | Byte exact |
| Same-schedule cached logits vs resident CUDA reference | Byte exact |
| Fixed 8-token greedy continuation, cached and uncached | Identical IDs |
| Raw-token history / KV and convolution history | CPU / CUDA |
| Native pinned host staging | 2 slots, 2,048 bytes; 104 copy chunks, 102 slot reuses |

The real trained branch has **648 rows × 8 float32 values**, one memory layer,
original head sizes 257/263/61/67. This table is deliberately small: raw vectors
occupy 20,736 bytes while its native table occupies 25,327 bytes **before adapter
metadata**. It establishes correctness, with format overhead exceeding vector
size here; it does not establish useful compression on this table.
The [original CPU/MPS training evidence](../tinyengram_qwen_evidence_20261030/README.md)
describes how the memory/gates/conv branch was trained on a frozen backbone.
This is TinyEngram integration, not a trained DeepSeek Engram checkpoint.

Resident and native verification ran in **separate processes** using
[`verify_tinyengram_shared_cuda.py`](../../tools/verify_tinyengram_shared_cuda.py)
at commit `11cad573441918bec7a1a78b2ac5db311dcbb0c0`. CPU safetensor logits bridge
the two phases. The verifier rejects changed checkpoint/config/corpus/bundle,
source hashes, reference checksums, device or stable software settings.
The reference tensors stay local; their SHA-256 is in [resident.json](resident.json),
whose SHA-256 is in [native.json](native.json). Original tensor fingerprints are
in [expected-tensors.json](expected-tensors.json).

The unchanged cached TinyShakespeare corpus has SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
Evaluation takes the first 32 tokenizer IDs from each held-out validation/test
segment of the first 16,384 characters, using a contiguous 80/10/10 split.
It compares full and cached schedules separately, with 8 generated tokens.
Cross-schedule maximum absolute logit differences were 3.71933e-5 on validation
and 3.62396e-5 on test, with zero sampled top-1 changes. Byte equality applies to
matching schedules; it is not a claim of general floating-point associativity.

Hardware: A100-SXM4-80GB physical GPU 3; two AMD EPYC 7742 CPUs, 256 logical CPUs.
Software: Python 3.11.15, PyTorch 2.11.0+cu130, CUDA runtime 13.0,
Transformers 4.57.3, tokenizers 0.22.2, safetensors 0.7.0;
Linux 5.4.0-42-generic, driver 535.54.03. This records the observed working
stack, not a claim that other driver/runtime combinations are supported.
CUDA determinism enabled; TF32 and cuDNN benchmarking disabled; Torch 1/1 threads.
The [hardware/load snapshot](hardware-and-load.json) was collected after the
completed correctness processes. It is not a time-series utilization trace.

The trained verification used a Torch allocator fraction of 0.04, a cap of
3,399,509,606 bytes, and a 2 GiB free-memory preflight reserve. These do not
reserve resources or bound CUDA context/library allocations; other jobs may
change available memory after the check.

| Process | Peak Torch allocated bytes | Peak Torch reserved bytes |
| --- | ---: | ---: |
| Resident reference | 2,473,268,736 | 2,495,610,880 |
| Native backend | 2,473,247,232 | 2,497,708,032 |

These are correctness-process diagnostics dominated by the Qwen backbone,
outputs and verification tensors. They are not a fair memory-efficiency
benchmark; the small differences do not support a memory-saving headline.
The separate unit run used a 0.01 allocator fraction and peaked at 10,014,208
allocated / 41,943,040 reserved bytes. Its artificial bit/lifetime fixtures are
interface tests, not trained compression evidence.

No retraining or external downloads were needed for the remote replay. The
unchanged backbone came from a maintainer-provided local Qwen checkpoint; a
4,581,293-byte archive carried the previously trained branch, configs/tokenizer,
corpus and fingerprints. [The reconstruction command](original-reconstruction-command.txt)
verifies every tensor before and after serialization; [its report](reconstruction.json)
records success. Corpus, weights and large reference logits remain local.
Cost: existing shared hardware/environment, no paid services. An isolated
compatibility environment was used without changing the supplied environment.

Raw records: [CUDA tests](cuda-unit-tests.txt) · [original test command](original-unit-command.txt)
· [native C build](native-build.txt) · [resident log](resident.txt)
· [native log](native.txt) · [reconstruction log](reconstruction.txt).
Tokenizer regex warnings are retained. Equality proves unchanged tokenizer IDs
for this replay, not independent tokenizer correctness or full model quality.

To reproduce after obtaining a suitable card, use the same trained checkpoint,
exported bundle and pinned upstream source from the original training workflow:

```sh
export CUDA_VISIBLE_DEVICES=YOUR_SELECTED_IDLE_CARD
export OMP_NUM_THREADS=1 USE_TF=0
for phase in resident native; do
  python -m tools.verify_tinyengram_shared_cuda \
    --phase "$phase" --checkpoint /path/to/trained/checkpoint \
    --bundle /path/to/native/bundle --corpus /path/to/cached/corpus.txt \
    --upstream-dir /path/to/pinned/upstream --native-library build/libmeft_rows.so \
    --output-dir artifacts/fresh-cuda-correctness \
    --allocator-fraction .04 --reserve-bytes 2147483648 \
    --eval-tokens 32 --decode-tokens 8 --max-characters 16384 \
    --staging-rows 32 --staging-slots 2 || break
done
```

This reproduces correctness, not speed. [The prepared three-baseline protocol](../device_gather_benchmark.md)
separates resident GPU, dense CPU offload and native CPU offload costs.
The earlier [local CUDA-blocked report](../device_memory_evidence_20261030/cuda-blocked.json)
is preserved.
