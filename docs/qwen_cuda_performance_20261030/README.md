# Trained Qwen CUDA prefill and fixed greedy generation, 2026-10-30

**All three storage modes preserve same-schedule outputs on a real A100. On
this small trained memory branch, native timings are close to resident timings;
no model speedup or meaningful whole-model memory saving is established.**
The [larger trained-table gather experiment](../cuda_gather_evidence_20261030/README.md)
separately shows substantial native CPU query costs.

[`benchmark_tinyengram_cuda.py`](../../tools/benchmark_tinyengram_cuda.py) loads
the previously trained Qwen3-0.6B checkpoint/bundle in one process per mode.
All three modes use the **same CPU hash/history and layer-prefetch wrapper**.
Resident selects original dense GPU rows; dense offload replaces only the CPU
reader while retaining the production staging schedule; native uses the
production mmap reader. This controlled resident baseline avoids conflating
table placement with the upstream GPU hash-ID round trip.

Original weights/dtype/addressing remain unchanged. Full prefill logits and
fixed-length generated IDs are hashed and compared to the resident reference
for every case. Dense/native outputs match byte for byte; cached and uncached
generation are compared within their own schedules. No retraining or post-
compression adaptation occurs. The branch is **648 × 8 float32 values** at one
layer on a frozen pretrained backbone, as documented in the
[original training evidence](../tinyengram_qwen_evidence_20261030/README.md).
Its native artifact is larger than its 20,736 raw vector bytes because of
format overhead. It is not a trained DeepSeek Engram checkpoint.

| Input / method | Resident median ms | Dense offload median ms | Native median ms |
| --- | ---: | ---: | ---: |
| 32-token prefill | 76.368 | 74.722 | 78.040 |
| 32-token input, 16 tokens cached generation | 1,084.318 | 1,067.934 | 1,094.645 |
| 32-token input, 16 tokens uncached generation | 1,225.496 | 1,213.364 | 1,236.843 |
| 128-token prefill | 77.933 | 79.853 | 78.284 |
| 128-token input, 16 tokens cached generation | 1,087.648 | 1,076.802 | 1,100.401 |
| 128-token input, 16 tokens uncached generation | 1,252.510 | 1,235.332 | 1,286.719 |

Native median differences versus resident range from about +0.45% to +2.73%
in these six cases. Five samples per case are diagnostic; no statistical claim
of equality or a universally small overhead follows. Raw p95 and every sample
are in reports. Native cached generation produces about 14.5–14.6 tokens/sec
**including prompt prefill, host hashing and per-token greedy ID readback**;
this is not steady-state decoder-only throughput or a vLLM serving benchmark.
Uncached generation repeatedly processes the full growing prefix.

The maximum observed peak Torch allocation across these cases is
2,593,456,128 resident versus 2,593,435,136 bytes for either offload mode,
only **20,992 bytes apart**. Backbone/logits/KV dominate. Driver/context memory
is excluded. This supports no substantial whole-model memory headline.

Batch 1, eager unpadded inference, original float32 checkpoint, deterministic
algorithms, TF32/cuDNN benchmark disabled, Torch 1/1 threads. Inputs are first
32/128 tokenizer IDs of the chronological validation segment from the first
16,384 cached TinyShakespeare characters, using the original 80/10/10 character
split. 16 fixed greedy tokens, 2 warmups and 5 timed repeats per case. Staging:
2 slots × 256 rows × 8 float32 values, at most 16 KiB; native cache budget 1 MiB.
Correctness logits readback/hashing is outside timed prefill. Generation timing
includes its required token readback. Outputs are released between repeats;
real forward/generation cache lifetimes are retained within each call.

GPU 7: A100-SXM4-80GB, two EPYC 7742 CPUs / 256 logical CPUs, Linux
5.4.0-42-generic, driver 535.54.03, Python 3.11.15, Torch 2.11.0+cu130,
CUDA runtime 13.0, Transformers 4.57.3. Existing isolated compatibility
environment, local checkpoint/bundle/corpus, no downloads, training or paid
services. Other cards and CPU/I/O were shared. Empty-device gate and competing-
client checks bracket each timed case; these do not reserve hardware or rule
out short interference. A 0.04 Torch allocator fraction (3,399,509,606 bytes)
plus 2 GiB free-memory reserve excludes context/library allocations.

Code commit `89e09f4`, with exact commands and source/input hashes in
[resident.json](resident.json), [dense-offload.json](dense-offload.json), and
[native.json](native.json). Native's first run was interrupted by an occupancy
program; [that record](native-interrupted.json) is retained. The completed
native retry uses the unchanged, checksum-inspected resident reference from
the same GPU/software/settings; it does not reselect input or timing samples.
Each completed mode has all six cases. No background workers remain.

```sh
export CUDA_VISIBLE_DEVICES=YOUR_SELECTED_IDLE_CARD OMP_NUM_THREADS=1 USE_TF=0
for mode in resident dense-offload native; do
  python -m tools.benchmark_tinyengram_cuda \
    --mode "$mode" --checkpoint /path/to/checkpoint --bundle /path/to/bundle \
    --corpus /path/to/cached/corpus.txt --upstream-dir /path/to/pinned/upstream \
    --native-library build/libmeft_rows.so --output-dir artifacts/fresh-qwen-cuda \
    --tokens 32,128 --decode-tokens 16 --warmup 2 --repeat 5 || break
done
```

This records executable inference behavior, not model quality, large learned-
memory scaling or deployable service performance. [The prior real CUDA correctness
checks](../cuda_a100_evidence_20261030/README.md) cover stream lifetime/cleanup and
same-schedule cached logits on validation/test.
