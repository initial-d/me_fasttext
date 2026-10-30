# Three storage baselines for device gather

[`prepare_device_gather.py`](../tools/prepare_device_gather.py) creates fixtures
from **original trained memory weights and original held-out query IDs**.
[`benchmark_device_gather.py`](../tools/benchmark_device_gather.py) runs each
baseline in a fresh process. CPU protocol validation passes for trained orders
4/8/12, seed 3. **[Completed A100 CUDA measurements](cuda_gather_evidence_20261030/README.md)**
now record all three modes and two staging configurations. A separate
[trained Qwen performance experiment](qwen_cuda_performance_20261030/README.md)
measures the much smaller memory inside the full Transformer. The earlier
[A100 correctness result](cuda_a100_evidence_20261030/README.md) is retained.

| Mode | Table placement | Work measured |
| --- | --- | --- |
| resident | Dense table on target device | CPU ID validation, ID transfer, device index-select |
| dense-offload | Original dense table on CPU | CPU index-select, identical production host slots/copy stream/events |
| native | Compressed mmap table on CPU | Native C gather, identical host slots/copy stream/events |

Dense offload subclasses the production staging adapter, changing only the CPU
row reader. This separates generic offload savings from native storage/query
costs. Native workers never open `dense.pt`; checksums and expected output hashes
allow validation without retaining the source table. A regression test removes
that file entirely and still verifies native outputs, including signed zero
and a noncanonical NaN payload.

Prepared tables are independent character-GRU references from
[the existing trained causal-memory experiment](causal_lm_evidence_20261024/README.md),
not TinyEngram/DeepSeek weights. Orders 4/8/12 contain 11,760 / 36,778 / 43,152
original rows of 64 float32 values. No row duplication manufactures larger
tables, and no approximate cosine/norm variants enter this comparison.
For each order, 8 distinct chronological test windows per size use
1/32/128/512 positions after the original 64-character burn-in. All suffix
slots, repeated IDs and missing-key zero IDs are retained. Timing repeats these
fixed windows; this is a gather microbenchmark, not full-model throughput or a
production trace.

```sh
python -m tools.prepare_device_gather \
  --training-dir artifacts/causal-lm-compression-20261024 \
  --corpus artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt \
  --order 12 --seed 3 --positions 1,32,128,512 --queries 8 \
  --native-library build/libmeft_rows.so \
  --output-dir artifacts/fresh-device-fixture-order12

# CPU-only protocol check; no CUDA probe or GPU computation.
python -m tools.benchmark_device_gather \
  --fixture artifacts/fresh-device-fixture-order12 \
  --native-library build/libmeft_rows.so --device cpu \
  --output-dir artifacts/fresh-device-cpu --warmup 1 --repeat 2

# Run only after the selected card is available.
CUDA_VISIBLE_DEVICES=YOUR_SELECTED_IDLE_CARD \
python -m tools.benchmark_device_gather \
  --fixture artifacts/fresh-device-fixture-order12 \
  --native-library build/libmeft_rows.so --device cuda \
  --output-dir artifacts/fresh-device-cuda --warmup 3 --repeat 10 \
  --staging-rows 256 --staging-slots 2 --cache-bytes 4194304
```

Preparation checks training-report/checkpoint/corpus/test-ID fingerprints and
every exported native row. All output digests are computed from original dense
weights, and each worker checks every prepared query before warm timing.
Fresh output directories preserve previous runs. Source hashes, commit,
commands, fixture inputs, runtime, memory, raw per-query times and failures
are recorded. Fixtures retain weights/IDs locally; published manifests expose
hashes and aggregate shapes only.

On CUDA, expose exactly one card. The worker checks other compute processes
and low utilization before allocation, and checks competing processes before
and after each timed case. Competition blocks the run rather than producing a
speed conclusion. This is a sampled gate, not a resource reservation; short
interference and shared CPU/I/O effects remain possible. The default 0.04 Torch
allocator fraction and 2 GiB free-memory reserve exclude driver/context costs.

Warm wall-clock timing synchronizes before/after each query and includes CPU
validation/gather, ID or vector H2D and device readiness. Correctness readback
and hashing are excluded. First-pass diagnostics include readback; integrity
checks already read files, so neither first-pass nor open time establishes
disk-cold behavior. No global page-cache flush occurs on the shared server.

Torch allocated/reserved/peak GPU bytes and Linux current/process-peak RSS are
reported separately. CPU protocol checks on macOS expose process-peak RSS;
current RSS is unavailable. Peaks include initialization/transient tensors;
warm snapshots reflect each baseline's retained state. Native cache and pinned
staging are separate budgets. One output lives at a time, so full-model layer
prefetch outputs, KV cache and backbone are outside this measurement. Increasing
query size may increase output memory even with bounded staging.

Local preparation uses Python 3.12.14 / PyTorch 2.14.1, macOS M5 CPU, Torch 1/1
threads, cached corpus and existing trained checkpoints; no downloads,
retraining or paid services. [Preparation manifests and CPU raw reports](device_gather_preparation_20261030/README.md)
record the completed protocol checks. The later CUDA reports and actual
measurement curves are linked above; no projected speed curve is published.
