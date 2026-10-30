# Device gather preparation, 2026-10-30

**CPU protocol validation passed; CUDA performance remains pending.**
The [preparation and benchmark commands](../device_gather_benchmark.md) run at
commit `8f6ae4e563c9c0bf462e40a4e93cc6c2c76f608a`, with source hashes in each
report. Fixtures contain original learned character-GRU memory, not artificial
vectors or the much smaller TinyEngram/Qwen CUDA correctness table.

| Trained max order / seed | Original rows × dimension | Fixture manifest | CPU protocol report |
| --- | --- | --- | --- |
| 4 / 3 | 11,760 × 64 | [manifest](order-4-seed-3-manifest.json) | [report](cpu-order-4-seed-3.json) |
| 8 / 3 | 36,778 × 64 | [manifest](order-8-seed-3-manifest.json) | [report](cpu-order-8-seed-3.json) |
| 12 / 3 | 43,152 × 64 | [manifest](order-12-seed-3-manifest.json) | [report](cpu-order-12-seed-3.json) |

Every native row matches its original trained vector bytes. All three baseline
workers produce byte-exact outputs for every prepared query at all four sizes
(1/32/128/512 chronological held-out positions, 8 windows each). They run in
separate processes with the same IDs and original float32 dtype. These CPU
times validate executable plumbing and are **not GPU performance evidence**.
No CPU-to-GPU pinned copy or CUDA overlap occurs in CPU mode.

Mac M5 CPU, Python 3.12.14, PyTorch 2.14.1, Torch 1/1 threads;
cached TinyShakespeare and previously trained checkpoints, no retraining,
downloads or paid services. Exact source/checkpoint/corpus/test-ID hashes,
commands, baseline RSS and diagnostic raw times are in the manifests/reports.
Weights, token/key content and query tensors remain local. Source/native
artifact hashes differ across platform builds as expected; byte outputs do not.

[Local tests](local-tests.txt): 97 run, 3 environment skips. The new protocol
checks verify special float bytes, input corruption rejection, other-process
detection and native validation without access to the dense source file.
These handcrafted bit fixtures are tests, not trained compression evidence.

The same scripts and real fixtures are staged in the owner's remote work
subdirectory, using the isolated compatibility environment. **Remote Linux CPU
validation also passed** for all three modes and all four query sizes:
[order 4](remote-cpu-order-4-seed-3.json),
[order 8](remote-cpu-order-8-seed-3.json),
[order 12](remote-cpu-order-12-seed-3.json).
Those runs used Python 3.11.15 / PyTorch 2.11.0+cu130 on the two-EPYC host with
`CUDA_VISIBLE_DEVICES=""` and `--device cpu`, so they performed no CUDA work.
Performance runs
will start only after the owner supplies an available GPU. There is no idle-card
poller or recurring GPU job.
