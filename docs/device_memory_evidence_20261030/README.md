# Trained Qwen checkpoint: CPU memory with MPS backbone and prefetch

Implementation commit: `1bef0944368bf51911cd385ff43eacec42993a63`.
This replays the [previously trained Qwen3-0.6B + small TinyEngram branch](../tinyengram_qwen_evidence_20261030/README.md)
without retraining, downloads, dtype conversion or artifact format changes.
The branch has 648 original memory rows and is an integration example, not a
large-table compression or model-quality study.

| Check | Source MPS | Installed wheel CPU |
|---|---|---|
| Every memory row byte-exact | Yes | Yes |
| Full logits vs same-device resident table | Byte-exact | Byte-exact |
| Same-schedule cached logits vs resident table | Byte-exact | Byte-exact |
| 8-token greedy continuation, cached and full | Identical | Identical |
| Raw-token history | CPU | CPU |
| KV / convolution history | MPS | CPU |
| Queries across both evaluation samples | 84 | 84 |
| Host transfer buffers allocated | 2 (2,048 bytes total) | 0 |
| Host buffer reuses | 102 | 0 |

Each validation/test sample contains 32 tokens, from the unchanged 80/10/10
character split of the first 16,384 cached TinyShakespeare characters. Check
the split token hashes, corpus and bundle fingerprints in the raw reports.
Backbone/gates run on MPS; the immutable memory table and original hash IDs stay
on CPU. Vector output transfers preserve all bytes. **MPS is synchronous and
does not verify CUDA streams, pinned buffers or transfer overlap.**

Full and cached **execution schedules** round differently: maximum logit
differences were 1.1158e-4 / 6.2943e-5 on MPS and 5.2452e-5 / 6.6757e-5 on CPU
(validation/test). No sampled top-1 IDs changed. This is separate from storage
replacement: matching schedules had byte-exact logits. CPU and MPS logits are
not asserted equal across devices.

Hardware: Apple M5, 32 GiB RAM, macOS 26.6.2 arm64, Python 3.12.14,
PyTorch 2.14.1, Transformers 4.57.3. Reports retain exact commands and source
hashes. Local suite: 93 tests, 3 CUDA tests skipped, no failures. The tiny model
fixtures use real Qwen classes and learned nonzero convolution weights; vector
bit-pattern fixtures check dtype/NaN/lifetime behavior, not compression gains.

The installed wheel was built from the implementation commit and tested from
`/tmp` with automatic packaged C-library lookup. Its report has no Git SHA
because the installed package is outside a checkout; source hashes identify
the implementation. Local wheel SHA256:
`ca2255f7607812cb7e43fc3e9c48b7a6d47b01c6e54fa740d886fa50b2704c2f`
(`meft_memory-0.3.0-py3-none-macosx_26_0_arm64.whl`). Release CI rebuilds have
their own checksums; this local fingerprint identifies the tested build.

## CUDA blocker

The explicit CUDA replay wrote `status: blocked` and exited **2** before model
loading. `torch.cuda.is_available()` was false; `torch.version.cuda` was null.
On 2026-10-30, `gh api repos/initial-d/me_fasttext/actions/runners` returned
`total_count: 0`, with no configured self-hosted GPU runner. The public CPU CI
cannot supply CUDA validation. No CUDA performance or functional pass is claimed.

## Raw evidence and reproduction

- [MPS trained-checkpoint report](mps.json) and [log](mps.txt).
- [Installed-wheel CPU report](installed-wheel-cpu.json) and [log](installed-wheel-cpu.txt).
- [CUDA blocked report](cuda-blocked.json) and [log](cuda-blocked.txt).
- [Local tests](local-tests.txt) and [wheel build](wheel-build.txt).
- [API, limits and exact verifier command](../device_memory_prefetch.md).

The verifier's cold latency samples include logit readback and asymmetric
initial warmup. They are diagnostic only. Both models coexist; no comparative
peak memory, throughput or large-table savings were measured. Prefetched outputs
for all memory layers live until the forward finishes. Buffer bounds exclude
those outputs, ID tensors, native caches, mmap/RSS and model/KV allocations.
Tokenizer regex warnings were retained; restored token IDs and hash lookup
fingerprints match the original artifacts without changing tokenization.
