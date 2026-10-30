# Native memory in a trained TinyEngram/Qwen3 checkpoint

The new converter/loader works on a complete trained Qwen3 checkpoint, rather
than replacing a table in the official standalone mock backbone. The run uses
pretrained **Qwen3-0.6B** revision `c1899de289a04d12100db370d81485cdf75e47ca`
and TinyEngram source `317f23832aa8f8e382a0916ec469e81ba12b6dff`.
Only the inserted memory/gates/ShortConv branch is trained: 78,912 of 596,128,832
parameters, 32 Adam steps on 32-token sequences. **636 of 648 memory rows changed**,
and convolution weights became nonzero. This small finite run tests checkpoint
integration; it does not establish convergence, biomedical quality, trained
DeepSeek compatibility, compression benefit, or serving throughput.

The corpus is the locally cached first 16,384 TinyShakespeare characters, split
contiguously 80/10/10 before tokenization. Each held-out check scores only the
first **32 tokens / 31 next-token targets** of its split. Source tokenizer IDs
survive checkpoint and native-bundle serialization exactly on all three text
segments. Corpus and weights stay local; they are not distributed here.

| Check | Validation slice | Test slice |
|---|---:|---:|
| All native memory row bytes | Exact (all 648 rows) | Same table |
| Whole-model full-forward logits vs resident table | Byte-exact | Byte-exact |
| Native vs resident cached logits, same execution schedule | Byte-exact | Byte-exact |
| Greedy continuation, cached/native vs uncached/resident | Same 8 tokens | Same 8 tokens |
| Corrected cached vs full max absolute logit difference | 5.25e-5 | 6.68e-5 |
| Corrected cached vs full top-1 changes | 0 / 32 | 0 / 32 |
| Original history-free upstream cached vs full max difference | 2.64 | 2.06 |
| Original history-free upstream cached vs full top-1 changes | 4 / 32 | 5 / 32 |

The wrapper retains the last two raw tokens and six per-layer gated values with
each request's KV cache. Both n-gram hashing and the dilated ShortConv need this
history. Direct raw-upstream model calls bypass that request context, giving the
original incremental control without modifying upstream hashing/convolution.
Turning memory off leaves the backbone's own full-vs-cached differences of
5.64e-5 / 5.34e-5, with no top-1 changes. Thus byte preservation is checked under
matched execution schedules; different matrix shapes have a separate rounding
measurement. These slice-level results are not a general generation guarantee.

The first run failed a full-vs-cached `rtol=2e-4, atol=2e-5` comparison, despite
exact row/full-forward checks. Its [report](training-and-initial-failure.json)
and [log](training.log) remain unchanged. The follow-up added resident cached,
history-free upstream, and memory-off backbone controls; it did not retrain,
reselect weights, change source rows, or silently relax that assertion.
[Verification](verification.json) records the successful matched-schedule checks.

## Install and use

```bash
python -m pip install 'meft-memory[tinyengram] @ git+https://github.com/initial-d/me_fasttext.git@v0.2.0'
meft-tiny-export --checkpoint /path/to/trained-checkpoint \
  --engram-config /path/to/engram_config.json \
  --upstream-dir /path/to/pinned-TinyEngram --output-dir /path/to/frozen-bundle
```

Source installation needs a C compiler and zlib headers. The platform wheels
include the standalone C reader and require no manual native-library path.
No old FastText binaries or bundled static libraries are included. This release
targets tested Linux/macOS hosts, not Windows or universal manylinux compatibility.

```python
from transformers import AutoTokenizer
from meft_memory import load_tinyengram_checkpoint

tokenizer = AutoTokenizer.from_pretrained("/path/to/frozen-bundle/tokenizer", local_files_only=True)
ids = tokenizer("A short prompt", return_tensors="pt").input_ids
with load_tinyengram_checkpoint("/path/to/frozen-bundle", upstream_dir="/path/to/pinned-TinyEngram") as model:
    output = model.generate(ids, max_new_tokens=8)
    print(tokenizer.decode(output[0]))
```

Use `python -m tools.tinyengram_checkpoint` from the source checkout for the same
converter. `meft-tiny-verify` replays an existing dense checkpoint and bundle;
`meft-tiny-train-reference` reproduces the small training run with local weights
and corpus paths. Upstream files `engram.py` and `engram_qwen.py` are hash-checked;
Transformers 4.57.3 is required. Export full single/indexed safetensors checkpoints;
adapter-only or pickle checkpoints are not supported. Supply the original Engram
config because the upstream Trainer does not embed it in the ordinary Qwen config.

Models load on CPU/eager attention, with unpadded batches. Greedy generation is
fixed length: no EOS stopping, sampling, beams, custom positions or HF generation
API. Row lookup is immutable; all other model parameters are frozen on load.
Metadata-only meta construction avoids allocating the original dense memory
tables, while backbone weights still occupy CPU memory. Full-file checksum scans
add startup I/O. There is no CUDA offload, prefetch, RSS or speedup measurement.

## Reproducible evidence

- [Source-run verification](verification.json) and [log](verification.log):
  exact commands, source commit/file hashes, split fingerprints, output hashes,
  all controls and full results. Runtime source commit: `fcbc6766647869c5decad02c2e12dce9ef2c905b`.
- [Installed-wheel verification](wheel-verification.json) and
  [log](wheel-verification.log): real Qwen generation and the same controls run
  outside the checkout, using the packaged native library and CLI.
- [Wheel build](wheel-build.txt): exact filename and SHA256 of the local build.
- [Local suite](local-tests.txt): 84 tests passed. CI additionally builds and
  installs Linux/macOS wheels outside the checkout without Torch, and checks
  complete Qwen checkpoint/decoder behavior on tiny trained fixtures.

Environment: Apple M5, macOS 26.6.2 arm64, Python 3.12.14, PyTorch 2.14.1,
Transformers 4.57.3; MPS branch training, CPU inference, four CPU threads/one
interop thread. No rented compute, paid API or new corpus download. Timings in
the reports are run duration, not a serving benchmark. Monetary/energy cost is
unmeasured. The existing FastText paper's results are separate from this follow-up.
