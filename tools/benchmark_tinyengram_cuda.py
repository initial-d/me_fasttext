"""Same-schedule trained Qwen CUDA prefill/decode, one storage mode per process.

All modes use the same host hashing and layer-prefetch wrapper. Resident changes
only row placement; dense offload shares native staging. Small trained memory
does not establish large-table compression or deployment throughput.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer

from .benchmark_device_gather import DenseStagedBaseline, gpu_snapshot, monitor_client_pid, rss
from .device_row_adapter import GatherTicket
from .engram_row_adapter import file_sha256
from .prepare_device_gather import tensor_sha256
from .tinyengram_checkpoint import load_tinyengram_checkpoint, source_revision
from .verify_tinyengram_shared_cuda import _fingerprint


class ResidentPrefetchedBaseline(DenseStagedBaseline):
    """Benchmark-only resident rows with identical host hash/prefetch ordering."""
    def __init__(self, fixture, **kwargs):
        super().__init__(fixture, **kwargs)
        self.weight = self.weight.to(self.device)

    @torch.no_grad()
    def prefetch(self, input_ids):
        ids = self.gather._global_ids(input_ids).reshape(-1).to(self.device)
        output = self.weight.index_select(0, ids).reshape(*input_ids.shape, self.embedding_dim)
        self._stats["queries"] += 1
        # Production wrapper consumes on this same default CUDA stream.
        return GatherTicket(output)


def load_mode(args):
    model = load_tinyengram_checkpoint(args.bundle, upstream_dir=args.upstream_dir,
        native_library=args.native_library, device="cuda", staging_rows=args.staging_rows,
        staging_slots=2, native_cache_bytes=args.cache_bytes)
    if args.mode == "native":
        return model
    manifest = json.loads((args.bundle / "bundle.json").read_text())
    with safe_open(args.checkpoint / "model.safetensors", framework="pt", device="cpu") as source:
        for layer_id, item in manifest["layers"].items():
            fixture = args.output_dir / (args.mode + "-dense-memory-" + layer_id)
            fixture.mkdir()
            (fixture / "native").symlink_to((args.bundle / item["path"]).resolve(), target_is_directory=True)
            torch.save(source.get_tensor(item["weight_key"]), fixture / "dense.pt")
            layer = model.model.model.engram_layers[layer_id]
            original = layer.multi_head_embedding
            cls = ResidentPrefetchedBaseline if args.mode == "resident" else DenseStagedBaseline
            replacement = cls(fixture, device="cuda", native_library=args.native_library,
                              staging_rows=args.staging_rows, staging_slots=2, native_cache_bytes=0)
            # Preserve TinyEngramModel's context-ticket hook; host layer/hash
            # ordering is identical. Direct layer calls are not benchmarked.
            replacement.forward = original.forward
            layer.multi_head_embedding = replacement
            original.close()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "bundle", "corpus", "upstream-dir", "native-library", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("resident", "dense-offload", "native"), required=True)
    parser.add_argument("--tokens", default="32,128")
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--staging-rows", type=int, default=256)
    parser.add_argument("--cache-bytes", type=int, default=1048576)
    args = parser.parse_args()
    counts = [int(v) for v in args.tokens.split(",")]
    if min(*counts, args.decode_tokens, args.repeat, args.staging_rows) < 1 or min(args.warmup, args.cache_bytes) < 0:
        parser.error("positive token counts/repeat/staging and nonnegative warmup/cache required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (args.mode + ".json")
    if path.exists():
        parser.error("previous report exists; choose a fresh output directory")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    report = {"status": "failed", "mode": args.mode, "commit": source_revision(),
              "command": shlex.join(sys.orig_argv), "inputs": _fingerprint(args.checkpoint, args.bundle, args.corpus),
              "source_hashes": {name: file_sha256(Path(__file__).parent / name) for name in
                                ("benchmark_tinyengram_cuda.py", "benchmark_device_gather.py",
                                 "device_row_adapter.py", "tinyengram_checkpoint.py")},
              "settings": {"tokens": counts, "decode_tokens": args.decode_tokens, "warmup": args.warmup,
                           "repeat": args.repeat, "staging_rows": args.staging_rows, "cache_bytes": args.cache_bytes},
              "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                           "cuda_runtime": torch.version.cuda}, "cases": [],
              "limitations": ["One small trained Qwen3-0.6B branch: 648x8 float32 memory, not large-table memory savings.",
                              "Eager unpadded batch-one inference, fixed-length greedy decode; no deployment traffic or quality benchmark.",
                              "Same host hash/prefetch schedule for all modes; resident is a controlled baseline, not upstream GPU hashing.",
                              "Warm wall-clock measurements include host work and per-token greedy ID readback; prefill excludes logits D2H hashing.",
                              "Driver/context memory excluded from Torch allocator counters; RSS includes startup/framework state.",
                              "Sampled empty-device gate and competing-client checks do not reserve the GPU or exclude shared CPU/I/O interference.",
                              "Cached local corpus/checkpoints; no downloads, retraining or paid services."]}
    model = None
    try:
        selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not selector or "," in selector:
            raise ValueError("expose exactly one selected GPU")
        uuid = subprocess.check_output(["nvidia-smi", "--id=" + selector, "--query-gpu=uuid",
                                       "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
        report["gpu_before"] = gpu_snapshot(uuid, own_pid=0)
        if report["gpu_before"]["compute_pids"] or report["gpu_before"]["utilization_percent"] > 5:
            report.update(status="blocked", error="GPU contended before context creation")
            return 2
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        runtime_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if (runtime_uuid if runtime_uuid.startswith("GPU-") else "GPU-" + runtime_uuid) != uuid:
            raise ValueError("CUDA device differs from monitored GPU")
        free, total = torch.cuda.mem_get_info()
        cap = int(total * .04)
        report.update(allocator_limit_bytes=cap, reserve_bytes=2 * 1024**3)
        if free < cap + 2 * 1024**3:
            report.update(status="blocked", error="insufficient free memory for capped model plus reserve")
            return 2
        torch.cuda.set_per_process_memory_fraction(.04)
        probe = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        monitor_pid = monitor_client_pid(gpu_snapshot(uuid, own_pid=0))
        report.update(monitor_client_pid=monitor_pid, worker_namespace_pid=os.getpid())
        del probe
        tokenizer = AutoTokenizer.from_pretrained(args.bundle / "tokenizer", local_files_only=True, trust_remote_code=False)
        text = args.corpus.read_text()[:16384]
        segment = text[int(len(text) * .8):int(len(text) * .9)]
        ids = tokenizer(segment, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] < max(counts):
            raise ValueError("insufficient chronological validation tokens")
        reference = None
        if args.mode != "resident":
            reference = json.loads((args.output_dir / "resident.json").read_text())
            if (reference["status"] != "passed" or reference["inputs"] != report["inputs"]
                    or reference["settings"] != report["settings"] or reference["source_hashes"] != report["source_hashes"]
                    or reference["software"] != report["software"] or reference["gpu_before"]["gpu_uuid"] != uuid):
                raise ValueError("resident reference provenance/settings differ")
        started = time.perf_counter()
        model = load_mode(args)
        torch.cuda.synchronize()
        report["open_seconds"] = time.perf_counter() - started
        report["rss_after_open"] = rss()
        with torch.inference_mode():
            for count in counts:
                sample = ids[:, :count].contiguous()
                for method in ("prefill", "greedy-cached", "greedy-uncached"):
                    def forward():
                        return (model(sample, use_cache=False).logits if method == "prefill" else
                                model.generate(sample, max_new_tokens=args.decode_tokens, use_cache=method == "greedy-cached"))
                    output = forward()
                    torch.cuda.synchronize()
                    digest = tensor_sha256(output)
                    generated = output.tolist() if method != "prefill" else None
                    del output
                    key = f"{count}/{method}"
                    if reference is not None:
                        oracle = next(c for c in reference["cases"] if c["key"] == key)
                        if oracle["output_sha256"] != digest or oracle["input_ids_sha256"] != tensor_sha256(sample):
                            raise AssertionError("storage baseline changed same-schedule output: " + key)
                    for _ in range(args.warmup):
                        output = forward()
                        torch.cuda.synchronize()
                        del output
                    before = gpu_snapshot(uuid, own_pid=monitor_pid)
                    if before["external_compute_pids"] or monitor_pid not in before["compute_pids"]:
                        report.update(status="blocked", error="competition before timing", gpu_during=before)
                        return 2
                    torch.cuda.reset_peak_memory_stats()
                    times = []
                    for _ in range(args.repeat):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        output = forward()
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - started)
                        del output
                    after = gpu_snapshot(uuid, own_pid=monitor_pid)
                    if after["external_compute_pids"] or monitor_pid not in after["compute_pids"]:
                        report.update(status="blocked", error="competition during timing", gpu_during=after)
                        return 2
                    report["cases"].append({"key": key, "input_tokens": count, "method": method,
                        "input_ids_sha256": tensor_sha256(sample), "output_sha256": digest, "generated_ids": generated,
                        "same_schedule_byte_exact": True if reference is not None else None, "seconds": times,
                        "median_seconds": statistics.median(times), "p95_seconds": float(np.percentile(times, 95)),
                        "tokens_per_second": (count if method == "prefill" else args.decode_tokens) / statistics.median(times),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "rss": rss()})
        report["status"] = "passed"
        return 0
    except Exception as error:
        report.update(status="failed", error=repr(error))
        return 1
    finally:
        if model is not None:
            model.close()
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("Report:", path.resolve(), report["status"], flush=True)


if __name__ == "__main__":
    sys.exit(main())
