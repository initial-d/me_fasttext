"""Replay a local trained checkpoint with resident vs staged device memory.

No downloads/training. Uses original validation/test split tokenization, exact
same-device execution schedules, row bytes and fixed greedy continuations.
CUDA absence writes a blocked report and exits 2. MPS is a synchronous transfer
reference; its results do not validate CUDA streams, pinned memory or speed.
"""
import argparse
import json
from pathlib import Path
import platform
import shlex
import sys
import time

import torch
import transformers
from transformers import AutoTokenizer

from .device_row_adapter import _device
from .engram_row_adapter import file_sha256
from .tinyengram_checkpoint import (
    SOURCE_HASHES, UPSTREAM_SHA, TinyEngramModel, load_tinyengram_checkpoint, load_upstream, source_revision,
)
from .train_tinyengram_reference import _tensor_hash, cached_logits


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def _verify(resident, mapped, ids, decode_tokens, device):
    _sync(device)
    start = time.perf_counter()
    reference = resident(ids, use_cache=False).logits.cpu()
    _sync(device)
    resident_seconds = time.perf_counter() - start
    start = time.perf_counter()
    packed = mapped(ids, use_cache=False).logits.cpu()
    _sync(device)
    staged_seconds = time.perf_counter() - start
    if _tensor_hash(reference) != _tensor_hash(packed):
        raise AssertionError("staged full-forward logits changed")
    cached, cache = cached_logits(mapped, ids)
    resident_cached, _ = cached_logits(resident, ids)
    cached, resident_cached = cached.cpu(), resident_cached.cpu()
    if _tensor_hash(cached) != _tensor_hash(resident_cached):
        raise AssertionError("staged/resident same-schedule cached logits changed")
    prompt = ids[:, :8]
    a = resident.generate(prompt, max_new_tokens=decode_tokens, use_cache=True)
    b = mapped.generate(prompt, max_new_tokens=decode_tokens, use_cache=True)
    c = mapped.generate(prompt, max_new_tokens=decode_tokens, use_cache=False)
    if not (torch.equal(a, b) and torch.equal(a, c)):
        raise AssertionError("greedy tokens changed")
    return {"full_forward_logits_byte_exact": True, "logits_sha256": _tensor_hash(reference),
            "cached_logits_byte_exact": True, "cached_logits_sha256": _tensor_hash(cached),
            "cached_vs_full_max_abs": (cached - reference).abs().max().item(),
            "cached_vs_full_top1_changed": int((cached.argmax(-1) != reference.argmax(-1)).sum()),
            "greedy_tokens_equal": True, "generated_ids": a.tolist(),
            "token_ids_sha256": _tensor_hash(ids), "tokens": ids.shape[1],
            "history_devices": {"raw_tokens": str(cache.token_tail.device),
                                "convolution": {k: str(v.device) for k, v in cache.conv_tails.items()},
                                "kv": str(cache.kv.layers[0].keys.device)},
            "single_cold_full_forward_seconds": {"resident": resident_seconds, "staged": staged_seconds}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--max-characters", type=int, default=16384)
    parser.add_argument("--eval-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--staging-rows", type=int, default=65536)
    parser.add_argument("--staging-slots", type=int, default=2)
    parser.add_argument("--native-cache-bytes", type=int, default=1024**2)
    args = parser.parse_args()
    if args.report_out.exists():
        parser.error("use a fresh report path to preserve earlier evidence")
    if min(args.max_characters, args.eval_tokens, args.threads, args.staging_rows, args.staging_slots) <= 0:
        parser.error("sizes and threads must be positive")
    if args.eval_tokens < 2 or args.decode_tokens < 0 or args.native_cache_bytes < 0:
        parser.error("require eval-tokens >= 2, decode-tokens/cache-bytes >= 0")
    report = {"command": shlex.join(sys.orig_argv), "code_sha": source_revision(),
              "upstream_sha": UPSTREAM_SHA, "upstream_source_hashes": SOURCE_HASHES,
              "source_file_hashes": {name: file_sha256(Path(__file__).parent / name) for name in
                                     ("device_row_adapter.py", "engram_row_adapter.py", "tinyengram_checkpoint.py",
                                      "train_tinyengram_reference.py", "verify_tinyengram_device.py")},
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "transformers": transformers.__version__, "os": platform.platform(),
                              "machine": platform.machine(), "cpu": platform.processor(),
                              "device": args.device, "threads": args.threads,
                              "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda},
              "settings": {"staging_rows": args.staging_rows, "staging_slots": args.staging_slots,
                           "native_cache_bytes_per_layer": args.native_cache_bytes},
              "limitations": ["small trained branch integration; not convergence or quality superiority",
                              "sampled held-out tokens; no trained DeepSeek checkpoint",
                              "one cold latency sample includes output readback, no speedup conclusion",
                              "resident/staged models coexist; no comparative peak memory benchmark",
                              "MPS copies synchronous; cannot validate CUDA pinned/event behavior",
                              "CPU hashing/decompression is synchronous; all layer outputs live for this forward",
                              "fixed unpadded greedy decode; no distributed/graph/serving integration"]}
    start = time.perf_counter()
    try:
        try:
            device = _device(args.device)
        except RuntimeError as error:
            report.update(status="blocked", error=str(error))
            return 2
        if device.type == "cuda":
            gpu = torch.cuda.get_device_properties(device)
            report["environment"].update(gpu=gpu.name, gpu_total_bytes=gpu.total_memory)
        report["corpus_sha256"] = file_sha256(args.corpus)
        report["bundle_manifest_sha256"] = file_sha256(args.bundle / "bundle.json")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        engram, qwen = load_upstream(args.upstream_dir)
        values = json.loads((args.checkpoint / "engram_config.json").read_text())
        values["tokenizer_name_or_path"] = str(args.checkpoint.resolve())
        engram_config = engram.EngramConfig(**values)
        manifest = json.loads((args.bundle / "bundle.json").read_text())
        engram.set_global_step(manifest["global_step"])
        model, loading = qwen.EngramQwenForCausalLM.from_pretrained(
            args.checkpoint, engram_cfg=engram_config, local_files_only=True,
            attn_implementation="eager", output_loading_info=True,
        )
        if any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise AssertionError(f"incomplete resident checkpoint: {loading}")
        report["resident_loading"] = loading
        model.requires_grad_(False).eval().to(device)
        resident = TinyEngramModel(model).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        restored = AutoTokenizer.from_pretrained(args.bundle / "tokenizer", local_files_only=True, trust_remote_code=False)
        text = args.corpus.read_text()[:args.max_characters]
        a, b = int(len(text) * .8), int(len(text) * .9)
        segments = {"train": text[:a], "validation": text[a:b], "test": text[b:]}
        ids = {k: tokenizer(v, return_tensors="pt", add_special_tokens=False).input_ids for k, v in segments.items()}
        for key, segment in segments.items():
            if not torch.equal(ids[key], restored(segment, return_tensors="pt", add_special_tokens=False).input_ids):
                raise AssertionError("restored tokenizer changed IDs")
        report["splits"] = {k: {"characters": len(segments[k]), "tokens": v.shape[1],
                                "token_ids_sha256": _tensor_hash(v)} for k, v in ids.items()}
        if min(ids[k].shape[1] for k in ("validation", "test")) < args.eval_tokens:
            raise ValueError("insufficient held-out tokens")
        with load_tinyengram_checkpoint(
            args.bundle, upstream_dir=args.upstream_dir, native_library=args.native_library,
            device=device, prefetch_memory=True, staging_rows=args.staging_rows, staging_slots=args.staging_slots,
            native_cache_bytes=args.native_cache_bytes,
        ) as mapped:
            for layer_id, layer in model.model.engram_layers.items():
                weight = layer.multi_head_embedding.embedding.weight.detach()
                table = mapped.model.model.engram_layers[layer_id].multi_head_embedding.gather.table
                for start_row in range(0, len(weight), 256):
                    end = min(start_row + 256, len(weight))
                    if table.rows(range(start_row, end)) != weight[start_row:end].cpu().contiguous().view(torch.uint8).numpy().tobytes():
                        raise AssertionError("original memory row bytes changed")
            report["all_memory_rows_byte_exact"] = True
            report["verification"] = {k: _verify(resident, mapped, ids[k][:, :args.eval_tokens],
                                                 args.decode_tokens, device) for k in ("validation", "test")}
            report["memory_layers"] = {k: {"stats": v.multi_head_embedding.stats,
                                           "head_sizes": v.multi_head_embedding.gather.metadata["head_sizes"],
                                           "table_dtype": str(v.multi_head_embedding.dtype),
                                           "device": str(v.multi_head_embedding.device)}
                                       for k, v in mapped.model.model.engram_layers.items()}
        report["status"] = "passed"
        return 0
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("Report:", args.report_out.resolve(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
