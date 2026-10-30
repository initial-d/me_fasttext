"""Sequential CUDA checkpoint validation for GPUs shared with other jobs.

Resident and native phases run in separate processes; CPU reference logits link
the phases. Checks provenance and bytes, never benchmarks contended throughput.
Preserves previous/failed reports and leaves a configured free-memory reserve.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import time

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .engram_row_adapter import file_sha256
from .tinyengram_checkpoint import (
    TinyEngramModel, load_tinyengram_checkpoint, load_upstream, source_revision,
)
from .train_tinyengram_reference import _tensor_hash, cached_logits


def _fingerprint(checkpoint, bundle, corpus):
    checkpoint_files = sorted(checkpoint.glob("*.safetensors"))
    if not checkpoint_files:
        raise ValueError("checkpoint requires local safetensors weights")
    return {"checkpoint_files": {p.name: file_sha256(p) for p in checkpoint_files},
            "config": file_sha256(checkpoint / "config.json"),
            "engram_config": file_sha256(checkpoint / "engram_config.json"),
            "bundle": file_sha256(bundle / "bundle.json"), "corpus": file_sha256(corpus)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("resident", "native"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--eval-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--max-characters", type=int, default=16384)
    parser.add_argument("--staging-rows", type=int, default=32)
    parser.add_argument("--staging-slots", type=int, default=2)
    parser.add_argument("--allocator-fraction", type=float, default=0.04)
    parser.add_argument("--reserve-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if (not 0 < args.allocator_fraction <= 1 or args.reserve_bytes < 0 or args.eval_tokens < 2
            or args.decode_tokens < 0 or min(args.max_characters, args.staging_rows, args.staging_slots) < 1):
        parser.error("invalid allocation/reserve/evaluation settings")
    report_path = args.output_dir / (args.phase + ".json")
    if report_path.exists():
        parser.error("use a fresh output directory; previous reports are preserved")
    references_path = args.output_dir / "resident-logits.safetensors"
    if args.phase == "resident" and references_path.exists():
        parser.error("resident reference already exists")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    report = {"phase": args.phase, "command": shlex.join(sys.orig_argv), "code_sha": source_revision(),
              "source_hashes": {name: file_sha256(Path(__file__).parent / name) for name in
                                ("verify_tinyengram_shared_cuda.py", "train_tinyengram_reference.py",
                                 "tinyengram_checkpoint.py", "device_row_adapter.py")},
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
                              "os": platform.platform(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "settings": {"allocator_fraction": args.allocator_fraction, "reserve_bytes": args.reserve_bytes,
                           "eval_tokens": args.eval_tokens, "decode_tokens": args.decode_tokens,
                           "max_characters": args.max_characters, "staging_rows": args.staging_rows,
                           "staging_slots": args.staging_slots},
              "limitations": ["shared GPU; not a latency/throughput benchmark",
                              "small trained memory branch, not large-table memory savings or model-quality evidence",
                              "allocator cap excludes CUDA context/library allocations; reserve is a preflight check",
                              "other jobs can change memory use after preflight; no exclusive resource reservation",
                              "same-device same-schedule equality; cross-device/full-vs-cached arithmetic can differ"]}
    started = time.perf_counter()
    model = None
    try:
        if not torch.cuda.is_available():
            report.update(status="blocked", error="CUDA unavailable")
            return 2
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        free, total = torch.cuda.mem_get_info()
        limit = int(total * args.allocator_fraction)
        report["environment"].update(gpu=torch.cuda.get_device_name(), total_bytes=total,
                                     free_after_context_init=free, allocator_limit_bytes=limit)
        if free < limit + args.reserve_bytes:
            report.update(status="blocked", error="insufficient free memory for allocator limit plus reserve")
            return 2
        torch.cuda.set_per_process_memory_fraction(args.allocator_fraction)
        report["inputs"] = _fingerprint(args.checkpoint, args.bundle, args.corpus)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        restored = AutoTokenizer.from_pretrained(args.bundle / "tokenizer", local_files_only=True, trust_remote_code=False)
        text = args.corpus.read_text()[:args.max_characters]
        a, b = int(len(text) * .8), int(len(text) * .9)
        segments = {"validation": text[a:b], "test": text[b:]}
        ids = {k: tokenizer(v, return_tensors="pt", add_special_tokens=False).input_ids[:, :args.eval_tokens]
               for k, v in segments.items()}
        for name, segment in segments.items():
            source = tokenizer(segment, return_tensors="pt", add_special_tokens=False).input_ids
            if source.shape[1] < args.eval_tokens:
                raise ValueError("insufficient held-out tokens")
            if not torch.equal(source, restored(segment, return_tensors="pt", add_special_tokens=False).input_ids):
                raise AssertionError("bundle tokenizer changed IDs")
        reference, tensors = None, {}
        if args.phase == "resident":
            engram, qwen = load_upstream(args.upstream_dir)
            values = json.loads((args.checkpoint / "engram_config.json").read_text())
            values["tokenizer_name_or_path"] = str(args.checkpoint.resolve())
            config = engram.EngramConfig(**values)
            step = json.loads((args.bundle / "bundle.json").read_text())["global_step"]
            engram.set_global_step(step)
            original, loading = qwen.EngramQwenForCausalLM.from_pretrained(
                args.checkpoint, engram_cfg=config, local_files_only=True, attn_implementation="eager",
                output_loading_info=True,
            )
            if any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise AssertionError(f"incomplete resident loading: {loading}")
            model = TinyEngramModel(original.requires_grad_(False).eval().to("cuda")).eval()
        else:
            reference = json.loads((args.output_dir / "resident.json").read_text())
            if (reference["status"] != "passed" or reference["inputs"] != report["inputs"]
                    or reference["settings"] != report["settings"]
                    or reference["source_hashes"] != report["source_hashes"]
                    or reference["environment"] != report["environment"]):
                # Free memory is expected to change on a shared device. Compare
                # stable environment fields separately from that observation.
                stable = ("python", "torch", "transformers", "cuda_runtime", "gpu", "total_bytes", "visible_devices")
                if (reference["status"] != "passed" or reference["inputs"] != report["inputs"]
                        or reference["settings"] != report["settings"]
                        or reference["source_hashes"] != report["source_hashes"]
                        or any(reference["environment"].get(k) != report["environment"].get(k) for k in stable)):
                    raise ValueError("resident reference provenance/settings/device differ")
            if reference["reference_sha256"] != file_sha256(references_path):
                raise ValueError("resident reference checksum changed")
            tensors = load_file(references_path, device="cpu")
            model = load_tinyengram_checkpoint(
                args.bundle, upstream_dir=args.upstream_dir, native_library=args.native_library, device="cuda",
                staging_rows=args.staging_rows, staging_slots=args.staging_slots, native_cache_bytes=1024**2,
            )
        report["verification"] = {}
        with torch.inference_mode():
            for name, sample in ids.items():
                full = model(sample, use_cache=False).logits.cpu().contiguous()
                cached, cache = cached_logits(model, sample)
                cached = cached.cpu().contiguous()
                generated = {str(use_cache): model.generate(sample[:, :8], max_new_tokens=args.decode_tokens,
                                                           use_cache=use_cache).tolist() for use_cache in (True, False)}
                item = {"tokens": sample.shape[1], "token_ids_sha256": _tensor_hash(sample),
                        "full_logits_sha256": _tensor_hash(full), "cached_logits_sha256": _tensor_hash(cached),
                        "generated_ids": generated,
                        "cached_vs_full_max_abs": (cached - full).abs().max().item(),
                        "cached_vs_full_top1_changed": int((cached.argmax(-1) != full.argmax(-1)).sum()),
                        "history_devices": {"raw_tokens": str(cache.token_tail.device),
                                            "kv": str(cache.kv.layers[0].keys.device),
                                            "conv": {k: str(v.device) for k, v in cache.conv_tails.items()}}}
                if reference is None:
                    tensors[name + ".full"], tensors[name + ".cached"] = full, cached
                else:
                    baseline = reference["verification"][name]
                    item["full_logits_byte_exact"] = torch.equal(full.view(torch.uint8), tensors[name + ".full"].view(torch.uint8))
                    item["cached_logits_byte_exact"] = torch.equal(cached.view(torch.uint8), tensors[name + ".cached"].view(torch.uint8))
                    item["greedy_tokens_equal"] = generated == baseline["generated_ids"]
                    report["verification"][name] = item
                    if (item["token_ids_sha256"] != baseline["token_ids_sha256"]
                            or not all(item[k] for k in ("full_logits_byte_exact", "cached_logits_byte_exact", "greedy_tokens_equal"))):
                        raise AssertionError("native CUDA differs from resident same-schedule reference: " + name)
                report["verification"][name] = item
        if reference is None:
            save_file(tensors, references_path)
            report["reference_sha256"] = file_sha256(references_path)
        else:
            report["resident_report_sha256"] = file_sha256(args.output_dir / "resident.json")
            report["memory_stats"] = {k: v.multi_head_embedding.stats for k, v in model.model.model.engram_layers.items()}
        torch.cuda.synchronize()
        report["memory"] = {"max_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "free_after": torch.cuda.mem_get_info()[0]}
        report["status"] = "passed"
        return 0
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        cleanup_error = None
        if model is not None:
            try:
                model.close()
            except BaseException as error:
                cleanup_error = error
                report.update(status="failed", cleanup_error=repr(error))
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("Report:", report_path.resolve(), flush=True)
        if cleanup_error is not None:
            raise cleanup_error


if __name__ == "__main__":
    sys.exit(main())
