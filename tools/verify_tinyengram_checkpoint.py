"""Replay existing TinyEngram/Qwen checkpoint + native bundle without retraining.

Measure full and same-schedule cached equality separately. Also run a memory-off
control for the backbone's own full-vs-incremental floating-point difference.
No weights/text are fetched and no backend is selected using held-out metrics.
"""
import argparse
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import torch
import transformers
from transformers import AutoTokenizer

from .engram_row_adapter import file_sha256
from .tinyengram_checkpoint import TinyEngramModel, load_tinyengram_checkpoint, load_upstream
from .train_tinyengram_reference import _tensor_hash, cached_logits, logit_delta, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--max-characters", type=int, default=16384)
    parser.add_argument("--eval-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.report_out.exists():
        parser.error("use a fresh report path to preserve earlier evidence")
    report = {"command": shlex.join([sys.executable, "-m", "tools.verify_tinyengram_checkpoint", *sys.argv[1:]]),
              "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_file_hashes": {name: file_sha256(Path(__file__).parent / name) for name in
                                     ("tinyengram_checkpoint.py", "train_tinyengram_reference.py", "verify_tinyengram_checkpoint.py")},
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "transformers": transformers.__version__, "os": platform.platform(),
                              "machine": platform.machine(), "device": "cpu", "threads": args.threads},
              "corpus_sha256": file_sha256(args.corpus),
              "bundle_manifest_sha256": file_sha256(args.bundle / "bundle.json"),
              "limitations": ["small trained branch on pretrained Qwen3; not convergence or upstream reproduction",
                              "only sampled validation/test tokens and fixed-length greedy decode",
                              "CPU eager inference; no CUDA offload, latency/RSS or compression-benefit measurement",
                              "full-vs-cached arithmetic can round differently; same-schedule equality isolates storage"]}
    start = time.perf_counter()
    try:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        engram, qwen = load_upstream(args.upstream_dir)
        config_values = json.loads((args.checkpoint / "engram_config.json").read_text())
        config_values["tokenizer_name_or_path"] = str(args.checkpoint.resolve())
        config = engram.EngramConfig(**config_values)
        step = json.loads((args.checkpoint / "trainer_state.json").read_text())["global_step"]
        engram.set_global_step(step)
        model, loading = qwen.EngramQwenForCausalLM.from_pretrained(
            args.checkpoint, engram_cfg=config, local_files_only=True,
            attn_implementation="eager", output_loading_info=True,
        )
        if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
            raise AssertionError(f"resident checkpoint was not completely restored: {loading}")
        report["resident_checkpoint_loading"] = loading
        model.requires_grad_(False).eval()
        resident = TinyEngramModel(model).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        restored_tokenizer = AutoTokenizer.from_pretrained(args.bundle / "tokenizer", local_files_only=True,
                                                          trust_remote_code=False)
        text = args.corpus.read_text()[:args.max_characters]
        a, b = int(len(text) * .8), int(len(text) * .9)
        segments = {"train": text[:a], "validation": text[a:b], "test": text[b:]}
        ids = {name: tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
               for name, text in segments.items()}
        for name, text in segments.items():
            if not torch.equal(ids[name], restored_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids):
                raise AssertionError("restored tokenizer changed token IDs")
        report["tokenizer_roundtrip_ids_exact"] = True
        if args.training_report:
            training = json.loads(args.training_report.read_text())
            if training["corpus_sha256"] != report["corpus_sha256"]:
                raise AssertionError("training corpus fingerprint changed")
            for name in segments:
                if training["splits"][name]["token_ids_sha256"] != _tensor_hash(ids[name]):
                    raise AssertionError("training/evaluation split token IDs changed")
            report["training_report_sha256"] = file_sha256(args.training_report)
            report["training"] = training["training"]
        if min(ids[name].shape[1] for name in ("validation", "test")) < args.eval_tokens:
            raise ValueError("insufficient held-out tokens")
        with load_tinyengram_checkpoint(args.bundle, upstream_dir=args.upstream_dir,
                                       native_library=args.native_library, native_cache_bytes=1024**2) as mapped:
            report["all_memory_rows_byte_exact"] = True
            for layer_id, layer in model.model.engram_layers.items():
                weight = layer.multi_head_embedding.embedding.weight.detach()
                if mapped.model.model.engram_layers[layer_id].multi_head_embedding.table.rows(range(len(weight))) != weight.contiguous().view(torch.uint8).numpy().tobytes():
                    raise AssertionError("memory row bytes changed")
            report["verification"] = {name: verify(resident, mapped, ids[name][:, :args.eval_tokens], args.decode_tokens)
                                      for name in ("validation", "test")}
        with torch.inference_mode():
            try:
                qwen.set_skip_engram(True)
                report["memory_off_backbone_rounding_control"] = {}
                for name in ("validation", "test"):
                    sample = ids[name][:, :args.eval_tokens]
                    full = model(sample, use_cache=False).logits
                    cached, _ = cached_logits(model, sample)
                    report["memory_off_backbone_rounding_control"][name] = logit_delta(cached, full, sample)
            finally:
                qwen.set_skip_engram(False)
        report["status"] = "passed"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("Report:", args.report_out.resolve(), flush=True)


if __name__ == "__main__":
    main()
