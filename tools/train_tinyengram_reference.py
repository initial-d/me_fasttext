"""Train a small Engram branch on a local pretrained Qwen3, convert, and verify.

This is a trained-checkpoint integration experiment, not a reproduction of the
upstream biomedical study, a convergence claim, or a memory/speed benchmark.
No downloads are performed; provide local base-model/tokenizer and text paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer
import transformers

from .engram_row_adapter import file_sha256
from .tinyengram_checkpoint import (
    SOURCE_HASHES, UPSTREAM_SHA, TinyEngramModel,
    export_tinyengram_checkpoint, load_tinyengram_checkpoint, load_upstream,
)


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _metrics(logits, ids):
    losses = torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1),
                                               ids[:, 1:].flatten(), reduction="none")
    loss = losses.mean().item()
    return {"next_token_cross_entropy": loss, "perplexity": float(torch.exp(losses.mean())),
            "scored_tokens": losses.numel(), "logits_sha256": _tensor_hash(logits)}


@torch.inference_mode()
def cached_logits(model, ids):
    cache, pieces = None, []
    sizes = [min(8, ids.shape[1]), *([1] * max(0, ids.shape[1] - 8))]
    start = 0
    for size in sizes:
        output = model(ids[:, start:start + size], past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        pieces.append(output.logits)
        start += size
    return torch.cat(pieces, dim=1), cache


def logit_delta(candidate, reference, ids):
    delta = candidate - reference
    return {"max_abs": delta.abs().max().item(), "mean_abs": delta.abs().mean().item(),
            "top1_changed_fraction": (candidate.argmax(-1) != reference.argmax(-1)).float().mean().item(),
            "cross_entropy_change": _metrics(candidate, ids)["next_token_cross_entropy"]
                                    - _metrics(reference, ids)["next_token_cross_entropy"]}


@torch.inference_mode()
def verify(resident, mapped, ids, decode_tokens):
    reference = resident(ids, use_cache=False).logits
    packed = mapped(ids, use_cache=False).logits
    if _tensor_hash(reference) != _tensor_hash(packed):
        raise AssertionError("mapped full-forward logits changed")
    cached, cache = cached_logits(mapped, ids)
    resident_cached, _ = cached_logits(resident, ids)
    # Match execution schedules to isolate the storage replacement. Full and
    # incremental matrix shapes differ; report that separate rounding delta.
    if _tensor_hash(cached) != _tensor_hash(resident_cached):
        raise AssertionError("mapped and resident same-schedule cached logits changed")
    prompt = ids[:, :8]
    a = resident.generate(prompt, max_new_tokens=decode_tokens, use_cache=False)
    b = mapped.generate(prompt, max_new_tokens=decode_tokens, use_cache=True)
    c = mapped.generate(prompt, max_new_tokens=decode_tokens, use_cache=False)
    if not (torch.equal(a, b) and torch.equal(a, c)):
        raise AssertionError("greedy tokens changed")
    return {"full_forward_logits_byte_exact": True, "full_forward": _metrics(reference, ids),
            "cached_mapped_vs_resident_logits_byte_exact": True,
            "cached_vs_full_forward_delta": logit_delta(cached, reference, ids),
            "generated_token_ids": a.tolist(), "greedy_cached_and_uncached_tokens_equal": True,
            "decode_tokens": decode_tokens, "memory_history_tokens": cache.token_tail.shape[1],
            "convolution_history_shapes": {key: list(value.shape) for key, value in cache.conv_tails.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--eval-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--max-characters", type=int, default=16384)
    parser.add_argument("--train-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    if min(args.train_steps, args.sequence_length, args.eval_tokens, args.max_characters, args.threads) < 1:
        parser.error("training/sequence/evaluation sizes and threads must be positive")
    if args.output_dir.exists():
        parser.error("use a fresh output directory")
    if not args.base_model.is_dir():
        parser.error("--base-model must be a local model snapshot")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    report = {"evidence": "small trained Qwen3 + TinyEngram checkpoint integration; not throughput/quality superiority",
              "command": shlex.join([sys.executable, "-m", "tools.train_tinyengram_reference", *sys.argv[1:]]),
              "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_file_hashes": {name: file_sha256(Path(__file__).parent / name) for name in
                                     ("tinyengram_checkpoint.py", "train_tinyengram_reference.py")},
              "upstream_sha": UPSTREAM_SHA, "upstream_source_hashes": SOURCE_HASHES,
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "transformers": transformers.__version__, "os": platform.platform(),
                              "machine": platform.machine(), "train_device": args.train_device,
                              "inference_device": "cpu", "threads": args.threads},
              "base_model": str(args.base_model.resolve()), "corpus_sha256": file_sha256(args.corpus),
              "limitations": ["small finite branch training; not upstream biomedical reproduction",
                              "one seed/corpus; no convergence or generalization claim",
                              "CPU inference; no CUDA offload, latency/RSS or compression-benefit benchmark",
                              "fixed unpadded greedy generation; no beams/sampling/EOS stopping",
                              "weights and corpus remain local; no trained DeepSeek checkpoint"]}
    start = time.perf_counter()
    try:
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        engram, qwen = load_upstream(args.upstream_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False)
        text = args.corpus.read_text()[:args.max_characters]
        a, b = int(len(text) * .8), int(len(text) * .9)
        segments = {"train": text[:a], "validation": text[a:b], "test": text[b:]}
        encoded = {key: tokenizer(value, return_tensors="pt", add_special_tokens=False).input_ids
                   for key, value in segments.items()}
        if (encoded["train"].shape[1] <= args.sequence_length or
                min(encoded[key].shape[1] for key in ("validation", "test")) < args.eval_tokens):
            raise ValueError("text split has too few tokens")
        report["splits"] = {key: {"characters": len(segments[key]), "tokens": value.shape[1],
                                   "token_ids_sha256": _tensor_hash(value)} for key, value in encoded.items()}
        config = engram.EngramConfig(tokenizer_name_or_path=str(args.base_model.resolve()),
                                     engram_vocab_size=[257, 61], max_ngram_size=3,
                                     n_embed_per_ngram=16, n_head_per_ngram=2, layer_ids=[1],
                                     pad_id=tokenizer.pad_token_id, kernel_size=3, seed=args.seed)
        model = qwen.EngramQwenForCausalLM.from_pretrained(
            args.base_model, engram_cfg=config, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("engram_layers" in name)
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        initial = {name: _tensor_hash(value) for name, value in trainable.items()}
        initial_rows = {name: value.detach().cpu().clone() for name, value in trainable.items()
                        if "multi_head_embedding.embedding.weight" in name}
        report["training"] = {"seed": args.seed, "steps": args.train_steps, "sequence_length": args.sequence_length,
                              "optimizer": "Adam", "learning_rate": 0.001,
                              "total_parameters": sum(p.numel() for p in model.parameters()),
                              "trainable_parameters": sum(p.numel() for p in trainable.values()),
                              "backbone_frozen": True, "losses": []}
        model.to(args.train_device).train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
        for step in range(args.train_steps):
            position = (step * args.sequence_length) % (encoded["train"].shape[1] - args.sequence_length)
            batch = encoded["train"][:, position:position + args.sequence_length].to(args.train_device)
            optimizer.zero_grad(set_to_none=True)
            engram.set_global_step(step)
            loss = model(batch, labels=batch, use_cache=False).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            value = loss.item()
            report["training"]["losses"].append(value)
            print(f"step {step + 1}/{args.train_steps}: loss={value:.6f}", flush=True)
        model.to("cpu").eval()
        engram.set_global_step(args.train_steps)
        changes = {name: _tensor_hash(value) != initial[name] for name, value in model.named_parameters() if name in initial}
        if not any(changed for name, changed in changes.items() if "embedding.weight" in name):
            raise AssertionError("no memory weights changed in training")
        if not any(changed for name, changed in changes.items() if "short_conv.conv.weight" in name):
            raise AssertionError("convolution remained untrained; decode-history check would be weak")
        report["training"]["parameter_changed"] = changes
        report["training"]["memory_rows"] = {
            name: {"total": len(value), "changed": int((value.detach().cpu() != initial_rows[name]).any(dim=1).sum())}
            for name, value in model.named_parameters() if name in initial_rows
        }
        checkpoint = args.output_dir / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        (checkpoint / "engram_config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": args.train_steps}) + "\n")
        bundle = args.output_dir / "bundle"
        manifest = export_tinyengram_checkpoint(checkpoint, bundle, upstream_dir=args.upstream_dir,
                                                block_rows=64, predictor="xor")
        report["bundle"] = {"manifest_sha256": file_sha256(bundle / "bundle.json"),
                            "layers": manifest["layers"], "file_count": len(manifest["files"])}
        resident = TinyEngramModel(model).eval()
        with load_tinyengram_checkpoint(bundle, upstream_dir=args.upstream_dir,
                                       native_library=args.native_library, native_cache_bytes=1024**2) as mapped:
            restored_tokenizer = AutoTokenizer.from_pretrained(bundle / "tokenizer", local_files_only=True,
                                                               trust_remote_code=False)
            for name, segment in segments.items():
                if not torch.equal(restored_tokenizer(segment, return_tensors="pt", add_special_tokens=False).input_ids,
                                   encoded[name]):
                    raise AssertionError("bundle tokenizer changed source token IDs")
            report["tokenizer_roundtrip_ids_exact"] = True
            report["all_memory_rows_byte_exact"] = True
            for layer_id, layer in mapped.model.model.engram_layers.items():
                weight = model.model.engram_layers[layer_id].multi_head_embedding.embedding.weight.detach()
                if layer.multi_head_embedding.table.rows(range(len(weight))) != weight.contiguous().view(torch.uint8).numpy().tobytes():
                    raise AssertionError("stored memory row bytes changed")
            report["verification"] = {
                name: verify(resident, mapped, encoded[name][:, :args.eval_tokens], args.decode_tokens)
                for name in ("validation", "test")
            }
        report["status"] = "passed"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("Report:", report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
