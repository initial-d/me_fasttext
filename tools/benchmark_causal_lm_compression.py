"""Full-softmax causal character LM: native storage, validation budgets, test.

Uses a trained GRU + exact character-suffix memory reference, not Engram weights.
All test characters are scored, including unknown-character targets. Approximate
variants are selected on validation PPL/KL/top-1 budgets; test is used once for
the chosen variant. Native gather remains in the model's actual forward path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

try:
    from .causal_ngram_lm import (CausalNgramLM, fit_registry, encode_segment,
                                 train_reference, evaluate, metric_summary, prediction_delta)
    from .benchmark_cosine_merge import structural_merge, fidelity
    from .engram_row_adapter import export_embedding, MappedEmbedding, file_sha256
except ImportError:
    from causal_ngram_lm import (CausalNgramLM, fit_registry, encode_segment,
                                train_reference, evaluate, metric_summary, prediction_delta)
    from benchmark_cosine_merge import structural_merge, fidelity
    from engram_row_adapter import export_embedding, MappedEmbedding, file_sha256


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def text_splits(text, max_characters):
    text = text[:max_characters] if max_characters else text
    a, b = int(len(text) * .8), int(len(text) * .9)
    if not 0 < a < b < len(text):
        raise ValueError("insufficient characters for contiguous 80/10/10 split")
    return {"train": text[:a], "validation": text[a:b], "test": text[b:]}


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_variant(variants, max_ppl_increase, max_kl, max_top1_flip):
    feasible = [v for v in variants
                if v["validation_delta"]["perplexity_relative_change"] <= max_ppl_increase
                and v["validation_delta"]["mean_prediction_kl_nats"] <= max_kl
                and v["validation_delta"]["top1_changed_fraction"] <= max_top1_flip]
    if not feasible:
        raise ValueError("no feasible variant; lossless storage must remain a candidate")
    return max(feasible, key=lambda v: v["memory_raw_to_artifact_factor"])


def paired_ppl_interval(candidate, reference, block_characters=256, repeats=1000, seed=42):
    if not np.array_equal(candidate["positions"], reference["positions"]):
        raise ValueError("bootstrap positions differ")
    _, inverse = np.unique(candidate["positions"] // block_characters, return_inverse=True)
    count = np.bincount(inverse)
    sums = np.bincount(inverse, weights=candidate["loss"] - reference["loss"])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        ids = rng.integers(0, len(count), len(count))
        draws.append(float(np.expm1(sums[ids].sum() / count[ids].sum())))
    return {"perplexity_relative_change_95_percent_interval": np.quantile(draws, [.025, .975]).tolist(),
            "block_characters": block_characters, "observed_blocks": len(count),
            "repeats": repeats, "seed": seed,
            "limitation": "Paired contiguous character blocks within one text split; not independent corpora."}


def pack_variant(weights, registry, cfg, directory, threshold=None):
    rows = weights[1:].numpy().astype("<f4")
    mapping = (np.arange(len(rows)) if threshold is None else
               structural_merge(rows, registry["grams"], threshold, cfg.min_norm_ratio))
    restored = torch.cat([weights[:1], weights[1:][torch.from_numpy(mapping)]])
    name = "lossless" if threshold is None else f"cos-{threshold:g}-norm-{cfg.min_norm_ratio:g}"
    output = directory / name
    # Ordinary Embedding -> native adapter; the actual forward queries this file.
    source = nn.Embedding.from_pretrained(restored, freeze=True, padding_idx=0)
    metadata = export_embedding(source, output, cfg.block_rows, backend="native", predictor="xor")
    manifest = output / "APPROXIMATION.json"
    manifest.write_text(json.dumps({"approximate": threshold is not None,
                                    "source_weight_sha256": sha(weights.numpy().tobytes()),
                                    "reconstructed_weight_sha256": sha(restored.numpy().tobytes()),
                                    "mapping_sha256": sha(mapping.astype("<i8").tobytes()),
                                    "minimum_cosine": threshold,
                                    "minimum_norm_ratio": cfg.min_norm_ratio if threshold is not None else None,
                                    "contract": "native bytes exact to reconstructed table; approximation changes original weights"}, indent=2) + "\n")
    total = sum((output / n).stat().st_size for n in ("table.men", "embedding.json", "APPROXIMATION.json"))
    with MappedEmbedding(output, cfg.native_library, cfg.cache_bytes) as mapped:
        actual = mapped.gather.table.rows(range(len(weights)))
    if actual != restored.numpy().tobytes():
        raise AssertionError("native gather changed reconstructed table bytes")
    return output, {"name": name, "approximate": threshold is not None, "minimum_cosine": threshold,
                    "minimum_norm_ratio": cfg.min_norm_ratio if threshold is not None else None,
                    "retained_representatives": len(np.unique(mapping)), "metadata": metadata,
                    "total_memory_artifact_bytes": total, "memory_raw_to_artifact_factor": weights.numel() * 4 / total,
                    "original_weight_sha256": sha(weights.numpy().tobytes()),
                    "mapping_sha256": sha(mapping.astype("<i8").tobytes()),
                    "all_native_rows_exact_to_reconstruction": True,
                    "row_fidelity": fidelity(rows, rows[mapping])}


def with_native_scores(model, output, segments, cfg, names):
    original_memory = model.memory
    with MappedEmbedding(output, cfg.native_library, cfg.cache_bytes) as mapped:
        try:
            model.memory = mapped
            return {name: evaluate(model, segments[name], cfg.burn_in, cfg.eval_chunk) for name in names}
        finally:
            model.memory = original_memory


def run(split, cfg, order, seed, thresholds):
    start = time.perf_counter()
    registry = fit_registry(split["train"], order, cfg.min_count)
    segments = {name: encode_segment(text, registry) for name, text in split.items()}
    torch.manual_seed(seed)
    model = CausalNgramLM(len(registry["characters"]) + 1, len(registry["grams"]), cfg.dim)
    initial_memory = model.memory.weight.detach().clone()
    initial_validation = metric_summary(evaluate(model, segments["validation"], cfg.burn_in, cfg.eval_chunk))
    training_start = time.perf_counter()
    history = train_reference(model, segments["train"], cfg.steps, cfg.batch_size, cfg.sequence_length,
                              cfg.dense_lr, cfg.memory_lr, seed)
    training_seconds = time.perf_counter() - training_start
    original_memory = model.memory.weight.detach().clone()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite trained parameters")
    directory = cfg.output_dir / f"order-{order}-seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "reference.pt"
    torch.save({"registry": registry, "model_state": model.state_dict(), "dim": cfg.dim}, checkpoint)
    registry_path = directory / "registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=True, sort_keys=True) + "\n")
    references = {name: evaluate(model, segments[name], cfg.burn_in, cfg.eval_chunk) for name in ("validation", "test")}
    ablated = {name: evaluate(model, segments[name], cfg.burn_in, cfg.eval_chunk, use_memory=False)
               for name in ("validation", "test")}
    raw_memory = original_memory.numel() * 4
    other_weights = sum(p.numel() * p.element_size() for name, p in model.named_parameters() if name != "memory.weight")
    registry_bytes = registry_path.stat().st_size
    entry = {"max_order": order, "seed": seed, "memory_rows": len(registry["grams"]),
             "vocabulary_size_including_unknown": len(registry["characters"]) + 1,
             "training_seconds": training_seconds, "training_loss_samples": history,
             "sampled_training_next_characters": cfg.steps * cfg.batch_size * cfg.sequence_length,
             "initial_validation": initial_validation,
             "trained_state_sha256": state_sha256(model), "checkpoint_sha256": file_sha256(checkpoint),
             "registry_sha256": file_sha256(registry_path), "registry_bytes": registry_bytes,
             "memory_raw_parameter_bytes": raw_memory, "unchanged_other_parameter_bytes": other_weights,
             "updated_memory_row_fraction": float((original_memory[1:] != initial_memory[1:]).any(-1).float().mean()),
             "baseline": {name: metric_summary(scores) for name, scores in references.items()},
             "inference_memory_ablation": {name: {"metrics": metric_summary(scores),
                                                    "delta": prediction_delta(scores, references[name])}
                                           for name, scores in ablated.items()},
             "evaluation": {name: {"input_ids_sha256": sha(segment["characters"].numpy().astype("<i8").tobytes()),
                                   "memory_ids_sha256": sha(segment["memory_ids"].numpy().astype("<i8").tobytes()),
                                   "unknown_characters": segment["unknown_characters"],
                                   "known_memory_slot_fraction": segment["known_memory_slot_fraction"],
                                   "positions_with_memory_fraction": segment["positions_with_memory_fraction"]}
                            for name, segment in segments.items()}, "variants": []}
    for name, scores in references.items():
        np.savez(directory / f"{name}-reference.npz", logits=scores["logits"].numpy(),
                 targets=scores["targets"].numpy(), positions=scores["positions"])
    for threshold in [None, *thresholds]:
        output, variant = pack_variant(original_memory, registry, cfg, directory, threshold)
        names = ("validation", "test") if threshold is None else ("validation",)
        native_scores = with_native_scores(model, output, segments, cfg, names)
        variant["validation"] = metric_summary(native_scores["validation"])
        variant["validation_delta"] = prediction_delta(native_scores["validation"], references["validation"])
        total_parameters = other_weights + variant["total_memory_artifact_bytes"]
        variant.update({"full_parameter_payload_to_compressed_factor": (raw_memory + other_weights) / total_parameters,
                        "parameter_and_registry_to_compressed_factor": (raw_memory + other_weights + registry_bytes)
                        / (total_parameters + registry_bytes)})
        if threshold is None:
            exact = all(torch.equal(native_scores[n]["logits"], references[n]["logits"])
                        and np.array_equal(native_scores[n]["loss"], references[n]["loss"]) for n in names)
            if not exact:
                raise AssertionError("lossless native model changed logits/CE on a held-out split")
            variant["both_split_logits_and_cross_entropy_bit_identical"] = True
        entry["variants"].append(variant)
    selected = select_variant(entry["variants"], cfg.max_ppl_increase, cfg.max_kl, cfg.max_top1_flip)
    tested = with_native_scores(model, directory / selected["name"], segments, cfg, ("test",))["test"]
    entry["selection"] = {"name": selected["name"], "validation_ppl_budget": cfg.max_ppl_increase,
                          "validation_mean_kl_budget": cfg.max_kl, "validation_top1_flip_budget": cfg.max_top1_flip,
                          "rule": "largest memory compression within validation budgets; no test selection",
                          "test": metric_summary(tested), "test_delta": prediction_delta(tested, references["test"]),
                          "test_paired_block_interval": paired_ppl_interval(tested, references["test"], seed=cfg.eval_seed)}
    entry["total_seconds"] = time.perf_counter() - start
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-source", required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-characters", type=int, default=131072, help="fixed corpus prefix; 0 means all text")
    parser.add_argument("--max-orders", default="4,8,12")
    parser.add_argument("--seeds", default="3,7,11")
    parser.add_argument("--thresholds", default="0.99,0.98,0.97,0.95,0.90")
    parser.add_argument("--min-norm-ratio", type=float, default=.99)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--dense-lr", type=float, default=.002)
    parser.add_argument("--memory-lr", type=float, default=.01)
    parser.add_argument("--burn-in", type=int, default=64)
    parser.add_argument("--eval-chunk", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--block-rows", type=int, default=64)
    parser.add_argument("--cache-bytes", type=int, default=4194304)
    parser.add_argument("--max-ppl-increase", type=float, default=.005)
    parser.add_argument("--max-kl", type=float, default=.0001)
    parser.add_argument("--max-top1-flip", type=float, default=.01)
    cfg = parser.parse_args()
    orders, seeds = [int(v) for v in cfg.max_orders.split(",")], [int(v) for v in cfg.seeds.split(",")]
    thresholds = [float(v) for v in cfg.thresholds.split(",")]
    if (min(cfg.min_count, cfg.dim, cfg.steps, cfg.batch_size, cfg.sequence_length, cfg.eval_chunk, cfg.block_rows) < 1
            or min(orders) < 2 or cfg.burn_in < max(orders) - 1 or cfg.max_characters < 0 or cfg.cache_bytes < 0
            or not 0 < cfg.min_norm_ratio <= 1 or any(not 0 < t <= 1 for t in thresholds)
            or min(cfg.dense_lr, cfg.memory_lr) <= 0 or min(cfg.max_ppl_increase, cfg.max_kl, cfg.max_top1_flip) < 0):
        parser.error("invalid dimensions, rates, budgets or burn-in (must cover every maximum order)")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    corpus = cfg.corpus.read_bytes()
    split = text_splits(corpus.decode("utf-8"), cfg.max_characters)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"category": "public-text trained causal character LM reference; native parameter compression",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version, "pytorch": torch.__version__,
              "numpy": np.__version__, "platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
              "device": "cpu", "gpu": "not used", "threads": 1, "interop_threads": 1,
              "corpus_source": cfg.corpus_source, "corpus_sha256": sha(corpus), "corpus_bytes": len(corpus),
              "used_text_sha256": sha("".join(split.values()).encode()),
              "native_library_sha256": file_sha256(cfg.native_library),
              "split": {name: {"characters": len(text), "sha256": sha(text.encode())} for name, text in split.items()},
              "settings": {k: v for k, v in vars(cfg).items() if not isinstance(v, Path)},
              "architecture": "character embedding + mean exact suffix ngram memory -> one-layer GRU -> full character-vocabulary softmax",
              "training_protocol": "train-only character/key registry; Adam dense weights, SparseAdam memory, standard averaged-memory gradients; random causal chunks, no retraining after compression",
              "cost_assumptions": "cached public text and local CPU; no paid services, energy/RSS/latency claims",
              "limitations": ["Small character GRU reference, not trained Engram, a Transformer or original paper reproduction.",
                              "Fixed corpus prefix and contiguous split; character PPL is specific to this alphabet/unknown mapping.",
                              "All targets after common burn-in are included; unseen characters map to explicit unknown ID.",
                              "Repeated exact keys only (training min_count); missing validation/test keys use zero memory.",
                              "Inference memory ablation is not a separately trained parameter-matched baseline.",
                              "Parameter-plus-registry artifact bytes exclude runtime lookup dictionaries, framework, tokenizer code and resident memory.",
                              "Approximation changes original weights; native gather is lossless only to reconstructed bytes.",
                              "Finite training, one corpus and three seeds; no convergence or general LM-quality guarantee.",
                              "Corpus licensing not independently established; text, checkpoints and key strings remain local."],
              "runs": [], "failures": []}
    start = time.perf_counter()
    for order in orders:
        for seed in seeds:
            try:
                result = run(split, cfg, order, seed, thresholds)
                report["runs"].append(result)
                print(json.dumps({"order": order, "seed": seed, "selection": result["selection"]}), flush=True)
            except Exception as error:
                report["failures"].append({"order": order, "seed": seed, "type": type(error).__name__, "error": str(error)})
                print(json.dumps(report["failures"][-1]), flush=True)
            report["total_seconds"] = time.perf_counter() - start
            (cfg.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit("failed runs preserved in report.json")


if __name__ == "__main__":
    main()
