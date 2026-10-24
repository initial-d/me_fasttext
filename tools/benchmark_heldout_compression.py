"""Train-only SGNS; validation-selected native compression, held-out test audit.

Measures unseen-context SGNS objective and sampled context discrimination, not
perplexity, semantic benchmarks, or trained Engram quality. Exact character
n-grams (including multiplicities) plus word rows form the full input embedding.
Word/output rows remain unchanged. No retraining after approximation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .train_ngram_similarity import prepare
    from .benchmark_cosine_merge import structural_merge, fidelity
    from .native_row_store import NativeRows, pack_native
except ImportError:
    from train_ngram_similarity import prepare
    from benchmark_cosine_merge import structural_merge, fidelity
    from native_row_store import NativeRows, pack_native


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def split_tokens(text):
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    a, b = int(len(tokens) * 0.8), int(len(tokens) * 0.9)
    if not 0 < a < b < len(tokens):
        raise ValueError("corpus too short for contiguous 80/10/10 token split")
    return {"train": tokens[:a], "validation": tokens[a:b], "test": tokens[b:]}


def context_pairs(tokens, words):
    """Keep original positions, OOV gaps and split boundaries intact."""
    lookup = {word: i for i, word in enumerate(words)}
    seq = [lookup.get(word, -1) for word in tokens]
    rows = [(seq[i], seq[j], i) for i in range(len(seq)) if seq[i] >= 0
            for j in range(max(0, i - 2), min(len(seq), i + 3)) if i != j and seq[j] >= 0]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def evaluation_set(tokens, words, counts, limit, negatives, seed):
    pool = context_pairs(tokens, words)
    if not len(pool):
        raise ValueError("held-out split has no in-vocabulary context pairs")
    rng = np.random.default_rng(seed)
    sample = pool[rng.choice(len(pool), min(limit, len(pool)), replace=False)]
    probability = np.asarray([counts[w] ** 0.5 for w in words], dtype=np.float64)
    probability /= probability.sum()
    negative = rng.choice(len(words), (len(sample), negatives), p=probability)
    arrays = {"pairs": sample[:, :2].copy(), "positions": sample[:, 2].copy(), "negative": negative}
    arrays["sha256"] = digest(b"".join(arrays[k].astype("<i8").tobytes()
                                     for k in ("pairs", "positions", "negative")))
    arrays["pool_pairs"] = len(pool)
    vocabulary = set(words)
    arrays["token_coverage"] = sum(t in vocabulary for t in tokens) / len(tokens)
    arrays["negative_target_collision_fraction"] = float((negative == sample[:, 1:2]).mean())
    return arrays


def compose(input_rows, bags, lengths):
    """The same full forward embedding as training, including word rows."""
    with torch.no_grad():
        return F.embedding_bag(bags, input_rows, mode="sum", padding_idx=0) / lengths[:, None]


def score(hidden, outputs, evaluation, batch_size=1024):
    losses, wins = [], []
    with torch.no_grad():
        for start in range(0, len(evaluation["pairs"]), batch_size):
            pairs = torch.from_numpy(evaluation["pairs"][start:start + batch_size])
            negative = torch.from_numpy(evaluation["negative"][start:start + batch_size])
            h = hidden[pairs[:, 0]]
            positive = (h * outputs[pairs[:, 1]]).sum(1)
            sampled = (h[:, None] * outputs[negative]).sum(2)
            loss = -F.logsigmoid(positive) - F.logsigmoid(-sampled).sum(1)
            win = ((positive[:, None] > sampled).float()
                   + 0.5 * (positive[:, None] == sampled).float()).mean(1)
            losses.append(loss.numpy().astype(np.float64))
            wins.append(win.numpy().astype(np.float64))
    return {"loss": np.concatenate(losses), "discrimination": np.concatenate(wins)}


def summarize(scores):
    return {"pairs": len(scores["loss"]), "sgns_loss_per_pair": float(scores["loss"].mean()),
            "context_vs_negative_discrimination": float(scores["discrimination"].mean())}


def deltas(candidate, baseline):
    return {"loss_relative_change": float(candidate["loss"].mean() / baseline["loss"].mean() - 1),
            "discrimination_change": float(candidate["discrimination"].mean()
                                             - baseline["discrimination"].mean())}


def choose_variant(variants, max_loss_increase, max_discrimination_drop):
    """Selection reads validation metrics only. Lossless is the feasible fallback."""
    eligible = [v for v in variants if v["validation_delta"]["loss_relative_change"] <= max_loss_increase
                and v["validation_delta"]["discrimination_change"] >= -max_discrimination_drop]
    if not eligible:
        raise ValueError("no feasible compression variant")
    return max(eligible, key=lambda v: v["ngram_raw_to_artifact_factor"])


def block_bootstrap(candidate, baseline, positions, seed=42, repeats=1000, block_tokens=128):
    """Paired token-block resampling, avoiding independent-context-pair CIs."""
    groups, inverse = np.unique(positions // block_tokens, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    totals = {prefix + metric: np.bincount(inverse, weights=scores[metric])
              for prefix, scores in (("a", candidate), ("b", baseline))
              for metric in ("loss", "discrimination")}
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(repeats):
        ids = rng.integers(0, len(groups), len(groups))
        changes.append([totals["aloss"][ids].sum() / totals["bloss"][ids].sum() - 1,
                        (totals["adiscrimination"][ids].sum() - totals["bdiscrimination"][ids].sum())
                        / count[ids].sum()])
    changes = np.asarray(changes)
    return {"block_tokens": block_tokens, "observed_blocks": len(groups), "repeats": repeats,
            "seed": seed, "loss_relative_change_95_percent_interval": np.quantile(changes[:, 0], [.025, .975]).tolist(),
            "discrimination_change_95_percent_interval": np.quantile(changes[:, 1], [.025, .975]).tolist(),
            "limitation": "Within one text split; blocks are a correlation approximation, not independent corpora."}


def train_model(words, grams, pairs, bags, lengths, counts, cfg, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    inputs = nn.EmbeddingBag(1 + len(words) + len(grams), cfg.dim, mode="sum", sparse=True, padding_idx=0)
    outputs = nn.Embedding(len(words), cfg.dim, sparse=True)
    with torch.no_grad():
        inputs.weight.uniform_(-1 / cfg.dim, 1 / cfg.dim)
        inputs.weight[0].zero_()
        outputs.weight.zero_()
    optimizer = torch.optim.SGD(list(inputs.parameters()) + list(outputs.parameters()), lr=cfg.lr)
    probability = torch.tensor([counts[w] ** 0.5 for w in words], dtype=torch.float32)
    probability /= probability.sum()
    samples, start = [], time.perf_counter()
    for step in range(cfg.steps):
        batch = pairs[rng.integers(len(pairs), size=cfg.batch_size)]
        center, target = torch.from_numpy(batch[:, 0].copy()), torch.from_numpy(batch[:, 1].copy())
        summed = inputs(bags[center])
        averaged = summed / lengths[center, None]
        # Preserve the legacy average-forward/unscaled-input-gradient convention.
        hidden = summed + (averaged - summed).detach()
        negative = torch.multinomial(probability, cfg.batch_size * cfg.negatives, replacement=True)
        sampled = outputs(negative.reshape(cfg.batch_size, cfg.negatives))
        positive = (hidden * outputs(target)).sum(1)
        logits = (hidden[:, None] * sampled).sum(2)
        loss = (-F.logsigmoid(positive) - F.logsigmoid(-logits).sum(1)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == cfg.steps - 1:
            samples.append({"step": step, "sgns_loss_per_pair": float(loss.detach()) / cfg.batch_size})
    return inputs.weight.detach().clone(), outputs.weight.detach().clone(), {
        "seconds": time.perf_counter() - start, "loss_samples": samples,
        "sampled_pairs": cfg.steps * cfg.batch_size, "training_pool_pairs": len(pairs),
        "training_pool_sha256": digest(pairs.astype("<i8").tobytes())}


def native_variant(rows, grams, cfg, directory, name, threshold=None):
    mapping = (np.arange(len(rows)) if threshold is None else
               structural_merge(rows, grams, threshold, cfg.min_norm_ratio))
    reconstructed = rows[mapping].astype("<f4")
    directory = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    raw, artifact = directory / "rows.f32", directory / "rows.men"
    expected = reconstructed.tobytes()
    raw.write_bytes(expected)
    packed = pack_native(raw, artifact, rows.shape[1] * 4, 64, predictor="xor")
    with NativeRows(artifact, cfg.native_library) as native:
        actual = native.rows(range(len(rows)))
    if actual != expected:
        raise AssertionError("native row reconstruction is not byte exact")
    manifest = directory / "MANIFEST.json"
    manifest.write_text(json.dumps({"approximate": threshold is not None, "source_sha256": digest(rows.tobytes()),
                                    "reconstructed_sha256": digest(actual), "mapping_sha256": digest(mapping.astype("<i8").tobytes()),
                                    "minimum_cosine": threshold, "minimum_norm_ratio": cfg.min_norm_ratio if threshold else None,
                                    "native_library_sha256": digest(cfg.native_library.read_bytes()),
                                    "rows": len(rows), "dim": rows.shape[1], "dtype": "little-endian float32"}, indent=2) + "\n")
    total = packed["artifact_bytes"] + manifest.stat().st_size
    return torch.from_numpy(np.frombuffer(actual, dtype="<f4").reshape(rows.shape).copy()), {
        "name": name, "approximate": threshold is not None, "minimum_cosine": threshold,
        "minimum_norm_ratio": cfg.min_norm_ratio if threshold else None,
        "retained_representatives": len(np.unique(mapping)), "storage": packed,
        "manifest_bytes": manifest.stat().st_size, "total_ngram_artifact_bytes": total,
        "ngram_raw_to_artifact_factor": rows.nbytes / total,
        "native_artifact_sha256": digest(artifact.read_bytes()), "native_rows_byte_exact_to_reconstruction": True,
        "row_fidelity": fidelity(rows, reconstructed),
        "source_sha256": digest(rows.tobytes()), "mapping_sha256": digest(mapping.astype("<i8").tobytes())}


def run(split, cfg, order, seed, thresholds):
    words, grams, _, pairs, bags, lengths, counts = prepare(" ".join(split["train"]), order, cfg.vocab_limit, cfg.min_count)
    if len(words) < 2 or not len(pairs):
        raise ValueError("training vocabulary/context pool too small")
    evaluation = {name: evaluation_set(split[name], words, counts, cfg.eval_pairs, cfg.negatives,
                                       cfg.eval_seed + (name == "test")) for name in ("validation", "test")}
    inputs, outputs, training = train_model(words, grams, pairs, bags, lengths, counts, cfg, seed)
    if not torch.isfinite(inputs).all() or not torch.isfinite(outputs).all():
        raise ValueError("nonfinite trained weights; preserve failure, lower learning rate")
    directory = cfg.output_dir / f"order-{order}-seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"input_rows": inputs, "output_rows": outputs, "words": words, "grams": grams,
                "bags": bags, "lengths": lengths, "order": order, "seed": seed}, directory / "reference.pt")
    for name, data in evaluation.items():
        np.savez(directory / f"{name}-evaluation.npz", **{k: data[k] for k in ("pairs", "negative", "positions")})
    offset = 1 + len(words)
    rows = inputs[offset:].numpy().astype("<f4")
    original_hidden = compose(inputs, bags, lengths)
    baseline = {name: score(original_hidden, outputs, data) for name, data in evaluation.items()}
    unchanged = (inputs[:offset].numel() + outputs.numel()) * 4
    original_payload = inputs.numel() * 4 + outputs.numel() * 4
    entry = {"max_order": order, "seed": seed, "vocabulary": len(words), "ngram_rows": len(grams),
             "training": training, "full_input_sha256": digest(inputs.numpy().tobytes()),
             "output_sha256": digest(outputs.numpy().tobytes()),
             "word_ngram_layout_sha256": digest(json.dumps([words, grams], ensure_ascii=True).encode()),
             "checkpoint_sha256": digest((directory / "reference.pt").read_bytes()),
             "ngram_raw_bytes": rows.nbytes, "unchanged_word_padding_output_bytes": unchanged,
             "original_full_vector_payload_bytes": original_payload,
             "baseline": {name: summarize(data) for name, data in baseline.items()},
             "evaluation": {name: {k: v for k, v in data.items() if k not in ("pairs", "positions", "negative")}
                            for name, data in evaluation.items()}, "variants": []}
    for name, threshold in [("lossless", None)] + [(f"cos-{tau:g}-norm-{cfg.min_norm_ratio:g}", tau) for tau in thresholds]:
        decoded, variant = native_variant(rows, grams, cfg, directory, name, threshold)
        restored = torch.cat([inputs[:offset], decoded])
        hidden = compose(restored, bags, lengths)
        scores = score(hidden, outputs, evaluation["validation"])
        variant.update({"validation": summarize(scores), "validation_delta": deltas(scores, baseline["validation"]),
                        "full_vector_payload_to_compressed_factor": original_payload /
                        (unchanged + variant["total_ngram_artifact_bytes"]),
                        "full_compressed_vector_payload_bytes": unchanged + variant["total_ngram_artifact_bytes"]})
        if threshold is None:
            test_scores = score(hidden, outputs, evaluation["test"])
            exact = torch.equal(hidden, original_hidden) and all(np.array_equal(scores[k], baseline["validation"][k])
                                                                 for k in scores)
            exact &= all(np.array_equal(test_scores[k], baseline["test"][k]) for k in test_scores)
            if not exact:
                raise AssertionError("lossless storage changed full embeddings or held-out outputs")
            variant["full_embedding_and_both_split_outputs_bit_identical"] = True
        entry["variants"].append(variant)
    selected = choose_variant(entry["variants"], cfg.max_loss_increase, cfg.max_discrimination_drop)
    # Test only the validation-chosen candidate; no test-set threshold tuning.
    with NativeRows(directory / selected["name"] / "rows.men", cfg.native_library) as native:
        decoded = torch.from_numpy(np.frombuffer(native.rows(range(len(rows))), dtype="<f4").reshape(rows.shape).copy())
    selected_scores = score(compose(torch.cat([inputs[:offset], decoded]), bags, lengths), outputs, evaluation["test"])
    entry["selection"] = {"name": selected["name"], "validation_max_loss_relative_increase": cfg.max_loss_increase,
                          "validation_max_discrimination_drop": cfg.max_discrimination_drop,
                          "rule": "largest ngram artifact compression among validation-feasible variants; no test metrics used",
                          "test": summarize(selected_scores), "test_delta": deltas(selected_scores, baseline["test"]),
                          "test_paired_token_block_bootstrap": block_bootstrap(selected_scores, baseline["test"],
                                                                               evaluation["test"]["positions"], cfg.eval_seed)}
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-source", required=True, help="source URL/provenance; no network request is made")
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-orders", default="4,8,12")
    parser.add_argument("--seeds", default="3,7,11")
    parser.add_argument("--thresholds", default="0.99,0.98,0.97,0.95,0.90")
    parser.add_argument("--min-norm-ratio", type=float, default=.99)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--vocab-limit", type=int, default=1200)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--lr", type=float, default=.05)
    parser.add_argument("--eval-pairs", type=int, default=20000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--max-loss-increase", type=float, default=.005)
    parser.add_argument("--max-discrimination-drop", type=float, default=.005)
    args = parser.parse_args()
    orders, seeds = [int(v) for v in args.max_orders.split(",")], [int(v) for v in args.seeds.split(",")]
    thresholds = [float(v) for v in args.thresholds.split(",")]
    if (min(args.dim, args.vocab_limit, args.min_count, args.steps, args.batch_size,
            args.negatives, args.eval_pairs) < 1 or min(orders) < 2 or
            not 0 < args.min_norm_ratio <= 1 or any(not 0 < t <= 1 for t in thresholds) or
            not 0 < args.lr or args.max_loss_increase < 0 or args.max_discrimination_drop < 0):
        parser.error("invalid positive training/evaluation dimensions or thresholds/budgets")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = args.corpus.read_bytes()
    split = split_tokens(corpus.decode("utf-8"))
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"category": "public-text trained SGNS reference, native compression and held-out objective",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
              "pytorch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
              "cpu": cpu, "logical_cpus": os.cpu_count(), "device": "cpu", "gpu": "not used",
              "threads": 1, "interop_threads": 1, "corpus_source": args.corpus_source,
              "corpus_sha256": digest(corpus), "corpus_bytes": len(corpus),
              "native_library_sha256": digest(args.native_library.read_bytes()),
              "split": {name: {"tokens": len(tokens), "sha256": digest("\n".join(tokens).encode())}
                        for name, tokens in split.items()},
              "settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
              "cost_assumptions": "local CPU, cached public text, no paid service; no latency/energy/RSS claim",
              "training_protocol": "train-only vocabulary/counts, +/-2 original-token context; exact char-ngrams and word row; average forward, unscaled input gradient; summed sparse SGD, unigram exponent 0.5 negatives",
              "limitations": ["One small English corpus, contiguous split, in-vocabulary context pairs only; coverage is reported.",
                              "Reference SGNS, not original C++ reproduction or trained Engram weights; no downstream LM quality/perplexity claim.",
                              "Finite training and fixed settings; no convergence claim. Negative samples can include the positive target.",
                              "Native decode feeds full averaged input embeddings with unchanged word/output rows; no training through mapped storage.",
                              "Ngram ratio includes native metadata/manifest; full vector payload ratio excludes key index, tokenizer and model architecture.",
                              "Approximate variants change original vectors; lossless native storage is bit exact to its input bytes.",
                              "Cosine/norm acceptance is direct to immutable structural representatives; no monotonicity or global optimum guarantee.",
                              "Corpus license/provenance has not been independently established; corpus and weights are not redistributed."],
              "runs": [], "failures": []}
    start = time.perf_counter()
    for order in orders:
        for seed in seeds:
            try:
                entry = run(split, args, order, seed, thresholds)
                report["runs"].append(entry)
                print(json.dumps({"order": order, "seed": seed, "selection": entry["selection"]}), flush=True)
            except Exception as error:
                report["failures"].append({"order": order, "seed": seed, "type": type(error).__name__, "error": str(error)})
                print(json.dumps(report["failures"][-1]), flush=True)
            report["total_seconds"] = time.perf_counter() - start
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit("failures preserved in report.json")


if __name__ == "__main__":
    main()
