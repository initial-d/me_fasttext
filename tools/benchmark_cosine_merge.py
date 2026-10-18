"""Measure approximate prefix/suffix row sharing on hashed trained n-gram rows.

Each original row is compared directly to an unchanged retained representative.
No transitive threshold assumption, retraining, or lossless-original claim.
Written MEFTNAT1 artifacts preserve the APPROXIMATED rows and original IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

try:
    from .native_row_store import NativeRows, pack_native
    from .train_ngram_similarity import ngrams, prepare
except ImportError:
    from native_row_store import NativeRows, pack_native
    from train_ngram_similarity import ngrams, prepare


def structural_merge(rows, grams, threshold, min_norm_ratio=0.0):
    """Shortest first; best qualifying retained proper prefix/suffix ancestor.

    Spelling only generates candidates; actual float64 cosine/norm tests decide.
    Zero-norm rows are kept, and representatives never change after retention.
    """
    if not -1 <= threshold <= 1 or not 0 <= min_norm_ratio <= 1:
        raise ValueError("invalid similarity/norm thresholds")
    if len(rows) != len(grams) or len(set(grams)) != len(grams) or not np.isfinite(rows).all():
        raise ValueError("invalid trained rows/keys")
    values = rows.astype(np.float64)
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.maximum(norms[:, None], np.finfo(np.float64).tiny)
    lookup = {g: i for i, g in enumerate(grams)}
    mapping = np.arange(len(rows), dtype=np.int64)
    retained = set()
    for row_id in sorted(range(len(rows)), key=lambda i: (len(grams[i]), grams[i])):
        gram = grams[row_id]
        ancestors = {lookup[x] for length in range(2, len(gram))
                     for x in (gram[:length], gram[-length:]) if x in lookup}
        candidates = sorted(ancestors & retained)
        if norms[row_id] > 0 and candidates:
            candidates = np.asarray(candidates, dtype=np.int64)
            cosine = normalized[candidates] @ normalized[row_id]
            ratios = np.minimum(norms[candidates], norms[row_id]) / np.maximum(norms[candidates], norms[row_id])
            eligible = (norms[candidates] > 0) & (cosine >= threshold) & (ratios >= min_norm_ratio)
            if eligible.any():
                positions = np.flatnonzero(eligible)
                score = ratios if threshold == -1 else cosine
                best = positions[np.argmax(score[eligible])]
                mapping[row_id] = candidates[best]
                continue
        retained.add(row_id)
    return mapping


def fidelity(original, reconstructed, weights=None):
    a, b = original.astype(np.float64), reconstructed.astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (norm_a > 0) & (norm_b > 0)
    relative = np.linalg.norm(b - a, axis=1) / np.maximum(norm_a, np.finfo(np.float64).tiny)
    cosine = np.ones(len(a))
    cosine[valid] = np.sum(a[valid] * b[valid], axis=1) / (norm_a[valid] * norm_b[valid])
    cosine[(norm_a > 0) & (norm_b == 0)] = 0
    weights = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=np.float64)
    return {"rows": len(a), "minimum_cosine": float(cosine.min()),
            "mean_cosine": float(cosine.mean()), "p05_cosine": float(np.quantile(cosine, 0.05)),
            "relative_l2_mean": float(relative.mean()), "relative_l2_p95": float(np.quantile(relative, 0.95)),
            "relative_l2_max": float(relative.max()),
            "weighted_relative_l2_mean": float(np.average(relative, weights=weights)),
            "weighted_relative_frobenius": float(np.sqrt(np.sum(weights[:, None] * (b - a) ** 2)
                                                          / np.sum(weights[:, None] * a ** 2))),
            "weighted_mean_cosine": float(np.average(cosine, weights=weights)),
            "max_relative_norm_change": float(np.max(np.abs(norm_b - norm_a) / np.maximum(norm_a, np.finfo(np.float64).tiny)))}


def compose_subwords(rows, words, grams, order):
    lookup = {g: i for i, g in enumerate(grams)}
    # Only the ngram component is available; original separate word rows are
    # not exported by the training experiment. Multiplicity remains intact.
    return np.stack([rows[[lookup[g] for g in ngrams(word, order)]].astype(np.float64).mean(axis=0)
                     for word in words])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.999,0.995,0.99,0.98,0.97,0.95,0.90")
    args = parser.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]
    if not thresholds or any(not 0 < x <= 1 for x in thresholds):
        parser.error("thresholds must lie in (0, 1]")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_bytes = (args.training_dir / "report.json").read_bytes()
    training = json.loads(training_bytes)
    corpus = args.corpus.read_bytes()
    if hashlib.sha256(corpus).hexdigest() != training["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    settings = training["settings"]
    report = {"category": "approximate structural cosine row merge on public-text trained SGNS reference",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "exact_command": " ".join([sys.executable, *sys.argv]),
              "training_commit": training["commit"], "training_report_sha256": hashlib.sha256(training_bytes).hexdigest(),
              "corpus_sha256": training["corpus_sha256"], "python": sys.version,
              "pytorch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
              "device": "CPU", "threads": 1,
              "native_library_sha256": hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
              "thresholds": thresholds, "block_rows": 64,
              "algorithm": "shortest-key-first, retained proper prefix/suffix ancestors, highest direct cosine (highest norm ratio for norm-only); no transitive chains",
              "norm_policies": {"cosine_only": 0.0, "cosine_and_norm_0_99": 0.99, "norm_only_0_99": 0.99},
              "limitations": ["Same small English reference as the lossless study, not original C++ or Engram weights.",
                              "Approximate sharing changes source bytes; native store is exact only for reconstructed rows.",
                              "Compression denominators cover ngram vector rows, not word/output rows, key index or full model.",
                              "Composed vectors cover only the ngram mean component, not the full word embedding.",
                              "No held-out task metric, retraining, logits, model-quality or RSS/cold-I/O claim.",
                              "Greedy structural candidate policy is a concrete experiment, not globally optimal clustering."], "runs": []}
    for source in training["runs"]:
        order, seed = source["max_order"], source["seed"]
        words, grams, signatures, _, _, _, counts = prepare(corpus.decode("utf-8"), order, settings["vocab_limit"], settings["min_count"])
        raw = (args.training_dir / f"order-{order}-seed-{seed}" / "trained-ngram.f32").read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["learned_row_sha256"]:
            raise ValueError("trained row hash mismatch")
        rows = np.frombuffer(raw, dtype="<f4").reshape(len(grams), settings["dim"])
        mass = [sum(counts[words[w]] * multiplicity for w, multiplicity in signatures[i]) for i in range(len(grams))]
        word_mass = [counts[w] for w in words]
        original_words = compose_subwords(rows, words, grams, order)
        for threshold in thresholds:
            for policy, norm_threshold in report["norm_policies"].items():
                if policy == "norm_only_0_99" and threshold != thresholds[0]:
                    continue
                cosine_threshold = -1.0 if policy == "norm_only_0_99" else threshold
                start = time.perf_counter()
                mapping = structural_merge(rows, grams, cosine_threshold, norm_threshold)
                reconstructed = rows[mapping]
                merged = mapping != np.arange(len(rows))
                if fidelity(rows, reconstructed)["minimum_cosine"] < cosine_threshold - 1e-12:
                    raise AssertionError("direct representative threshold violated")
                directory = args.output_dir / f"order-{order}-seed-{seed}" / f"tau-{threshold}-{policy}"
                directory.mkdir(parents=True, exist_ok=True)
                approximated_raw, artifact = directory / "approximated.f32", directory / "approximated.men"
                expected = reconstructed.tobytes()
                approximated_raw.write_bytes(expected)
                packed = pack_native(approximated_raw, artifact, settings["dim"] * 4, 64, deduplicate=True)
                with NativeRows(artifact, args.native_library) as table:
                    if table.rows(range(len(rows))) != expected:
                        raise AssertionError("native gather disagrees with approximated rows")
                manifest = directory / "APPROXIMATION.json"
                manifest.write_text(json.dumps({"approximate": True, "source_row_sha256": source["learned_row_sha256"],
                                                "reconstructed_row_sha256": hashlib.sha256(expected).hexdigest(),
                                                "minimum_cosine": cosine_threshold, "minimum_norm_ratio": norm_threshold,
                                                "policy": policy, "original_rows": len(rows), "dtype": "little-endian float32",
                                                "dim": settings["dim"], "warning": "Reconstruction changes original source vectors; not lossless."}, indent=2) + "\n")
                artifact_bytes = packed["artifact_bytes"] + manifest.stat().st_size
                entry = {"max_order": order, "seed": seed, "threshold": threshold if policy != "norm_only_0_99" else None, "policy": policy,
                         "minimum_cosine_threshold": cosine_threshold, "minimum_norm_ratio": norm_threshold,
                         "rows": len(rows), "retained_representatives": int(len(np.unique(mapping))),
                         "merged_fraction": float(merged.mean()), "source_row_sha256": source["learned_row_sha256"],
                         "input_bytes": len(raw), "representative_float_bytes": int(len(np.unique(mapping)) * settings["dim"] * 4),
                         "storage": packed, "approximation_manifest_bytes": manifest.stat().st_size,
                         "total_artifact_bytes": artifact_bytes,
                         "compression_factor_raw_to_artifact": len(raw) / artifact_bytes,
                         "compression_factor_lossless_to_artifact": source["native_lossless_storage"]["artifact_bytes"] / artifact_bytes,
                         "row_fidelity": fidelity(rows, reconstructed, mass),
                         "merged_row_fidelity": fidelity(rows[merged], reconstructed[merged], np.asarray(mass)[merged]) if merged.any() else {"rows": 0},
                         "word_ngram_component_fidelity": fidelity(original_words, compose_subwords(reconstructed, words, grams, order), word_mass),
                         "native_all_approximated_rows_byte_exact": True, "native_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                         "mapping_sha256": hashlib.sha256(mapping.astype("<u8").tobytes()).hexdigest(),
                         "build_and_measure_seconds": time.perf_counter() - start}
                report["runs"].append(entry)
                print(json.dumps({k: entry[k] for k in ("max_order", "seed", "threshold", "policy", "merged_fraction", "compression_factor_raw_to_artifact", "row_fidelity", "word_ngram_component_fidelity")}), flush=True)
                (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
