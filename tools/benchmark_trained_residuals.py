"""Offline byte-exact XOR prediction on trained n-gram rows, not a native codec.

Consumes a hashed training report/corpus/row files. Compare lexical, shuffled
and identical-update-group layouts using ordinary zlib and reversible XOR.
No floating-point subtraction, quantization or approximate row merging.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zlib

import numpy as np

try:
    from .train_ngram_similarity import prepare
except ImportError:
    from train_ngram_similarity import prepare


def xor_encode(rows):
    encoded = rows.copy()
    encoded[1:] = rows[1:] ^ rows[:-1]
    return encoded


def xor_decode(encoded):
    return np.bitwise_xor.accumulate(encoded, axis=0)


def measure(rows, permutation, block_rows):
    ordered = rows[permutation]
    decoded = np.empty_like(ordered)
    plain_bytes = xor_bytes = 0
    for start in range(0, len(rows), block_rows):
        block = ordered[start:start + block_rows]
        raw = block.tobytes()
        ordinary = zlib.compress(raw)
        residual = zlib.compress(xor_encode(block).tobytes())
        plain_bytes += min(len(raw), len(ordinary))
        xor_bytes += min(len(raw), len(residual))
        restored = xor_decode(np.frombuffer(zlib.decompress(residual), dtype=np.uint8).reshape(block.shape))
        decoded[start:start + len(block)] = restored
    # Restore original numeric IDs even after the construction permutation.
    original = np.empty_like(rows)
    original[permutation] = decoded
    if not np.array_equal(original, rows):
        raise AssertionError("XOR codec did not preserve every original byte")
    blocks = (len(rows) + block_rows - 1) // block_rows
    # Estimate uses existing native map/descriptor widths, not a written format.
    overhead = 56 + 8 * len(rows) + 32 * blocks
    return {"plain_payload_bytes": plain_bytes, "xor_payload_bytes": xor_bytes,
            "estimated_overhead_bytes": overhead,
            "estimated_plain_container_bytes": overhead + plain_bytes,
            "estimated_xor_container_bytes": overhead + xor_bytes,
            "estimated_xor_to_raw_ratio": (overhead + xor_bytes) / rows.nbytes,
            "original_id_roundtrip_byte_exact": True,
            "permutation_sha256": hashlib.sha256(np.asarray(permutation, dtype="<u8").tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=64)
    cfg = parser.parse_args()
    if cfg.block_rows <= 0:
        parser.error("block rows must be positive")
    source = (cfg.training_dir / "report.json").read_bytes()
    training = json.loads(source)
    corpus = cfg.corpus.read_bytes()
    if hashlib.sha256(corpus).hexdigest() != training["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    settings = training["settings"]
    report = {"category": "offline trained-row lossless prediction/layout experiment; no native gather codec",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "exact_command": " ".join([sys.executable, *sys.argv]),
              "training_report_sha256": hashlib.sha256(source).hexdigest(),
              "training_commit": training["commit"], "corpus_sha256": training["corpus_sha256"],
              "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
              "block_rows": cfg.block_rows, "runs": [],
              "limitations": ["Small English SGNS reference, not learned Engram or paper weights.",
                              "Update signatures are derived from the training corpus; this metadata is not in a frozen Engram table.",
                              "Container sizes are estimates with map/header/descriptors; this codec is not in MEFTNAT1.",
                              "No construction/RSS/query latency or downstream-quality measurement."]}
    for run in training["runs"]:
        order, seed = run["max_order"], run["seed"]
        _, grams, signatures, *_ = prepare(corpus.decode("utf-8"), order, settings["vocab_limit"], settings["min_count"])
        raw = (cfg.training_dir / f"order-{order}-seed-{seed}" / "trained-ngram.f32").read_bytes()
        if hashlib.sha256(raw).hexdigest() != run["learned_row_sha256"]:
            raise ValueError("trained row hash mismatch")
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(len(grams), settings["dim"] * 4)
        permutations = {"lexical": np.arange(len(rows)),
                        "shuffled": np.random.default_rng(seed + 1000).permutation(len(rows)),
                        "identical_update_group": np.asarray(sorted(range(len(rows)), key=lambda i: (tuple(signatures[i]), grams[i])))}
        report["runs"].append({"max_order": order, "seed": seed, "input_bytes": len(raw),
                               "layouts": {name: measure(rows, perm, cfg.block_rows) for name, perm in permutations.items()}})
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
