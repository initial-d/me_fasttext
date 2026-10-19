"""Actual native XOR artifacts and warm gathers on hashed trained SGNS rows.

Uses existing training evidence; no training or external data download. Layouts
include lexical/shuffled and training-signature grouping (needs corpus metadata).
Timing compares ordinary native zlib with adaptive XOR on identical queries.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

try:
    from .engram_row_adapter import file_sha256
    from .native_row_store import NativeRows, pack_native
    from .train_ngram_similarity import prepare
except ImportError:
    from engram_row_adapter import file_sha256
    from native_row_store import NativeRows, pack_native
    from train_ngram_similarity import prepare

ROOT = Path(__file__).resolve().parents[1]


def benchmark(table, library, rows, queries, repeat):
    start = time.perf_counter_ns()
    store = NativeRows(table, library)
    opening_ms = (time.perf_counter_ns() - start) / 1e6
    with store:
        # Bounded verification batches test every original ID, including layout
        # remapping. This also warms the mapped payload; timings are not cold I/O.
        for start in range(0, len(rows), 1024):
            if store.rows(range(start, min(start + 1024, len(rows)))) != rows[start:start + 1024].tobytes():
                raise AssertionError("native rows changed source bytes")
        timings = {}
        for size, query in queries.items():
            expected = rows[query].tobytes()
            ids = (ctypes.c_uint64 * size)(*map(int, query))
            output = ctypes.create_string_buffer(len(expected))
            def gather():
                store.gather_into(ctypes.addressof(ids), size, ctypes.addressof(output), len(output))
            for _ in range(3):
                gather()
            samples = []
            for _ in range(repeat):
                start = time.perf_counter_ns()
                gather()
                samples.append((time.perf_counter_ns() - start) / 1000)
            if output.raw != expected:
                raise AssertionError("query order/bytes changed")
            timings[str(size)] = {"p50_us": float(np.median(samples)),
                                  "p95_us": float(np.percentile(samples, 95)),
                                  "block_reads": store.last_block_reads,
                                  "decompressions": store.last_decompressions,
                                  "ids_sha256": hashlib.sha256(np.asarray(query, dtype="<u8").tobytes()).hexdigest()}
    return {"open_ms_after_export": opening_ms, "all_rows_byte_exact": True, "warm_queries": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=40)
    args = parser.parse_args()
    if args.block_rows <= 0 or args.repeat <= 0:
        parser.error("block rows/repeat must be positive")
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = json.loads((args.training_dir / "report.json").read_text())
    corpus = args.corpus.read_bytes()
    if hashlib.sha256(corpus).hexdigest() != training["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    settings = training["settings"]
    report = {"category": "public-text trained SGNS rows; native lossless storage/warm CPU gather only",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "exact_command": shlex.join([sys.executable, *sys.argv]),
              "source_sha256": {name: file_sha256(ROOT / name) for name in (
                  "native/meft_rows.c", "tools/native_row_store.py", "tools/benchmark_native_residuals.py")},
              "training_commit": training["commit"],
              "training_report_sha256": file_sha256(args.training_dir / "report.json"),
              "corpus_sha256": training["corpus_sha256"], "corpus_bytes": len(corpus),
              "corpus_source": "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
              "native_library_sha256": file_sha256(args.native_library),
              "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": "CPU", "threads": 1,
              "block_rows": args.block_rows, "repeat": args.repeat, "runs": [],
              "cost": "Local CPU; no paid API or rented compute. Monetary/energy cost not measured.",
              "limitations": ["One small English SGNS reference, not trained Engram or paper reproduction.",
                              "Update-group ordering requires corpus/training signatures; lexical ordering requires an external key-to-row mapping.",
                              "Corpus license not established; raw corpus/weights are not redistributed.",
                              "Latency includes ctypes call and C gather; IDs/output are preallocated. No tensor or direct mmap latency baseline.",
                              "Data are warmed by full row verification; opening follows export, without checksum scanning or page-cache reset.",
                              "No RSS, cold disk I/O, GPU transfer, concurrency or model-quality measurement."]}
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu"):
            report[key] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    else:
        report["cpu"] = platform.processor()
    report["compiler"] = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
    for run in training["runs"]:
        order, seed = run["max_order"], run["seed"]
        _, grams, signatures, *_ = prepare(corpus.decode("utf-8"), order,
                                           settings["vocab_limit"], settings["min_count"])
        source = args.training_dir / f"order-{order}-seed-{seed}" / "trained-ngram.f32"
        if file_sha256(source) != run["learned_row_sha256"]:
            raise ValueError("trained row hash mismatch")
        rows = np.fromfile(source, dtype=np.uint8).reshape(len(grams), settings["dim"] * 4)
        layouts = {"lexical": list(range(len(rows))),
                   "shuffled": np.random.default_rng(seed + 1000).permutation(len(rows)).tolist(),
                   "identical_update_group": sorted(range(len(rows)), key=lambda i: (tuple(signatures[i]), grams[i]))}
        rng = np.random.default_rng(seed + 2000)
        queries = {size: rng.integers(0, len(rows), size=size) for size in (1, 32, 256)}
        # Explicit duplicate IDs stress repeated n-gram queries.
        queries[32][::4] = queries[32][0]
        queries[256][::4] = queries[256][0]
        result = {"max_order": order, "seed": seed, "input_bytes": rows.nbytes,
                  "rows": len(rows), "source_sha256": run["learned_row_sha256"], "layouts": {}}
        for layout, permutation in layouts.items():
            variants = {}
            for predictor in ("none", "xor"):
                table = args.output_dir / f"order-{order}-seed-{seed}-{layout}-{predictor}.men"
                start = time.perf_counter_ns()
                packed = pack_native(source, table, rows.shape[1], args.block_rows, False,
                                     predictor, None if layout == "lexical" else permutation)
                packed["export_seconds"] = (time.perf_counter_ns() - start) / 1e9
                packed["artifact_sha256"] = file_sha256(table)
                packed["permutation_sha256"] = hashlib.sha256(np.asarray(permutation, dtype="<u8").tobytes()).hexdigest()
                packed["artifact_to_raw_ratio"] = packed["artifact_bytes"] / rows.nbytes
                variants[predictor] = {**packed, **benchmark(table, args.native_library, rows, queries, args.repeat)}
            if variants["xor"]["artifact_bytes"] > variants["none"]["artifact_bytes"]:
                raise AssertionError("adaptive candidate selection expanded artifact")
            result["layouts"][layout] = variants
        report["runs"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"order={order} seed={seed}: all 6 native artifacts byte-exact", flush=True)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
