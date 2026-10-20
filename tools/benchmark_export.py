"""Compare source-pinned exporters on existing trained rows, in fresh processes.

No external data or training. Original nine tables test XOR file equality/time.
A repeated trained table is a memory-scaling fixture, not compression evidence.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.engram_row_adapter import file_sha256
from tools.native_row_store import NativeRows, pack_native


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Repeated learned rows isolate full-table copying; their duplication is
    # intentional and must not be represented as large-checkpoint evidence.
    base = torch.from_numpy(np.fromfile(args.memory_source, dtype="<f4").reshape(-1, 128))
    weight = base.repeat((args.memory_rows + len(base) - 1) // len(base), 1)[:args.memory_rows].contiguous()
    before = rss_bytes()
    if args.worker == "baseline":
        module = load_module("tools._baseline_export", args.output_dir / "baseline_adapter.py")
        module.pack_native = load_module("baseline_native", args.output_dir / "baseline_native.py").pack_native
    else:
        from tools import engram_row_adapter as module
    with tempfile.TemporaryDirectory(dir=args.output_dir) as tmp:
        start = time.perf_counter()
        meta = module.export_multihead(weight, [len(weight)], tmp, 64, "native", False, "xor")
        seconds = time.perf_counter() - start
        peak = rss_bytes()
        with NativeRows(Path(tmp) / meta["table_file"], args.native_library) as table:
            for offset in range(0, len(weight), 1024):
                expected = weight[offset:offset + 1024].view(torch.uint8).numpy().tobytes()
                if table.rows(range(offset, min(offset + 1024, len(weight)))) != expected:
                    raise AssertionError("export changed learned row bytes")
        print(json.dumps({"mode": args.worker, "export_seconds": seconds,
                          "peak_rss_bytes": peak, "pre_export_peak_rss_bytes": before,
                          "additional_peak_rss_bytes": max(0, peak - before),
                          "input_bytes": meta["input_bytes"], "artifact_bytes": meta["artifact_bytes"],
                          "weight_sha256": meta["weight_sha256"], "artifact_sha256": meta["artifact_sha256"],
                          "all_rows_byte_exact": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="2fd270fd2a6ef4e69fed38c1c27a1ce90917008e")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--memory-rows", type=int, default=131072)
    parser.add_argument("--worker", choices=("baseline", "optimized"), help=argparse.SUPPRESS)
    parser.add_argument("--memory-source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeat <= 0 or args.memory_rows <= 0:
        parser.error("repeat/memory rows must be positive")
    if args.output_dir.exists():
        raise FileExistsError("use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    baseline_sha = subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip()
    for name, source in (("baseline_native.py", "tools/native_row_store.py"),
                         ("baseline_adapter.py", "tools/engram_row_adapter.py")):
        (args.output_dir / name).write_bytes(subprocess.check_output(["git", "show", f"{baseline_sha}:{source}"], cwd=ROOT))
    baseline = load_module("baseline_native", args.output_dir / "baseline_native.py")
    training = json.loads((args.training_dir / "report.json").read_text())
    report = {"category": "trained SGNS export timing; repeated-trained-row memory fixture separately",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "baseline_commit": baseline_sha, "exact_command": shlex.join([sys.executable, *sys.argv]),
              "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": "CPU", "threads": 1, "repeat": args.repeat,
              "training_commit": training["commit"], "training_report_sha256": file_sha256(args.training_dir / "report.json"),
              "corpus_sha256": training["corpus_sha256"],
              "corpus_source": "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
              "source_sha256": {name: file_sha256(ROOT / name) for name in (
                  "tools/native_row_store.py", "tools/engram_row_adapter.py", "tools/benchmark_export.py")},
              "native_library_sha256": file_sha256(args.native_library),
              "cost": "Local CPU, no paid API or rented compute; monetary/energy cost not measured.",
              "limitations": ["Small English SGNS tables, not trained Engram or the paper corpus.",
                              "Memory fixture tiles learned rows; repeated data are not representative compression evidence.",
                              "CPU only, no GPU export/transfer measurement, task-quality or query-speed claim.",
                              "Filesystem cache is warm; no cold I/O control. Timing order alternates but has no statistical significance analysis.",
                              "RSS is process high-water mark; delta is above pre-export peak, not exact live allocation; excludes disk/cache space.",
                              "Native no-dedup export is block-bounded except descriptors; Python builder/dedup dictionary remain unbounded.",
                              "Corpus license not established; corpus and weights not redistributed."],
              "trained_exports": [], "memory_fixture": []}
    if sys.platform == "darwin":
        report["hardware"] = {k: subprocess.check_output(["sysctl", "-n", k], text=True).strip()
                              for k in ("machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu")}
    else:
        report["hardware"] = {"cpu": platform.processor()}
    report["compiler"] = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
    for run in training["runs"]:
        order, seed = run["max_order"], run["seed"]
        source = args.training_dir / f"order-{order}-seed-{seed}" / "trained-ngram.f32"
        if file_sha256(source) != run["learned_row_sha256"]:
            raise ValueError("trained row hash mismatch")
        result = {"max_order": order, "seed": seed, "source_sha256": run["learned_row_sha256"],
                  "input_bytes": source.stat().st_size, "samples": []}
        for repeat in range(args.repeat):
            variants = {}
            for name in (("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")):
                output = args.output_dir / "timing.men"
                start = time.perf_counter()
                (baseline.pack_native if name == "baseline" else pack_native)(source, output, 512, 64, False, "xor")
                seconds = time.perf_counter() - start
                variants[name] = {"seconds": seconds, "artifact_sha256": file_sha256(output),
                                  "artifact_bytes": output.stat().st_size}
                with NativeRows(output, args.native_library) as store, source.open("rb") as raw:
                    for offset in range(0, store.header["n_rows"], 1024):
                        ids = range(offset, min(offset + 1024, store.header["n_rows"]))
                        if store.rows(ids) != raw.read(len(ids) * 512):
                            raise AssertionError("native row bytes changed")
            if variants["baseline"]["artifact_sha256"] != variants["optimized"]["artifact_sha256"]:
                raise AssertionError("optimized file differs from baseline")
            result["samples"].append(variants)
        result["all_rows_byte_exact"] = True
        report["trained_exports"].append(result)
        print(f"order={order} seed={seed}: baseline/optimized files identical", flush=True)
    memory_source = args.training_dir / "order-12-seed-7" / "trained-ngram.f32"
    for repeat in range(args.repeat):
        variants = {}
        for mode in (("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                       "--training-dir", str(args.training_dir.resolve()), "--native-library", str(args.native_library.resolve()),
                       "--output-dir", str(args.output_dir.resolve()), "--memory-source", str(memory_source.resolve()),
                       "--memory-rows", str(args.memory_rows)]
            variants[mode] = json.loads(subprocess.check_output(command, text=True))
        if variants["baseline"]["artifact_sha256"] != variants["optimized"]["artifact_sha256"]:
            raise AssertionError("streamed full export differs from baseline")
        report["memory_fixture"].append(variants)
        print(f"memory fixture pair {repeat + 1}: artifacts identical", flush=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
