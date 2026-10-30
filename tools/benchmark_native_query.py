"""Profile C gather on existing trained held-out rows, with paired libraries.

Each variant runs in a fresh process with preallocated query/output buffers.
Native variants never load dense.pt; original output hashes are the oracle.
This isolates CPU reads, not CUDA transfers or full-model inference.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

import torch

from .benchmark_device_gather import fixture_inputs, rss
from .engram_row_adapter import file_sha256
from .native_row_store import NativeRows

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, replay = fixture_inputs(args.fixture, "native")
    metadata = json.loads((args.fixture / "native/embedding.json").read_text())
    if (len(manifest["shape"]) != 2 or metadata["head_sizes"] != [manifest["shape"][0]]
            or metadata["offsets"] != [0]):
        raise ValueError("this replay requires an original single-head table with zero offset")
    if args.positions and not set(args.positions).issubset({c["positions"] for c in manifest["cases"]}):
        raise ValueError("requested positions are absent from the original fixture")
    report = {"status": "running", "command": shlex.join([sys.executable, *sys.argv]),
              "python": sys.version, "torch": torch.__version__, "os": platform.platform(),
              "machine": platform.machine(), "threads": 1, "interop_threads": 1,
              "fixture_sha256": file_sha256(args.fixture / "manifest.json"),
              "weight_sha256": manifest["weight_sha256"], "source_shape": manifest["shape"],
              "library_sha256": file_sha256(args.worker_library),
              "cache_budget_bytes": args.worker_cache, "chunk_rows": args.worker_chunk,
              "rss_before_open": rss(), "cases": []}
    if sys.platform == "darwin":
        report["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    elif Path("/proc/cpuinfo").exists():
        report["cpu"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                              if line.startswith("model name")), platform.machine())
    with NativeRows(args.fixture / "native/table.men", args.worker_library) as store:
        report["rss_after_open"] = rss()
        for case in manifest["cases"]:
            if args.positions and case["positions"] not in args.positions:
                continue
            # One head with zero offset in these original character-memory fixtures.
            queries = []
            for item in case["queries"]:
                ids = replay[item["key"]].reshape(-1).contiguous()
                output = ctypes.create_string_buffer(ids.numel() * store.header["row_bytes"])
                queries.append((item, ids, output))
            store.set_cache_bytes(args.worker_cache)  # independent empty-cache cases

            def gather(ids, output):
                blocks = decodes = hits = 0
                for start in range(0, ids.numel(), args.worker_chunk):
                    count = min(args.worker_chunk, ids.numel() - start)
                    store.gather_into(ids.data_ptr() + start * ids.element_size(), count,
                                      ctypes.addressof(output) + start * store.header["row_bytes"],
                                      count * store.header["row_bytes"])
                    blocks += store.last_block_reads
                    decodes += store.last_decompressions
                    hits += store.last_cache_hits
                return blocks, decodes, hits

            def check(item, output):
                if hashlib.sha256(output.raw).hexdigest() != item["output_sha256"]:
                    raise AssertionError("native bytes changed: " + item["key"])

            for _ in range(args.warmup):
                for item, ids, output in queries:
                    gather(ids, output)
                    check(item, output)
            samples, cpu_samples, counters = [], [], [0, 0, 0]
            peak_cache = store.cache_bytes
            for _ in range(args.repeat):
                for item, ids, output in queries:
                    cpu_start, start = time.process_time_ns(), time.perf_counter_ns()
                    counts = gather(ids, output)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                    cpu_samples.append((time.process_time_ns() - cpu_start) / 1e6)
                    counters = [a + b for a, b in zip(counters, counts)]
                    peak_cache = max(peak_cache, store.cache_bytes)
                    if peak_cache > args.worker_cache:
                        raise AssertionError("cache payload exceeded configured budget")
                    check(item, output)
            ordered = sorted(samples)
            report["cases"].append({"positions": case["positions"], "queries": len(samples),
                                    "query_hashes": [{"ids": q[0]["ids_sha256"], "output": q[0]["output_sha256"]}
                                                     for q in queries],
                                    "p50_ms": statistics.median(samples),
                                    "p95_ms": ordered[max(0, int(.95 * len(ordered) + .999999) - 1)],
                                    "samples_ms": samples, "cpu_samples_ms": cpu_samples,
                                    "block_reads": counters[0], "decompressions": counters[1],
                                    "cache_hits": counters[2], "peak_cached_payload_bytes": peak_cache,
                                    "all_outputs_byte_exact": True, "rss_after_case": rss()})
    report["status"] = "passed"
    args.worker_report.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--library", action="append", default=[], help="label=/absolute/library/path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-bytes", default="0,1048576,4194304")
    parser.add_argument("--chunk-rows", default="256,8192")
    parser.add_argument("--positions", default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--worker-library", type=Path)
    parser.add_argument("--worker-cache", type=int)
    parser.add_argument("--worker-chunk", type=int)
    parser.add_argument("--worker-report", type=Path)
    args = parser.parse_args()
    args.positions = [int(n) for n in args.positions.split(",") if n]
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup nonnegative")
    if args.worker_library:
        if args.worker_cache is None or args.worker_cache < 0 or not args.worker_chunk or args.worker_chunk < 1:
            parser.error("worker requires nonnegative cache bytes and positive chunk rows")
        worker(args)
        return
    budgets, chunks = [int(n) for n in args.cache_bytes.split(",")], [int(n) for n in args.chunk_rows.split(",")]
    if not args.output_dir or not args.library or min(budgets) < 0 or min(chunks) < 1:
        parser.error("libraries, fresh output directory, nonnegative budgets and positive chunks required")
    libraries = [value.split("=", 1) for value in args.library]
    labels = [value[0] for value in libraries]
    if len(set(labels)) != len(labels) or any(not label.replace("-", "").replace("_", "").isalnum() for label in labels):
        parser.error("unique alphanumeric library labels required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "category": "trained character-memory held-out CPU gather replay",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]),
              "benchmark_sha256": file_sha256(__file__), "warmup": args.warmup, "repeat": args.repeat,
              "cost": "Existing local/shared CPU; no paid API or new rented compute. Energy/monetary cost unmeasured.",
              "limitations": ["Small original trained GRU tables, not TinyEngram/DeepSeek or paper reproduction.",
                              "Repeated held-out windows and warm filesystem; no cold I/O or production distribution.",
                              "Preallocated CPU output; excludes GPU transfers, staging events and full-model inference.",
                              "Cache payload budget excludes hash/LRU metadata, scratch, output, mmap and allocator overhead.",
                              "Fixed variant order, single worker, shared host; no statistical significance claim."], "variants": []}
    for chunk in chunks:
        for budget in budgets:
            for label, library in libraries:
                target = args.output_dir / f"{label}-cache{budget}-chunk{chunk}.json"
                cmd = [sys.executable, "-m", "tools.benchmark_native_query", "--fixture", str(args.fixture),
                       "--worker-library", str(Path(library).resolve()), "--worker-cache", str(budget),
                       "--worker-chunk", str(chunk), "--worker-report", str(target),
                       "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
                if args.positions:
                    cmd.extend(["--positions", ",".join(map(str, args.positions))])
                subprocess.run(cmd, cwd=ROOT, check=True)
                result = json.loads(target.read_text())
                report["variants"].append({"label": label, "report": target.name, **result})
                (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(label, budget, chunk, [(c["positions"], round(c["p50_ms"], 3), c["decompressions"], c["cache_hits"])
                                             for c in result["cases"]], flush=True)
    report["status"] = "passed"
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
