"""Bounded native cache on hashed trained rows and explicit CPU query traces.

Corpus words use their original character-ngram IDs. Hot and uniform controls
are query-locality fixtures on the same learned rows, not learned Engram traces.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.engram_row_adapter import file_sha256
from tools.native_row_store import NativeRows, pack_native
from tools.train_ngram_similarity import prepare, ngrams


def measure(path, library, raw, trace, budget, repeat):
    queries = []
    for ids in trace:
        ids = list(map(int, ids))
        query = (ctypes.c_uint64 * len(ids))(*ids)
        out = ctypes.create_string_buffer(len(ids) * raw.shape[1])
        expected = raw[ids].tobytes()
        queries.append((query, out, expected))
    with NativeRows(path, library) as store:
        for offset in range(0, len(raw), 1024):
            if store.rows(range(offset, min(offset + 1024, len(raw)))) != raw[offset:offset + 1024].tobytes():
                raise AssertionError("original rows changed")
        store.set_cache_bytes(budget)  # Each trace starts with an empty cache.
        def gather(query, out):
            store.gather_into(ctypes.addressof(query), len(query), ctypes.addressof(out), len(out))
        for _ in range(3):
            for query, out, expected in queries:
                gather(query, out)
                if out.raw != expected:
                    raise AssertionError("warmup bytes changed")
        samples, hits, decodes, blocks, peak_cache = [], 0, 0, 0, store.cache_bytes
        for _ in range(repeat):
            for query, out, expected in queries:
                start = time.perf_counter_ns()
                gather(query, out)
                samples.append((time.perf_counter_ns() - start) / 1000)
                hits += store.last_cache_hits
                decodes += store.last_decompressions
                blocks += store.last_block_reads
                peak_cache = max(peak_cache, store.cache_bytes)
                if out.raw != expected or peak_cache > budget:
                    raise AssertionError("cache output or byte budget violated")
        return {"p50_us": float(np.median(samples)), "p95_us": float(np.percentile(samples, 95)),
                "query_count": len(samples), "block_hits": hits, "decompressions": decodes,
                "block_reads": blocks, "peak_cached_payload_bytes": peak_cache,
                "all_source_rows_and_queries_byte_exact": True,
                "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cache-bytes", type=int, default=262144)
    parser.add_argument("--larger-cache-bytes", type=int, default=2097152)
    args = parser.parse_args()
    if args.repeat <= 0 or args.cache_bytes <= 0 or args.larger_cache_bytes <= args.cache_bytes:
        parser.error("repeat and cache bytes must be positive")
    if args.output_dir.exists():
        raise FileExistsError("use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    training = json.loads((args.training_dir / "report.json").read_text())
    corpus = args.corpus.read_bytes()
    if hashlib.sha256(corpus).hexdigest() != training["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    text = corpus.decode("utf-8")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    report = {"category": "trained character-ngram rows; controlled CPU cache/locality traces, not Engram workload",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "exact_command": shlex.join([sys.executable, *sys.argv]),
              "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": "CPU", "threads": 1,
              "repeat": args.repeat, "cache_budget": args.cache_bytes,
              "cache_entry_cap": None, "cache_index": "lazy hash index and linked LRU; payload-budget bounded",
              "cache_budgets": [0, args.cache_bytes, args.larger_cache_bytes],
              "training_commit": training["commit"], "training_report_sha256": file_sha256(args.training_dir / "report.json"),
              "corpus_sha256": training["corpus_sha256"], "corpus_bytes": len(corpus),
              "corpus_source": "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
              "source_sha256": {name: file_sha256(ROOT / name) for name in (
                  "native/meft_rows.c", "native/meft_rows.h", "tools/native_row_store.py", "tools/benchmark_native_cache.py")},
              "native_library_sha256": file_sha256(args.native_library),
              "cost": "Local CPU; no paid API or rented compute, monetary/energy cost unmeasured.",
              "limitations": ["Small English SGNS rows, not trained Engram or original paper reproduction.",
                              "Corpus trace replays a short retained-word prefix; excludes OOV words and word rows, is not held-out serving evidence.",
                              "Hot/uniform controls manipulate ID locality explicitly; random IDs are not random embeddings.",
                              "Warm filesystem; three trace warmups, then repeat passes over the same 16 batches. No cold I/O or real trace distribution.",
                              "ctypes/C gather timed with preallocated IDs/output; no resident tensor/raw mmap latency baseline or end-to-end model speedup.",
                              "Cache payload counter excludes dynamic hash/LRU metadata, allocator overhead, scratch, per-batch requests, mapped pages and process RSS.",
                              "CPU single-worker only; no GPU, concurrency, task quality or trained Engram checkpoint.",
                              "Corpus license not established; corpus/weights are not redistributed."], "runs": []}
    if sys.platform == "darwin":
        report["hardware"] = {k: subprocess.check_output(["sysctl", "-n", k], text=True).strip()
                              for k in ("machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu")}
    report["compiler"] = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
    settings = training["settings"]
    for run in training["runs"]:
        order, seed = run["max_order"], run["seed"]
        words, grams, *_ = prepare(text, order, settings["vocab_limit"], settings["min_count"])
        gram_ids = {g: i for i, g in enumerate(grams)}
        vocabulary = set(words)
        retained = [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()) if w in vocabulary][:128]
        source = args.training_dir / f"order-{order}-seed-{seed}" / "trained-ngram.f32"
        if file_sha256(source) != run["learned_row_sha256"]:
            raise ValueError("trained row hash mismatch")
        raw = np.fromfile(source, dtype=np.uint8).reshape(len(grams), 512)
        rng = np.random.default_rng(seed + 3000)
        trace = {"hot_8_blocks": [rng.integers(0, 512, size=64).tolist() for _ in range(16)],
                 "uniform_table": [rng.integers(0, len(raw), size=64).tolist() for _ in range(16)],
                 "corpus_words": [[gram_ids[g] for w in retained[i:i + 8] for g in ngrams(w, order)]
                                  for i in range(0, len(retained), 8)]}
        table = args.output_dir / f"order-{order}-seed-{seed}.men"
        packed = pack_native(source, table, 512, 64, False, "xor")
        result = {"max_order": order, "seed": seed, "source_sha256": run["learned_row_sha256"],
                  "artifact_sha256": file_sha256(table), "artifact": packed, "traces": {}}
        for name, batches in trace.items():
            variants = {}
            # Reverse mode order across seeds to reduce a fixed ordering bias.
            budgets = [0, args.cache_bytes, args.larger_cache_bytes]
            for budget in (budgets if seed != 7 else list(reversed(budgets))):
                variants[str(budget)] = measure(table, args.native_library, raw, batches, budget, args.repeat)
            result["traces"][name] = {"batch_sizes": [len(x) for x in batches],
                                      "trace_sha256": hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
                                      "variants": variants}
        report["runs"].append(result)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"order={order} seed={seed}: all rows and cache traces byte-exact", flush=True)


if __name__ == "__main__":
    main()
