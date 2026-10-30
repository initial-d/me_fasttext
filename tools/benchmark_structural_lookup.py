"""Three C++ serving-index baselines on exactly the stored memory-ablation traces.

Hash lookup is std::unordered_map, flat lookup binary search over sorted owned
strings, and DATrie lookup the original library's converted DASTrie. No Python
key conversion or tensor allocation is timed. No hash-index RSS/storage claim.
"""
import argparse
import ctypes
import json
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

from tools.benchmark_structural_native import ROOT, sha, file_sha
from tools.structural_native import StructuralPlan, _keys
from tools.train_ngram_similarity import prepare, ngrams
from tools.causal_ngram_lm import encode_segment
from tools.benchmark_causal_lm_compression import text_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--sgns-dir", type=Path, required=True)
    parser.add_argument("--gru-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--structural-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=120)
    cfg = parser.parse_args()
    if cfg.repeat <= 0:
        parser.error("repeat must be positive")
    if cfg.output.exists():
        raise FileExistsError("use a new output path")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evidence = json.loads(cfg.memory_report.read_text())
    if evidence["failures"]:
        raise ValueError("memory ablation has failed runs")
    corpus = cfg.corpus.read_bytes()
    if sha(corpus) != evidence["corpus_sha256"]:
        raise ValueError("corpus provenance mismatch")
    sgns = json.loads((cfg.sgns_dir / "report.json").read_text())
    gru = json.loads((cfg.gru_dir / "report.json").read_text())
    validation = text_splits(corpus.decode(), gru["settings"]["max_characters"])["validation"]
    report = {"category": "trained SGNS/GRU key traces; C++ serving-index-only ablation",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
              "pytorch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
              "memory_report_sha256": file_sha(cfg.memory_report), "corpus_sha256": sha(corpus),
              "library_sha256": file_sha(cfg.structural_library),
              "source_sha256": {name: file_sha(ROOT / name) for name in (
                  "native/meft_structural.cc", "tools/structural_native.py", "tools/benchmark_structural_lookup.py")},
              "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
              "device": "CPU", "gpu": "unused", "threads": 1, "repeat": cfg.repeat,
              "method_order_seed": 42, "cpu": evidence["cpu"],
              "cost_assumptions": evidence["cost_assumptions"],
              "limitations": ["Only key->original ID, no row decode/whole-model/GPU speedup.",
                  "Warm built indices include baseline-owned strings/hash entries; serialized DASTrie is verified in the memory report.",
                  "Same ctypes batch boundary and query/output allocation outside timing for all three C++ implementations.",
                  "Hash uses standard library defaults; no optimized hash-table, minimal perfect hash, concurrency or RSS baseline.",
                  "Serving-size comparisons in memory report are flat serialized keys vs DASTrie; hash bytes are not estimated.",
                  "Same small trained reference data as memory report; not trained DeepSeek Engram or original paper reproduction."],
              "runs": []}
    prepared = {}
    rng = random.Random(42)
    for run in evidence["runs"]:
        category, order, seed = run["category"], run["max_order"], run["seed"]
        if category == "sgns":
            if order not in prepared:
                words, grams, *_ = prepare(corpus.decode(), order, sgns["settings"]["vocab_limit"],
                                           sgns["settings"]["min_count"])
                ids = {key: i for i, key in enumerate(grams)}
                trace = [ids[g] for word in words for g in ngrams(word, order)]
                prepared[order] = (dict(enumerate(grams)), trace)
            keys, trace = prepared[order]
            raw = (cfg.sgns_dir / f"order-{order}-seed-{seed}/trained-ngram.f32").read_bytes()
        else:
            checkpoint = cfg.gru_dir / f"order-{order}-seed-{seed}/reference.pt"
            if file_sha(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError("checkpoint provenance mismatch")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            raw = state["model_state"]["memory.weight"].numpy().astype("<f4").tobytes()
            keys = {i+1: key for i, key in enumerate(state["registry"]["grams"])}
            trace = encode_segment(validation, state["registry"])["memory_ids"].numpy().reshape(-1)
            trace = trace[trace != 0]
        trace = np.asarray(trace[:run["query_rows"]], dtype="<u8")
        if sha(raw) != run["source_sha256"] or sha(trace.tobytes()) != run["query_ids_sha256"]:
            raise ValueError("row or query trace mismatch")
        query = _keys([keys[int(i)] for i in trace])
        output = (ctypes.c_uint64 * len(query))()
        with StructuralPlan(raw, run["dim"], keys, cfg.structural_library, mode="layout") as plan:
            timings = {name: [] for name in ("flat", "hash", "datrie")}
            for _ in range(3):
                for method in timings:
                    plan.lookup_into(query, output, index=method)
            for _ in range(cfg.repeat):
                methods = list(timings)
                rng.shuffle(methods)
                for method in methods:
                    start = time.perf_counter_ns()
                    plan.lookup_into(query, output, index=method)
                    timings[method].append((time.perf_counter_ns() - start) / 1e6)
                    if list(output) != trace.tolist():
                        raise AssertionError("index changed original IDs")
        report["runs"].append({"category": category, "max_order": order, "seed": seed,
            "source_sha256": run["source_sha256"], "query_rows": len(trace),
            "query_ids_sha256": sha(trace.tobytes()), "query_key_bytes_sha256": run["query_key_bytes_sha256"],
            "lookup": {name: {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                              "samples_ms": values} for name, values in timings.items()}})
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(report['runs'])} trained query traces through all three C++ indices.")


if __name__ == "__main__":
    main()
