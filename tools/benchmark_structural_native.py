"""Isolate actual DATrie indexing and mark/compact export on trained memories.

Reuses local SGNS tables and causal GRU checkpoints, never random embeddings,
network downloads, retraining or test-selected thresholds. Flat/DASTrie serving
share the same production DATrie structural builder; this is a serving-index
ablation, not a trie-free build baseline. Numeric/hash IDs remain unchanged.
"""
import argparse
import ctypes
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

from tools.structural_native import StructuralPlan, StructuralIndex, _keys
from tools.native_row_store import NativeRows, pack_native
from tools.train_ngram_similarity import prepare
from tools.benchmark_cosine_merge import fidelity
from tools.causal_ngram_lm import CausalNgramLM, encode_segment, evaluate, metric_summary, prediction_delta
from tools.benchmark_causal_lm_compression import text_splits

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def measured(call, repeat):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "samples_ms": samples}


class NativeMemory(nn.Module):
    """CPU inference only: preserve original model embedding shape/ID contract."""
    def __init__(self, table, dim):
        super().__init__()
        self.table, self.dim = table, dim

    def forward(self, ids):
        ids = ids.contiguous()
        output = torch.empty((*ids.shape, self.dim), dtype=torch.float32)
        self.table.gather_into(ids.data_ptr(), ids.numel(), output.data_ptr(), output.numel() * 4)
        return output


def query_bench(path, expected, query_ids, plan, query_keys, cfg):
    ids = (ctypes.c_uint64 * len(query_ids))(*map(int, query_ids))
    output = ctypes.create_string_buffer(len(query_ids) * expected.shape[1] * 4)
    encoded = _keys(query_keys)
    resolved = (ctypes.c_uint64 * len(query_keys))()
    result = {}
    for policy in ("blocks", "rows"):
        result[policy] = {}
        for budget in (0, cfg.cache_bytes):
            with NativeRows(path, cfg.native_library, cache_bytes=budget, cache_policy=policy) as table:
                def gather():
                    table.gather_into(ctypes.addressof(ids), len(ids), ctypes.addressof(output), len(output))
                # Fresh handle / empty application cache. Filesystem is already
                # warm after export/verification, so this is not disk-cold I/O.
                start = time.perf_counter_ns()
                gather()
                first = {"ms": (time.perf_counter_ns() - start) / 1e6,
                         "decompressions": table.last_decompressions, "block_reads": table.last_block_reads}
                reference_bytes = expected[query_ids].tobytes()
                if output.raw != reference_bytes:
                    raise AssertionError("first query byte mismatch")
                for _ in range(3):
                    gather()
                warm = measured(gather, cfg.repeat)
                warm.update({"decompressions": table.last_decompressions,
                             "block_reads": table.last_block_reads,
                             "block_cache_hits": table.last_cache_hits,
                             "row_cache_hits": table.last_row_cache_hits})
                if output.raw != reference_bytes:
                    raise AssertionError("warm query byte mismatch")
                end_to_end = {}
                for index in ("flat", "datrie"):
                    def lookup_gather():
                        plan.lookup_into(encoded, resolved, index=index)
                        table.gather_into(ctypes.addressof(resolved), len(resolved),
                                          ctypes.addressof(output), len(output))
                    lookup_gather()
                    end_to_end[index] = measured(lookup_gather, cfg.repeat)
                    if list(resolved) != list(query_ids) or output.raw != reference_bytes:
                        raise AssertionError("serving index changed original IDs or rows")
                result[policy][str(budget)] = {"first_empty_application_cache": first,
                                               "warm_numeric_gather": warm,
                                               "warm_key_lookup_and_gather": end_to_end}
    return result


def run_dataset(category, order, seed, source, keys, dim, training_row_hash, cfg,
                model=None, segments=None):
    raw = source.read_bytes()
    if sha(raw) != training_row_hash:
        raise ValueError("trained row hash mismatch")
    rows = np.frombuffer(raw, dtype="<f4").reshape(-1, dim)
    directory = cfg.output_dir / f"{category}-order-{order}-seed-{seed}"
    directory.mkdir()
    # All benchmark IDs refer to real trained rows. GRU replays consecutive
    # suffix query slots, excluding padding for the key-index comparison. SGNS
    # replays lexical ngram bags from the training vocabulary.
    query_ids = np.asarray(list(keys)[:cfg.query_rows], dtype=np.int64)
    if segments is not None:
        trace = segments["validation"]["memory_ids"].numpy().reshape(-1)
        query_ids = trace[trace != 0][:cfg.query_rows]
    else:
        query_ids = np.asarray(keys.query_trace[:cfg.query_rows], dtype=np.int64)
    if not len(query_ids):
        raise ValueError("empty query trace")
    query_keys = [keys[int(i)] for i in query_ids]
    entry = {"category": category, "max_order": order, "seed": seed, "rows": len(rows), "dim": dim,
             "source_sha256": training_row_hash, "input_bytes": len(raw),
             "query_rows": len(query_ids), "query_ids_sha256": sha(query_ids.astype("<u8").tobytes()),
             "query_key_bytes_sha256": sha(b"".join(len(k.encode()).to_bytes(8, "little") + k.encode() for k in query_keys)),
             "variants": {}}
    baseline_scores = ({name: evaluate(model, segment) for name, segment in segments.items()}
                       if model is not None else None)
    if baseline_scores is not None:
        entry["baseline_quality"] = {name: metric_summary(score) for name, score in baseline_scores.items()}
    original_memory = model.memory if model is not None else None
    baseline_table = directory / "ordinary.men"
    baseline = pack_native(source, baseline_table, dim * 4, cfg.block_rows, False, "xor")
    for mode in ("layout", "exact", "joint"):
        start = time.perf_counter_ns()
        plan = StructuralPlan(raw, dim, keys, cfg.structural_library, mode=mode,
                              cosine=cfg.cosine, norm_ratio=cfg.norm_ratio)
        build_ms = (time.perf_counter_ns() - start) / 1e6
        with plan:
            output = directory / mode
            start = time.perf_counter_ns()
            metadata = plan.export(source, output, block_rows=cfg.block_rows)
            export_ms = (time.perf_counter_ns() - start) / 1e6
            plan.save_index(output / "keys.flat", index="flat")
            indices = {"flat": (output / "keys.flat").stat().st_size,
                       "datrie": (output / "keys.das").stat().st_size}
            expected = rows[np.asarray(plan.representatives)]
            with NativeRows(output / "table.men", cfg.native_library) as table:
                for begin in range(0, len(rows), 1024):
                    if table.rows(range(begin, min(begin + 1024, len(rows)))) != expected[begin:begin+1024].tobytes():
                        raise AssertionError("compaction reconstruction mismatch")
            with StructuralIndex(output / "keys.das", cfg.structural_library) as index:
                if index.lookup(query_keys) != query_ids.tolist():
                    raise AssertionError("serialized DASTrie lost original IDs")
            if mode != "joint" and expected.tobytes() != raw:
                raise AssertionError("lossless mode changed original bytes")
            encoded, ids = _keys(query_keys), (ctypes.c_uint64 * len(query_keys))()
            lookup = {}
            for index in ("flat", "datrie"):
                plan.lookup_into(encoded, ids, index=index)
                lookup[index] = measured(lambda: plan.lookup_into(encoded, ids, index=index), cfg.repeat)
                if list(ids) != query_ids.tolist():
                    raise AssertionError("lookup lost original IDs")
            result = {"plan_build_ms": build_ms, "export_ms": export_ms, "metadata": metadata,
                      "index_bytes": indices, "index_lookup": lookup,
                      "marked_rows": len(set(plan.representatives)),
                      "merged_rows": sum(i != r for i, r in enumerate(plan.representatives)),
                      "table_sha256": file_sha(output / "table.men"),
                      "fidelity": fidelity(rows, expected),
                      "all_original_ids_byte_exact_to_representatives": True,
                      "four_way_artifact_bytes": {
                          "flat_ordinary": baseline["artifact_bytes"] + indices["flat"],
                          "datrie_ordinary": baseline["artifact_bytes"] + indices["datrie"],
                          "flat_compact": metadata["artifact_bytes"] + indices["flat"],
                          "datrie_compact": metadata["artifact_bytes"] + indices["datrie"]},
                      "queries": query_bench(output / "table.men", expected, query_ids, plan, query_keys, cfg)}
            # Ordinary row table is the same across index factors / modes.
            if mode == "layout":
                entry["ordinary_queries"] = query_bench(baseline_table, rows, query_ids, plan, query_keys, cfg)
            if model is not None:
                with NativeRows(output / "table.men", cfg.native_library,
                                cache_bytes=cfg.cache_bytes, cache_policy="rows") as table:
                    model.memory = NativeMemory(table, dim)
                    scores = {name: evaluate(model, segment) for name, segment in segments.items()}
                    result["quality"] = {name: {"metrics": metric_summary(score),
                        "delta": prediction_delta(score, baseline_scores[name])} for name, score in scores.items()}
                    if mode != "joint" and any(not torch.equal(scores[name]["logits"], baseline_scores[name]["logits"])
                                               for name in scores):
                        raise AssertionError("lossless native model logits changed")
                model.memory = original_memory
            entry["variants"][mode] = result
    return entry


class TracedKeys(dict):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgns-dir", type=Path, required=True)
    parser.add_argument("--gru-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--structural-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orders", default="4,8,12")
    parser.add_argument("--seeds", default="3,7,11")
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--cache-bytes", type=int, default=4194304)
    parser.add_argument("--query-rows", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--cosine", type=float, default=.99)
    parser.add_argument("--norm-ratio", type=float, default=.99)
    cfg = parser.parse_args()
    if min(cfg.block_rows, cfg.query_rows, cfg.repeat, cfg.cache_bytes) <= 0:
        parser.error("positive block/query/repeat/cache dimensions required")
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sgns = json.loads((cfg.sgns_dir / "report.json").read_text())
    gru = json.loads((cfg.gru_dir / "report.json").read_text())
    corpus = cfg.corpus.read_bytes()
    if sha(corpus) != sgns["corpus_sha256"] or sha(corpus) != gru["corpus_sha256"]:
        raise ValueError("corpus provenance mismatch")
    report = {"category": "public-text trained SGNS + small causal GRU; original-algorithm native ablation",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
              "pytorch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
              "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "device": "CPU", "gpu": "unused",
              "threads": 1, "interop_threads": 1,
              "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
              "corpus_source": gru["corpus_source"], "corpus_sha256": sha(corpus),
              "corpus_bytes": len(corpus),
              "training_reports_sha256": {"sgns": file_sha(cfg.sgns_dir / "report.json"),
                                          "gru": file_sha(cfg.gru_dir / "report.json")},
              "source_sha256": {name: file_sha(ROOT / name) for name in (
                  "native/meft_structural.cc", "native/meft_rows.c", "src/legacy_merge.h", "src/row_merge.h",
                  "tools/structural_native.py", "tools/native_row_store.py", "tools/benchmark_structural_native.py")},
              "library_sha256": {"trie_archive": file_sha(ROOT / "mylib/lib/libtrie.a"),
                                 "pub_archive": file_sha(ROOT / "mylib/lib/libpub.a"),
                                 "structural": file_sha(cfg.structural_library), "rows": file_sha(cfg.native_library)},
              "settings": {k: v for k, v in vars(cfg).items() if not isinstance(v, Path)},
              "cost_assumptions": "shared existing CPU; cached public text; no network or paid services; cost/energy not measured",
              "limitations": ["Not original large-corpus paper reproduction or trained DeepSeek Engram evidence.",
                  "SGNS is vector-only; GRU task quality is a distinct trained architecture with different gradients.",
                  "Flat and DASTrie serving use one DATrie-based structural planner; flat is not a trie-free build baseline.",
                  "All timings use warm filesystem pages, a single worker, preallocated query buffers, ctypes included.",
                  "Fresh application cache and repeated warm working sets are separate; this does not measure continuous traffic.",
                  "Flat lookup uses sorted owned C++ strings; DASTrie serialized bytes are not runtime RSS measurements.",
                  "Artifact totals include serving key index + row forwarding + encoded table, exclude runtime/framework/backbone/character registry/JSON reports.",
                  "Known UTF-8 key-to-row mapping is required; hash bucket collisions cannot be treated as individual ngram ancestry.",
                  "Joint thresholds fixed before validation/test; no fine-tuning after compression.",
                  "No cold I/O, whole-model latency, GPU transfer, concurrent serving or full build peak RSS claim.",
                  "Corpus licensing is not independently established; raw text, keys and checkpoints are kept local."],
              "runs": [], "failures": []}
    if Path("/proc/cpuinfo").exists():
        report["cpu"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                             if line.startswith("model name"))
    orders, seeds = set(map(int, cfg.orders.split(","))), set(map(int, cfg.seeds.split(",")))
    split = text_splits(corpus.decode(), gru["settings"]["max_characters"])
    prepared = {}
    for category, training in (("sgns", sgns), ("gru", gru)):
        for run in training["runs"]:
            order, seed = run["max_order"], run["seed"]
            if order not in orders or seed not in seeds:
                continue
            try:
                kwargs = {}
                if category == "sgns":
                    if order not in prepared:
                        words, grams, *_ = prepare(corpus.decode(), order, sgns["settings"]["vocab_limit"],
                                                  sgns["settings"]["min_count"])
                        from tools.train_ngram_similarity import ngrams
                        ids = {key: i for i, key in enumerate(grams)}
                        keys = TracedKeys(enumerate(grams))
                        keys.query_trace = [ids[g] for word in words for g in ngrams(word, order)]
                        prepared[order] = keys
                    keys = prepared[order]
                    source = cfg.sgns_dir / f"order-{order}-seed-{seed}/trained-ngram.f32"
                    dim, row_hash = sgns["settings"]["dim"], run["learned_row_sha256"]
                else:
                    checkpoint = cfg.gru_dir / f"order-{order}-seed-{seed}/reference.pt"
                    if file_sha(checkpoint) != run["checkpoint_sha256"]:
                        raise ValueError("checkpoint provenance mismatch")
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    registry, dim = state["registry"], state["dim"]
                    model = CausalNgramLM(len(registry["characters"]) + 1, len(registry["grams"]), dim)
                    model.load_state_dict(state["model_state"])
                    model.eval()
                    raw = model.memory.weight.detach().numpy().astype("<f4").tobytes()
                    source = cfg.output_dir / f"gru-{order}-{seed}.f32"
                    source.write_bytes(raw)
                    row_hash = sha(raw)
                    keys = {i+1: key for i, key in enumerate(registry["grams"])}
                    kwargs = {"model": model, "segments": {name: encode_segment(text, registry)
                              for name, text in split.items() if name != "train"}}
                result = run_dataset(category, order, seed, source, keys, dim, row_hash, cfg, **kwargs)
                if category == "gru":
                    result["checkpoint_sha256"] = run["checkpoint_sha256"]
                    result["registry_sha256"] = run["registry_sha256"]
                report["runs"].append(result)
                print(json.dumps({"category": category, "order": order, "seed": seed,
                    "joint_merged_rows": result["variants"]["joint"]["merged_rows"],
                    "ordinary_bytes": result["variants"]["joint"]["four_way_artifact_bytes"]["flat_ordinary"],
                    "compact_bytes": result["variants"]["joint"]["four_way_artifact_bytes"]["datrie_compact"]}), flush=True)
            except Exception as error:
                report["failures"].append({"category": category, "order": order, "seed": seed,
                                            "type": type(error).__name__, "error": str(error)})
                print(json.dumps(report["failures"][-1]), flush=True)
            (cfg.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit("failed runs preserved in report.json")


if __name__ == "__main__":
    main()
