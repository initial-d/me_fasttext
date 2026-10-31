"""Same-checkpoint CPU controls for quantization followed by structural sharing.

Reuse cached SGNS rows / GRU checkpoints; never retrain, download or tune on
test. Native planner is the production DATrie path. The explicit reference
planner is a different portable greedy prefix/suffix control, not DATrie evidence.
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

from tools.benchmark_cosine_merge import fidelity, structural_merge
from tools.benchmark_causal_lm_compression import text_splits, paired_ppl_interval
from tools.causal_ngram_lm import CausalNgramLM, encode_segment, evaluate, metric_summary, prediction_delta
from tools.quantized_memory import encode_rows, decode_rows, pack_quantized, QuantizedRows
from tools.structural_native import StructuralPlan
from tools.train_ngram_similarity import prepare, ngrams

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def structural_maps(rows, keys, cfg, threshold):
    if cfg.planner == "native":
        with StructuralPlan(rows.tobytes(), rows.shape[1], keys, cfg.structural_library,
                            mode="layout" if threshold is None else "joint",
                            cosine=threshold or .99, norm_ratio=cfg.norm_ratio) as plan:
            return np.asarray(plan.representatives), plan.row_order
    keyed = list(keys)
    representatives = np.arange(len(rows), dtype=np.int64)
    if threshold is not None:
        partial = structural_merge(rows[keyed], [keys[i] for i in keyed], threshold, cfg.norm_ratio)
        representatives[keyed] = np.asarray(keyed)[partial]
    key_set = set(keys)
    order = ([i for i in range(len(rows)) if i not in key_set]
             + [i for i in sorted(keys, key=lambda i: keys[i]) if representatives[i] == i]
             + [i for i, r in enumerate(representatives) if i != r])
    if len(order) != len(rows) or len(set(order)) != len(rows) or not np.array_equal(representatives[representatives], representatives):
        raise AssertionError("invalid original-ID structural map")
    return representatives, order


class ReaderEmbedding(nn.Module):
    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    def forward(self, ids):
        values = self.reader.gather(ids.detach().cpu().numpy().reshape(-1))
        return torch.from_numpy(values).reshape(*ids.shape, self.reader.dim)


def score_bundle(model, original_memory, directory, segment, cfg):
    try:
        with QuantizedRows(directory, cfg.native_library, cache_bytes=cfg.cache_bytes) as reader:
            model.memory = ReaderEmbedding(reader)
            return evaluate(model, segment, cfg.burn_in, cfg.eval_chunk)
    finally:
        model.memory = original_memory


def choose_validation(variants, encoding, cfg):
    feasible = [v for v in variants if v["encoding"] == encoding
                and v["validation_delta"]["perplexity_relative_change"] <= cfg.max_ppl_increase
                and v["validation_delta"]["mean_prediction_kl_nats"] <= cfg.max_kl
                and v["validation_delta"]["top1_changed_fraction"] <= cfg.max_top1_flip]
    return min(feasible, key=lambda v: (v["backend_with_key_registry_bytes"], v["name"])) if feasible else None


def query_metrics(directory, expected, query, cfg):
    with QuantizedRows(directory, cfg.native_library, cache_bytes=cfg.cache_bytes) as reader:
        start = time.perf_counter_ns()
        output = reader.gather(query)
        first = (time.perf_counter_ns() - start) / 1e6
        if output.tobytes() != expected[query].tobytes():
            raise AssertionError("first gather changed encoded representative output")
        for _ in range(3):
            reader.gather(query)
        samples = []
        for _ in range(cfg.repeat):
            start = time.perf_counter_ns()
            reader.gather(query)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return {"first_empty_application_cache_ms": first,
                "warm_p50_ms": float(np.median(samples)), "warm_p95_ms": float(np.percentile(samples, 95)),
                "warm_p99_ms": float(np.percentile(samples, 99)), "samples_ms": samples,
                "warm_rows_per_second": len(query) / (np.median(samples) / 1000)}


def compare(category, order, seed, rows, keys, trace, cfg, model=None, segments=None):
    directory = cfg.output_dir / f"{category}-order-{order}-seed-{seed}"
    directory.mkdir()
    registry_path = directory / "keys.json"
    registry_path.write_text(json.dumps(keys, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    common = registry_path.stat().st_size
    index_bytes = None
    if cfg.planner == "native":
        with StructuralPlan(rows.tobytes(), rows.shape[1], keys, cfg.structural_library, mode="layout") as plan:
            index_path = directory / "keys.das"
            plan.save_index(index_path)
            index_bytes = index_path.stat().st_size
            if plan.lookup(list(keys.values()), index="datrie") != list(keys):
                raise AssertionError("DATrie key index changed original IDs")
    other_bytes = 0
    original_memory = model.memory if model is not None else None
    baseline = ({name: evaluate(model, segment, cfg.burn_in, cfg.eval_chunk) for name, segment in segments.items()}
                if model is not None else {})
    if model is not None:
        # Actual serialized unchanged model state/registry, not an estimated payload.
        backbone = {k: v for k, v in model.state_dict().items() if k != "memory.weight"}
        torch.save({"state": backbone, "registry": cfg.current_registry, "dim": rows.shape[1]}, directory / "backbone.pt")
        other_bytes = (directory / "backbone.pt").stat().st_size
    trace = np.asarray(trace, dtype=np.int64)
    query = trace[:cfg.query_rows]
    mass = np.bincount(trace, minlength=len(rows))
    entry = {"category": category, "max_order": order, "seed": seed, "rows": len(rows), "dim": rows.shape[1],
             "source_float32_sha256": sha(rows.tobytes()), "source_float32_bytes": rows.nbytes,
             "key_registry_bytes": common, "unchanged_backbone_artifact_bytes": other_bytes,
             "alternative_datrie_key_index_bytes": index_bytes,
             "query_rows": len(query), "query_ids_sha256": sha(query.astype("<u8").tobytes()),
             "frequency_trace_sha256": sha(trace.astype("<u8").tobytes()), "variants": [],
             "baseline_quality": {name: metric_summary(score) for name, score in baseline.items()}}
    quantized_scores = {}
    for encoding in ("fp32", "fp16", "int8"):
        encoded = encode_rows(rows, encoding)
        rounded = decode_rows(encoded, rows.shape[1], encoding)
        _, layout = structural_maps(rounded, keys, cfg, None)
        specs = [("raw", False, False, None), ("identity", True, False, None),
                 ("layout", True, False, None), ("dedup", True, True, None)]
        specs += [(f"joint-{t:g}", True, False, t) for t in cfg.threshold_values]
        for label, native, dedup, threshold in specs:
            name = f"{encoding}-{label}"
            mapping, order_map = np.arange(len(rows)), None
            if label == "layout":
                order_map = layout
            elif threshold is not None:
                mapping, order_map = structural_maps(rounded, keys, cfg, threshold)
            target = directory / name
            start = time.perf_counter()
            metadata = pack_quantized(rows, target, encoding, native=native,
                representatives=mapping.tolist() if threshold is not None else None,
                row_order=order_map, deduplicate=dedup, max_block_bytes=cfg.max_block_bytes)
            export_seconds = time.perf_counter() - start
            expected = rounded[mapping]
            with QuantizedRows(target, cfg.native_library) as reader:
                for begin in range(0, len(rows), 1024):
                    ids = np.arange(begin, min(begin + 1024, len(rows)), dtype=np.int64)
                    if reader.encoded_rows(ids).tobytes() != encoded[mapping[ids]].tobytes():
                        raise AssertionError("encoded original-ID forwarding mismatch")
                    if reader.gather(ids).tobytes() != expected[ids].tobytes():
                        raise AssertionError("decoded row mismatch")
            serving_bytes = sum(p.stat().st_size for p in target.iterdir())
            variant = {"name": name, "encoding": encoding, "policy": label, "minimum_cosine": threshold,
                       "minimum_norm_ratio": cfg.norm_ratio if threshold is not None else None,
                       "metadata": metadata, "export_seconds": export_seconds,
                       "representatives_sha256": sha(mapping.astype("<u8").tobytes()),
                       "structurally_merged_rows": int(np.sum(mapping != np.arange(len(rows)))),
                       "backend_artifact_bytes": serving_bytes,
                       "backend_with_key_registry_bytes": serving_bytes + common,
                       "backend_with_datrie_key_index_bytes": serving_bytes + index_bytes if index_bytes is not None else None,
                       "full_model_serving_artifact_bytes": serving_bytes + common + other_bytes if model is not None else None,
                       "fidelity_vs_float32": fidelity(rows, expected, mass),
                       "added_fidelity_vs_quantization_only": fidelity(rounded, expected, mass),
                       "all_encoded_rows_and_original_ids_verified": True,
                       "queries": query_metrics(target, expected, query, cfg)}
            if model is not None:
                scores = score_bundle(model, original_memory, target, segments["validation"], cfg)
                if label == "raw":
                    quantized_scores[encoding] = scores
                elif threshold is None and not torch.equal(scores["logits"], quantized_scores[encoding]["logits"]):
                    raise AssertionError("ordinary encoded storage changed model logits")
                variant.update({"validation": metric_summary(scores),
                                "validation_delta": prediction_delta(scores, baseline["validation"]),
                                "added_validation_delta_vs_quantization_only": prediction_delta(scores, quantized_scores[encoding])})
            entry["variants"].append(variant)
        print(json.dumps({"category": category, "order": order, "seed": seed, "encoding": encoding,
                          "completed_variants": len(entry["variants"])}), flush=True)
    if model is not None:
        entry["selection"] = {}
        # All choices are locked from validation before querying test candidates.
        selected = {encoding: choose_validation(entry["variants"], encoding, cfg) for encoding in ("fp32", "fp16", "int8")}
        for encoding, variant in selected.items():
            if variant is None:
                entry["selection"][encoding] = {"feasible": False}
                continue
            scores = score_bundle(model, original_memory, directory / variant["name"], segments["test"], cfg)
            delta = prediction_delta(scores, baseline["test"])
            entry["selection"][encoding] = {"feasible": True, "name": variant["name"],
                "backend_with_key_registry_bytes": variant["backend_with_key_registry_bytes"],
                "test": metric_summary(scores), "test_delta": delta,
                "test_paired_block_interval": paired_ppl_interval(scores, baseline["test"], seed=cfg.eval_seed),
                "test_all_budgets_passed": (delta["perplexity_relative_change"] <= cfg.max_ppl_increase
                    and delta["mean_prediction_kl_nats"] <= cfg.max_kl and delta["top1_changed_fraction"] <= cfg.max_top1_flip)}
        entry["fixed_quantization_test_controls"] = {}
        for encoding in ("fp16", "int8"):
            scores = score_bundle(model, original_memory, directory / f"{encoding}-raw", segments["test"], cfg)
            entry["fixed_quantization_test_controls"][encoding] = {"metrics": metric_summary(scores),
                "delta": prediction_delta(scores, baseline["test"])}
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sgns-dir", "gru-dir", "corpus", "native-library", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--planner", choices=("native", "reference"), default="native")
    p.add_argument("--structural-library", type=Path)
    p.add_argument("--orders", default="4,8,12")
    p.add_argument("--seeds", default="3,7,11")
    p.add_argument("--thresholds", default="0.99,0.98,0.97,0.95,0.90")
    p.add_argument("--norm-ratio", type=float, default=.99)
    p.add_argument("--max-block-bytes", type=int, default=8192)
    p.add_argument("--cache-bytes", type=int, default=4194304)
    p.add_argument("--query-rows", type=int, default=512)
    p.add_argument("--repeat", type=int, default=40)
    p.add_argument("--burn-in", type=int, default=64)
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--max-ppl-increase", type=float, default=.005)
    p.add_argument("--max-kl", type=float, default=.0001)
    p.add_argument("--max-top1-flip", type=float, default=.01)
    cfg = p.parse_args()
    orders, seeds = set(map(int, cfg.orders.split(","))), set(map(int, cfg.seeds.split(",")))
    cfg.threshold_values = list(map(float, cfg.thresholds.split(",")))
    if (not orders or not seeds or not cfg.threshold_values or min(orders) < 2 or
            min(cfg.query_rows, cfg.repeat, cfg.eval_chunk, cfg.max_block_bytes) < 1 or cfg.cache_bytes < 0
            or cfg.burn_in < max(orders) - 1 or not 0 < cfg.norm_ratio <= 1
            or any(not 0 < t <= 1 for t in cfg.threshold_values)
            or min(cfg.max_ppl_increase, cfg.max_kl, cfg.max_top1_flip) < 0
            or cfg.planner == "native" and cfg.structural_library is None):
        p.error("invalid dimensions, guards/budgets or missing native structural library")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    corpus = cfg.corpus.read_bytes()
    training = {kind: json.loads((path / "report.json").read_text()) for kind, path in
                (("sgns", cfg.sgns_dir), ("gru", cfg.gru_dir))}
    if any(t["corpus_sha256"] != sha(corpus) for t in training.values()):
        raise ValueError("cached corpus differs from training provenance")
    sources = ["tools/quantized_memory.py", "tools/benchmark_quantized_structural.py", "tools/benchmark_cosine_merge.py",
               "tools/structural_native.py", "tools/native_row_store.py", "tools/causal_ngram_lm.py", "native/meft_rows.c"]
    report = {"category": "cached public-text trained SGNS / causal GRU; CPU quantization + structural controls",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": shlex.join([sys.executable, "-m", "tools.benchmark_quantized_structural", *sys.argv[1:]]),
              "working_directory": str(ROOT), "python": sys.version, "pytorch": torch.__version__,
              "numpy": np.__version__, "platform": platform.platform(), "cpu": platform.processor(),
              "logical_cpus": os.cpu_count(), "device": "cpu", "gpu": "unused", "threads": 1, "interop_threads": 1,
              "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
              "source_sha256": {name: file_sha(ROOT / name) for name in sources},
              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
              "native_library_sha256": file_sha(cfg.native_library),
              "structural_library_sha256": file_sha(cfg.structural_library) if cfg.structural_library else None,
              "corpus_sha256": sha(corpus), "corpus_source": training["gru"]["corpus_source"],
              "training_report_sha256": {"sgns": file_sha(cfg.sgns_dir / "report.json"), "gru": file_sha(cfg.gru_dir / "report.json")},
              "settings": {k: v for k, v in vars(cfg).items() if not isinstance(v, Path)},
              "selection_rule": "minimum complete backend + common key registry bytes within all validation budgets, separately by encoding; freeze before test; no test reselection",
              "cost_assumptions": "existing cached weights/text; shared/local single CPU worker; no paid data or GPU; energy/currency cost unmeasured",
              "limitations": ["Small trained English SGNS/GRU references, not an official trained Engram checkpoint or paper-scale reproduction.",
                  "Reference planner is portable greedy prefix/suffix selection, not production DATrie; native planner is separately labeled.",
                  "Selection and primary storage totals use a common actual keys JSON; native runs also report a serialized DATrie alternative separately. No key lookup speed claim.",
                  "Full GRU bundle sums encoded backend, codec JSON, common keys JSON and actual serialized unchanged backbone/registry; excludes runtime/framework, logs and training checkpoint.",
                  "Row INT8 is symmetric nearest-even with float32 scale per row; no activation quantization or fine tuning.",
                  "Quantization occurs before planning; final error is measured against original float32 and quantization-only rows.",
                  "Filesystem pages are warm after row verification; first query has empty application cache, not disk-cold I/O.",
                  "Queries include Python/NumPy dequantization and allocations; no fused INT8 kernel, RSS, concurrent traffic or full-model latency claim.",
                  "One cached corpus, three seeds; SGNS reports vector distortion only, GRU reports character task quality.",
                  "Corpus license not independently established; raw text, keys, weights and serving bundles stay local."],
              "runs": [], "failures": []}
    if sys.platform == "darwin":
        report["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    elif Path("/proc/cpuinfo").exists():
        report["cpu"] = next(s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name"))
    prepared = {}
    split = text_splits(corpus.decode(), training["gru"]["settings"]["max_characters"])
    for kind, trained in training.items():
        for run in trained["runs"]:
            order, seed = run["max_order"], run["seed"]
            if order not in orders or seed not in seeds:
                continue
            try:
                kwargs = {}
                if kind == "sgns":
                    if order not in prepared:
                        words, grams, *_ = prepare(corpus.decode(), order, trained["settings"]["vocab_limit"], trained["settings"]["min_count"])
                        ids = {g: i for i, g in enumerate(grams)}
                        prepared[order] = (dict(enumerate(grams)), [ids[g] for w in words for g in ngrams(w, order)])
                    keys, trace = prepared[order]
                    source = cfg.sgns_dir / f"order-{order}-seed-{seed}/trained-ngram.f32"
                    if file_sha(source) != run["learned_row_sha256"]:
                        raise ValueError("SGNS source hash mismatch")
                    rows = np.frombuffer(source.read_bytes(), dtype="<f4").reshape(-1, trained["settings"]["dim"])
                    provenance = {"training_row_sha256": run["learned_row_sha256"]}
                else:
                    source = cfg.gru_dir / f"order-{order}-seed-{seed}/reference.pt"
                    if file_sha(source) != run["checkpoint_sha256"]:
                        raise ValueError("GRU checkpoint hash mismatch")
                    state = torch.load(source, map_location="cpu", weights_only=True)
                    registry = state["registry"]
                    model = CausalNgramLM(len(registry["characters"]) + 1, len(registry["grams"]), state["dim"])
                    model.load_state_dict(state["model_state"])
                    model.eval()
                    rows = model.memory.weight.detach().numpy().astype("<f4")
                    keys = {i + 1: g for i, g in enumerate(registry["grams"])}
                    segments = {name: encode_segment(text, registry) for name, text in split.items() if name != "train"}
                    trace = segments["validation"]["memory_ids"].numpy().reshape(-1)
                    cfg.current_registry = registry
                    kwargs = {"model": model, "segments": segments}
                    provenance = {"checkpoint_sha256": run["checkpoint_sha256"], "registry_sha256": run["registry_sha256"]}
                result = compare(kind, order, seed, rows, keys, trace, cfg, **kwargs)
                result.update(provenance)
                report["runs"].append(result)
            except Exception as error:
                report["failures"].append({"category": kind, "order": order, "seed": seed, "type": type(error).__name__, "error": str(error)})
                print(json.dumps(report["failures"][-1]), flush=True)
            (cfg.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    expected = sum(1 for t in training.values() for r in t["runs"] if r["max_order"] in orders and r["seed"] in seeds)
    if report["failures"] or len(report["runs"]) != expected or not expected:
        raise SystemExit("incomplete study; preserved failures in report.json")


if __name__ == "__main__":
    main()
