"""Summarize matched encoded baselines without reselecting on test outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ENCODINGS = ("fp32", "fp16", "int8")
BASELINES = ("raw", "identity", "layout", "dedup")
SIZE = "backend_with_key_registry_bytes"


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values)}


def analyze(report):
    if report["failures"] or not report["runs"]:
        raise ValueError("complete successful study required; retain failures in raw report")
    models, grouped, ids = [], {}, set()
    for run in report["runs"]:
        identity = (run["category"], run["max_order"], run["seed"])
        if identity in ids:
            raise ValueError("duplicate trained model")
        ids.add(identity)
        variants = {v["name"]: v for v in run["variants"]}
        raw32 = variants["fp32-raw"]
        for encoding in ENCODINGS:
            raw = variants[f"{encoding}-raw"]
            best = min((variants[f"{encoding}-{p}"] for p in BASELINES), key=lambda v: (v[SIZE], v["name"]))
            joint = variants[f"{encoding}-joint-0.99"]
            item = {"category": identity[0], "max_order": identity[1], "seed": identity[2], "encoding": encoding,
                    "raw_encoded_backend_with_keys_bytes": raw[SIZE],
                    "best_encoded_baseline": best["name"], "best_encoded_baseline_with_keys_bytes": best[SIZE],
                    "joint_099_backend_with_keys_bytes": joint[SIZE],
                    "joint_099_added_saving_vs_best_encoded_baseline": 1 - joint[SIZE] / best[SIZE],
                    "joint_099_saving_vs_raw_encoded": 1 - joint[SIZE] / raw[SIZE],
                    "joint_099_factor_vs_raw_float32_with_keys": raw32[SIZE] / joint[SIZE],
                    "joint_099_weighted_relative_frobenius": joint["fidelity_vs_float32"]["weighted_relative_frobenius"],
                    "joint_099_added_weighted_relative_frobenius": joint["added_fidelity_vs_quantization_only"]["weighted_relative_frobenius"],
                    "joint_099_max_relative_l2": joint["fidelity_vs_float32"]["relative_l2_max"],
                    "joint_099_warm_query_ratio_vs_raw_encoded": joint["queries"]["warm_p50_ms"] / raw["queries"]["warm_p50_ms"]}
            datrie_size = "backend_with_datrie_key_index_bytes"
            if joint.get(datrie_size) is not None:
                item["joint_099_added_saving_vs_best_encoded_baseline_with_datrie"] = 1 - joint[datrie_size] / best[datrie_size]
                item["joint_099_factor_vs_raw_float32_with_datrie"] = raw32[datrie_size] / joint[datrie_size]
            selected = run.get("selection", {}).get(encoding)
            if selected:
                # Preserve the validation choice, including failed test budgets.
                item["validation_selection"] = selected
                if selected["feasible"]:
                    chosen = variants[selected["name"]]
                    item["selected_added_saving_vs_best_encoded_baseline"] = 1 - chosen[SIZE] / best[SIZE]
                    item["selected_factor_vs_raw_float32_with_keys"] = raw32[SIZE] / chosen[SIZE]
                    item["selected_full_model_factor_vs_raw_float32"] = raw32["full_model_serving_artifact_bytes"] / chosen["full_model_serving_artifact_bytes"]
            models.append(item)
            grouped.setdefault((identity[0], identity[1], encoding), []).append(item)
    groups = []
    metrics = [k for k in models[0] if k.startswith("joint_099_") and k != "joint_099_backend_with_keys_bytes"]
    for (category, order, encoding), rows in sorted(grouped.items()):
        group = {"category": category, "max_order": order, "encoding": encoding,
                 "seeds": [r["seed"] for r in rows],
                 "joint_099": {key: distribution([r[key] for r in rows]) for key in metrics}}
        selected = [r for r in rows if r.get("validation_selection", {}).get("feasible")]
        if selected:
            group["validation_selected"] = {
                "feasible_models": len(selected),
                "test_all_budgets_passed_models": sum(r["validation_selection"]["test_all_budgets_passed"] for r in selected),
                "selected_names": [r["validation_selection"]["name"] for r in selected],
                "added_saving_vs_best_encoded_baseline": distribution([r["selected_added_saving_vs_best_encoded_baseline"] for r in selected]),
                "factor_vs_raw_float32_with_keys": distribution([r["selected_factor_vs_raw_float32_with_keys"] for r in selected]),
                "full_model_factor_vs_raw_float32": distribution([r["selected_full_model_factor_vs_raw_float32"] for r in selected])}
        groups.append(group)
    failed = [{"max_order": r["max_order"], "seed": r["seed"], "encoding": r["encoding"],
               "name": r["validation_selection"]["name"], "delta": r["validation_selection"]["test_delta"]}
              for r in models if r.get("validation_selection", {}).get("feasible")
              and not r["validation_selection"]["test_all_budgets_passed"]]
    return {"source_commit": report["commit"], "planner": report["settings"]["planner"],
            "trained_models": len(report["runs"]), "variant_count": sum(len(r["variants"]) for r in report["runs"]),
            "storage_denominator": "actual encoded backend + codec metadata + common keys JSON; whole GRU serving artifacts also include unchanged serialized backbone/registry",
            "baseline_rule": "best of raw mmap / native identity / native layout / exact encoded-byte dedup, matched encoding; structural candidates compared against this lower byte count",
            "error_rule": "final decoded vectors vs original FP32; additional error separately vs quantization-only; frequency weights from actual query traces",
            "uncertainty": "three-seed min/mean/max are descriptive ranges, not confidence intervals or cross-corpus evidence",
            "test_selection": "retain per-encoding validation choices; no test-based replacement",
            "test_budget_exceedances": failed, "models": models, "groups": groups}


def plots(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    orders = sorted({r["max_order"] for r in report["runs"]})
    categories = sorted({r["category"] for r in report["runs"]}, reverse=True)
    names = ["fp16-raw", "int8-raw", "fp32-joint-0.99", "fp16-joint-0.99", "int8-joint-0.99"]
    labels = ["FP16", "INT8", "FP32 + S", "FP16 + S", "INT8 + S"]
    colors = ["#5587b6", "#ed9c42", "#599268", "#83b4d7", "#f7bf7a"]
    fig, axes = plt.subplots(len(categories), len(orders), figsize=(12, 6), squeeze=False)
    for row, category in enumerate(categories):
        for col, order in enumerate(orders):
            ax = axes[row, col]
            runs = [r for r in report["runs"] if r["category"] == category and r["max_order"] == order]
            points = {name: [] for name in names}
            for run in runs:
                v = {v["name"]: v for v in run["variants"]}
                for name in names:
                    points[name].append(v["fp32-raw"][SIZE] / v[name][SIZE])
            means = [statistics.mean(points[n]) for n in names]
            errors = [[m - min(points[n]) for m, n in zip(means, names)],
                      [max(points[n]) - m for m, n in zip(means, names)]]
            ax.bar(range(len(names)), means, color=colors, yerr=errors, capsize=3)
            ax.axhline(1, color="gray", linewidth=.8)
            ax.set_xticks(range(len(names)), labels, rotation=25, ha="right")
            ax.set_title(f"{category.upper()}, order {order} (n={len(runs)})")
            ax.set_ylabel("FP32 raw / stored bytes")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Complete backend + common keys; S = cosine/norm >= 0.99 on decoded encoded rows\nSeed ranges; fixed threshold, no task-quality guarantee from this plot", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .90))
    for ext in ("svg", "png"):
        fig.savefig(directory / f"storage-controls.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(len(categories), len(orders), figsize=(12, 6), squeeze=False)
    for row, category in enumerate(categories):
        for col, order in enumerate(orders):
            ax = axes[row, col]
            runs = [r for r in report["runs"] if r["category"] == category and r["max_order"] == order]
            for encoding, color in zip(ENCODINGS, ("#599268", "#5587b6", "#ed9c42")):
                points = []
                policies = ["raw"] + [f"joint-{t:g}" for t in report["settings"]["threshold_values"]]
                for policy in policies:
                    ratios, errors = [], []
                    for run in runs:
                        v = {v["name"]: v for v in run["variants"]}
                        item = v[f"{encoding}-{policy}"]
                        ratios.append(v["fp32-raw"][SIZE] / item[SIZE])
                        errors.append(item["fidelity_vs_float32"]["weighted_relative_frobenius"] * 100)
                    points.append((statistics.mean(errors), statistics.mean(ratios)))
                x, y = np.array(points).T
                ax.plot(x, y, "o-", color=color, markersize=4, label=encoding.upper())
                for (xx, yy), policy in zip(points, policies):
                    if policy in ("joint-0.99", "joint-0.9"):
                        ax.annotate(policy.split("-")[1], (xx, yy), fontsize=7, xytext=(2, 4), textcoords="offset points")
            ax.set_title(f"{category.upper()}, order {order}")
            ax.set_xlabel("Query-weighted vector error (%)")
            ax.set_ylabel("FP32 raw / stored bytes")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("Final error vs original FP32; mean of trained seeds\nRaw controls + cosine sweep 0.99, 0.98, 0.97, 0.95, 0.90; norm ratio fixed at 0.99", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .90))
    for ext in ("svg", "png"):
        fig.savefig(directory / f"rate-distortion.{ext}", dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    summary = analyze(report)
    summary["raw_report_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plots:
        plots(report, args.output_dir)


if __name__ == "__main__":
    main()
