"""Audit test generalization of validation budgets, and plot existing evidence.

Never selects a new test threshold. Raw training reports remain unchanged.
Plotting requires optional matplotlib; JSON audit does not.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np


def analyze(report):
    if report["failures"]:
        raise ValueError("failed runs are present; inspect the raw evidence")
    result = {"category": "derived audit of validation-selected native causal-LM compression",
              "models": [], "test_budget_exceedances": [],
              "rule": "audit previously chosen candidates on test; no threshold reselection"}
    for run in report["runs"]:
        selection = run["selection"]
        variant = next(v for v in run["variants"] if v["name"] == selection["name"])
        delta = selection["test_delta"]
        checks = {"perplexity": delta["perplexity_relative_change"] <= selection["validation_ppl_budget"],
                  "mean_prediction_kl": delta["mean_prediction_kl_nats"] <= selection["validation_mean_kl_budget"],
                  "top1_flip": delta["top1_changed_fraction"] <= selection["validation_top1_flip_budget"]}
        row = {"max_order": run["max_order"], "seed": run["seed"], "chosen_variant": selection["name"],
               "memory_raw_to_artifact_factor": variant["memory_raw_to_artifact_factor"],
               "full_parameter_factor": variant["full_parameter_payload_to_compressed_factor"],
               "parameter_and_registry_factor": variant["parameter_and_registry_to_compressed_factor"],
               "baseline_test": run["baseline"]["test"], "selected_test": selection["test"],
               "test_delta": delta, "test_paired_block_interval": selection["test_paired_block_interval"],
               "test_budget_checks": checks, "test_all_budgets_passed": all(checks.values()),
               "test_without_memory_perplexity": run["inference_memory_ablation"]["test"]["metrics"]["character_perplexity"]}
        result["models"].append(row)
        if not all(checks.values()):
            result["test_budget_exceedances"].append({"max_order": row["max_order"], "seed": row["seed"],
                                                      "metrics": [k for k, v in checks.items() if not v]})
    result["models_count"] = len(result["models"])
    result["test_models_passing_all_validation_budgets"] = sum(m["test_all_budgets_passed"] for m in result["models"])
    return result


def plot(report, audit, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orders = sorted({r["max_order"] for r in report["runs"]})
    thresholds = [float(v) for v in report["settings"]["thresholds"].split(",")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(orders))]
    for order_index, order in enumerate(orders):
        color = colors[order_index]
        runs = [r for r in report["runs"] if r["max_order"] == order]
        points = []
        for threshold in thresholds:
            variants = [next(v for v in r["variants"] if v["minimum_cosine"] == threshold) for r in runs]
            points.append([[v["memory_raw_to_artifact_factor"], v["validation_delta"]["mean_prediction_kl_nats"]]
                           for v in variants])
        mean, std = np.mean(points, axis=1), np.std(points, axis=1)
        for metric, ax in enumerate(axes[0]):
            ax.errorbar(range(len(thresholds)), mean[:, metric], yerr=std[:, metric],
                        fmt="o-", color=color, capsize=3, label=f"max order {order} (n={len(runs)})")
        selected = [m for m in audit["models"] if m["max_order"] == order]
        for seed_index, model in enumerate(selected):
            x = order_index + (seed_index - (len(selected) - 1) / 2) * .12
            lo, hi = np.asarray(model["test_paired_block_interval"]["perplexity_relative_change_95_percent_interval"]) * 100
            axes[1, 0].vlines(x, lo, hi, color=color)
            axes[1, 0].plot(x, model["test_delta"]["perplexity_relative_change"] * 100, "o",
                            color=color, markeredgecolor="black" if model["test_all_budgets_passed"] else "red",
                            markeredgewidth=1.6, markersize=6)
        metrics = [("Original", "#657b83", [m["baseline_test"]["character_perplexity"] for m in selected]),
                   ("Native selected", "#2c9a71", [m["selected_test"]["character_perplexity"] for m in selected]),
                   ("Memory off at inference", "#c45c42", [m["test_without_memory_perplexity"] for m in selected])]
        for i, (label, fill, values) in enumerate(metrics):
            axes[1, 1].bar(order_index + (i - 1) * .23, np.mean(values), width=.23, color=fill,
                           yerr=np.std(values), capsize=3, label=label if order_index == 0 else None)
    cfg = report["settings"]
    axes[0, 0].set_title("Native memory artifact ratio (mean +/- seed SD)")
    axes[0, 0].set_ylabel("Original memory bytes / artifact bytes")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Validation prediction KL (mean +/- seed SD)")
    axes[0, 1].set_ylabel("Mean KL (nats)")
    axes[0, 1].axhline(cfg["max_kl"], color="grey", linestyle="--", label="Validation KL budget")
    axes[0, 1].legend(fontsize=8)
    for ax in axes[0]:
        ax.set_xticks(range(len(thresholds)), labels=[f"{t:.0%}" for t in thresholds])
        ax.set_xlabel(f"Minimum direct cosine (norm ratio >= {cfg['min_norm_ratio']:.0%})")
    axes[1, 0].set_title("Selected test PPL change / paired-block 95% interval")
    axes[1, 0].set_ylabel("Character perplexity change (%)")
    axes[1, 0].axhline(0, color="grey", linewidth=.7, linestyle="--")
    axes[1, 0].text(.02, .97, "Red edge: test KL budget exceeded", transform=axes[1, 0].transAxes,
                    va="top", fontsize=8, color="#ad3434")
    axes[1, 1].set_title("Inference memory ablation (not a retrained baseline)")
    axes[1, 1].set_ylabel("Test character perplexity")
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xticks(range(len(orders)), labels=orders)
        ax.set_xlabel("Maximum n-gram order")
    for ax in axes.flat:
        ax.grid(alpha=.15, axis="y")
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Trained causal character GRU + exact suffix memory / train-only registry / all test targets after burn-in\n"
                 f"Native gather in forward; validation budgets: PPL +{cfg['max_ppl_increase']:.1%}, KL {cfg['max_kl']:g}, top-1 flips {cfg['max_top1_flip']:.0%}\n"
                 "Small public-text reference; not trained Engram or a general LM-quality guarantee", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--plot-out", type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    audit = analyze(report)
    audit["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    audit["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    audit["analysis_command"] = shlex.join([sys.executable, *sys.argv])
    audit["python"] = sys.version
    audit["numpy"] = np.__version__
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2) + "\n")
    if args.plot_out:
        plot(report, audit, args.plot_out)
    print(json.dumps({"models": audit["models_count"], "test_budget_exceedances": audit["test_budget_exceedances"]}))


if __name__ == "__main__":
    main()
