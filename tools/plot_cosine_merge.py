"""Render threshold/storage and measured-error curves from aggregate evidence.

Optional plotting dependency: matplotlib. No corpus or trained weights needed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.report.read_bytes()
    report = json.loads(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    colors = {4: "#377eb8", 8: "#4daf4a", 12: "#9b42aa"}
    thresholds = sorted(report["thresholds"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for order, color in colors.items():
        for policy, style, marker in [("cosine_and_norm_0_99", "-", "o"), ("cosine_only", "--", "s")]:
            groups = [[r for r in report["runs"] if r["max_order"] == order and r["policy"] == policy
                       and r["threshold"] == tau] for tau in thresholds]
            factors = np.asarray([[r["compression_factor_raw_to_artifact"] for r in group] for group in groups])
            errors = np.asarray([[r["word_ngram_component_fidelity"]["relative_l2_mean"] * 100 for r in group] for group in groups])
            label = f"order {order}, " + ("cosine + norm" if style == "-" else "cosine only")
            indices = np.arange(len(thresholds))
            axes[0].plot(indices, factors.mean(1), color=color, ls=style, marker=marker, label=label)
            axes[0].fill_between(indices, factors.min(1), factors.max(1), color=color, alpha=0.08)
            axes[1].plot(errors.mean(1), factors.mean(1), color=color, ls=style, marker=marker, label=label)
            for group in groups:
                axes[1].scatter([r["word_ngram_component_fidelity"]["relative_l2_mean"] * 100 for r in group],
                                [r["compression_factor_raw_to_artifact"] for r in group], color=color, s=9, alpha=0.28)
            if order == 12 and policy == "cosine_and_norm_0_99":
                for tau in (0.99, 0.98, 0.90):
                    i = thresholds.index(tau)
                    axes[1].annotate(f"{tau:.2f}", (errors.mean(1)[i], factors.mean(1)[i]),
                                     xytext={0.99: (9, -16), 0.98: (-16, 15), 0.90: (26, 0)}[tau],
                                     textcoords="offset points", fontsize=9,
                                     arrowprops={"arrowstyle": "-", "color": "#777", "lw": 0.6})
    for ax in axes:
        ax.set_ylabel("Raw n-gram bytes / actual packed bytes (×)")
        ax.grid(alpha=0.17)
    axes[0].set_xticks(np.arange(len(thresholds)), [f"{x:g}" for x in thresholds])
    axes[0].set_xlabel("Minimum cosine (looser acceptance →)")
    axes[0].set_title("Actual storage across acceptance thresholds")
    axes[1].set_xlabel("Mean relative L2 error of word n-gram mean (%)")
    axes[1].set_title("Measured distortion versus storage reduction")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Trained character n-grams: approximation / storage trade-off", fontsize=14, y=0.98)
    fig.text(0.5, 0.015, "3 seeds/order; lines are seed means, dots/ranges show variation. N-gram component only; no task-quality result.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output_dir / f"cosine_tradeoff.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    policies = [("cosine_only", "Cosine ≥0.99"), ("cosine_and_norm_0_99", "Cosine + norm ≥0.99"),
                ("norm_only_0_99", "Norm only ≥0.99")]
    for order, color in colors.items():
        for policy, label in policies:
            group = [r for r in report["runs"] if r["max_order"] == order and r["policy"] == policy
                     and (r["threshold"] == 0.99 or r["threshold"] is None)]
            mark = {"cosine_only": "s", "cosine_and_norm_0_99": "o", "norm_only_0_99": "X"}[policy]
            for ax, field in zip(axes, ("relative_l2_mean", "relative_l2_max"), strict=True):
                ax.scatter([r["word_ngram_component_fidelity"][field] * 100 for r in group],
                           [r["compression_factor_raw_to_artifact"] for r in group], color=color, marker=mark,
                           s=48, label=f"order {order}: {label}" if ax is axes[1] else None)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.17)
        ax.set_ylabel("Raw / packed bytes (×)")
    axes[0].set_xlabel("Mean relative word n-gram component error (%)")
    axes[1].set_xlabel("Maximum relative word n-gram component error (%)")
    axes[0].set_title("Mean errors can hide tail distortion")
    axes[1].set_title("Norm ratio alone does not protect direction")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5)
    fig.suptitle("0.99 rule comparison: all three seeds", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output_dir / f"norm_rule_comparison.{extension}", dpi=180)
    plt.close(fig)
    fields = ["max_order", "seed", "threshold", "policy", "compression_factor", "merged_fraction",
              "row_relative_l2_mean", "row_relative_l2_p95", "row_relative_l2_max",
              "word_relative_l2_mean", "word_relative_l2_p95", "word_relative_l2_max", "word_minimum_cosine"]
    with (args.output_dir / "curve.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow({"max_order": row["max_order"], "seed": row["seed"], "threshold": row["threshold"],
                             "policy": row["policy"], "compression_factor": row["compression_factor_raw_to_artifact"],
                             "merged_fraction": row["merged_fraction"],
                             **{"row_" + name: row["row_fidelity"][name] for name in ("relative_l2_mean", "relative_l2_p95", "relative_l2_max")},
                             **{"word_" + name: row["word_ngram_component_fidelity"][name] for name in ("relative_l2_mean", "relative_l2_p95", "relative_l2_max", "minimum_cosine")}})
    meta = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "exact_command": " ".join([sys.executable, *sys.argv]), "python": sys.version,
            "matplotlib": matplotlib.__version__, "numpy": np.__version__, "platform": platform.platform(),
            "report_sha256": hashlib.sha256(source).hexdigest()}
    (args.output_dir / "plot_provenance.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
