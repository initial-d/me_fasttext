"""Plot validation curves and the validation-selected held-out test audit.

Requires optional matplotlib. Across-seed SD and token-block bootstrap intervals
are different uncertainty summaries, explicitly labelled on the figure.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads(args.report.read_text())
    if report["failures"]:
        raise ValueError("failed runs are present; inspect raw evidence before plotting")
    orders = sorted({run["max_order"] for run in report["runs"]})
    thresholds = report["settings"]["thresholds"].split(",")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(orders))]
    for order_index, order in enumerate(orders):
        color = colors[order_index]
        runs = [r for r in report["runs"] if r["max_order"] == order]
        series = []
        for threshold in thresholds:
            variants = [next(v for v in r["variants"] if v["minimum_cosine"] == float(threshold)) for r in runs]
            series.append([[v["ngram_raw_to_artifact_factor"], v["validation_delta"]["loss_relative_change"] * 100,
                            v["validation_delta"]["discrimination_change"] * 100] for v in variants])
        means, std = np.mean(series, axis=1), np.std(series, axis=1)
        for metric, ax in enumerate((axes[0, 0], axes[0, 1], axes[1, 0])):
            ax.errorbar(range(len(thresholds)), means[:, metric], yerr=std[:, metric], fmt="o-",
                        capsize=3, color=color, label=f"max order {order} (n={len(runs)})")
        for i, run in enumerate(runs):
            selection = run["selection"]
            value = selection["test_delta"]["loss_relative_change"] * 100
            lo, hi = np.asarray(selection["test_paired_token_block_bootstrap"]["loss_relative_change_95_percent_interval"]) * 100
            # Draw interval independently: the bootstrap median need not equal
            # the observed point, and intervals need not bracket that point.
            x = orders.index(order) + (i - (len(runs) - 1) / 2) * .12
            axes[1, 1].vlines(x, lo, hi, color=color)
            axes[1, 1].plot(x, value, "o", color=color, markersize=5)
    axes[0, 0].set_title("Native n-gram artifact compression (mean +/- seed SD)")
    axes[0, 0].set_ylabel("Raw n-gram bytes / artifact bytes")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Validation SGNS loss change (mean +/- seed SD)")
    axes[0, 1].set_ylabel("Relative loss change (%)")
    axes[1, 0].set_title("Validation context discrimination (mean +/- seed SD)")
    axes[1, 0].set_ylabel("Change (percentage points)")
    axes[1, 1].set_title("Test: validation-selected variant only")
    axes[1, 1].set_ylabel("Relative SGNS loss change (%)")
    axes[1, 1].set_xlabel("Maximum n-gram order (one point per seed)")
    axes[1, 1].set_xticks(range(len(orders)), labels=orders)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(range(len(thresholds)), labels=[f"{float(t):.0%}" for t in thresholds])
        ax.set_xlabel(f"Minimum direct cosine (norm ratio >= {report['settings']['min_norm_ratio']:.0%})")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axhline(0, color="grey", linewidth=.7, linestyle="--")
    for ax in axes.flat:
        ax.grid(alpha=.15)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    cfg = report["settings"]
    fig.suptitle("Train-only SGNS / contiguous 80-10-10 public-text split / full word + n-gram input\n"
                 f"Selection budgets: +{cfg['max_loss_increase']:.1%} validation loss, -{cfg['max_discrimination_drop'] * 100:g} pp discrimination; test bars: paired token-block 95% intervals\n"
                 "Small English reference; not Engram or downstream LM quality", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
