"""Plot checked-in CPU query evidence; no projected accelerator points."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text())["tables"]
    rows = sorted((r for r in rows if r["architecture"] == "epyc" and r["chunk_rows"] == 8192
                   and r["budget"] == 4194304), key=lambda r: r["order"])
    if [r["order"] for r in rows] != [4, 8, 12]:
        raise ValueError("expected the three measured EPYC orders")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x, width = np.arange(3), .24
    modes = [("before", "Old blocks: 64 entries", "#697386"),
             ("blocks", "New blocks", "#3677c4"), ("rows", "New queried rows", "#12877a")]
    for index, (key, label, color) in enumerate(modes):
        for panel, ax in enumerate(axes):
            values = [r["variants"][key]["empty_cache_first_trace"]["samples_ms"][0] if panel == 0
                      else r["variants"][key]["p50_ms"] for r in rows]
            bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
            ax.bar_label(bars, labels=[f"{v:.3f}" if v < 1 else f"{v:.2f}" for v in values], padding=3, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, ["Order 4", "Order 8", "Order 12"])
        ax.set_ylabel("CPU query latency (ms)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_title("First query after cache reset")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.15)
    axes[1].set_title("Warm replay median (log scale)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(.025, 120)
    fig.suptitle("Native C query cache · AMD EPYC 7742", fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .92), ncol=3, frameon=False)
    fig.text(.5, .045, "4 MiB payload budget · 512 positions · 8,192-row C chunks · eight original held-out windows", ha="center", fontsize=9)
    fig.text(.5, .012, "80 warm samples per case; repeated working set fits row cache. CPU only; no GPU/model speed claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .83))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(args.output_dir / ("cpu-query-latency." + ext), dpi=160)
    provenance = {"command": shlex.join(sys.orig_argv), "matplotlib": matplotlib.__version__,
                  "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                  "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "plot-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
