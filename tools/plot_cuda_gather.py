"""Plot completed CUDA gather reports; no projected points or model inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.evidence_dir / "summary.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    modes = [("resident", "Resident", "#187e70"), ("dense-offload", "Dense offload", "#287abb"),
             ("native", "Native offload", "#cf632e")]
    for column, order in enumerate((4, 8, 12)):
        for mode, label, color in modes:
            values = sorted((r for r in rows if r["order"] == order and r["mode"] == mode and
                             r["staging_rows"] == 8192), key=lambda r: r["positions"])
            x = [r["positions"] for r in values]
            axes[0, column].plot(x, [r["median_ms"] for r in values], "o-", color=color,
                                 label=label if mode == "resident" else label + " (8192)")
            axes[1, column].plot(x, [r["peak_gpu_mib"] for r in values],
                                 "^-" if mode == "native" else "o--", color=color, label=label)
        values = sorted((r for r in rows if r["order"] == order and r["mode"] == "native" and
                         r["staging_rows"] == 256), key=lambda r: r["positions"])
        axes[0, column].plot([r["positions"] for r in values], [r["median_ms"] for r in values],
                             "s--", color="#cf632e", alpha=.65, label="Native offload (256)")
        axes[0, column].set_title(f"Trained max order {order}, seed 3")
        for axis in axes[:, column]:
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 32, 128, 512], ["1", "32", "128", "512"])
            axis.grid(alpha=.15)
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylim(.08, 130)
        axes[1, column].set_xlabel("Chronological held-out positions per query")
        axes[1, column].set_ylim(0, 13)
    axes[0, 0].set_ylabel("Warm median gather latency (ms, log)")
    axes[1, 0].set_ylabel("Peak Torch GPU allocation (MiB)")
    axes[0, 2].legend(fontsize=8, loc="upper left")
    axes[1, 2].legend(fontsize=8, loc="upper left")
    fig.suptitle("A100: exact row outputs, memory versus query latency", fontsize=16)
    fig.text(.5, .93, "Three original trained character-GRU tables; 8 real windows/size, 3 warmups, 10 repeats", ha="center", fontsize=10)
    fig.text(.5, .045, "GPU values cover gather only: backbone, KV cache and driver/context are excluded. Dense/native curves overlap.", ha="center", fontsize=9)
    fig.text(.5, .017, "One GPU, one seed/corpus; other cards/CPU are shared. Native trades lower resident GPU memory for CPU query cost.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .91))
    for extension in ("svg", "png"):
        fig.savefig(args.evidence_dir / ("gather-curves." + extension), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
