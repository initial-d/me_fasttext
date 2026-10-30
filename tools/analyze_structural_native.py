"""Summarize the trained DATrie/compaction evidence and export scientific figures."""
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


def interval(values):
    return {"median": float(np.median(values)), "min": float(min(values)), "max": float(max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--lookup-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    cfg = parser.parse_args()
    memory, lookup = (json.loads(path.read_text()) for path in (cfg.memory_report, cfg.lookup_report))
    if memory["failures"] or len(memory["runs"]) != 18 or len(lookup["runs"]) != 18:
        raise ValueError("requires all 18 successful trained runs in both reports")
    if lookup["memory_report_sha256"] != hashlib.sha256(cfg.memory_report.read_bytes()).hexdigest():
        raise ValueError("lookup report references different memory evidence")
    lookups = {(r["category"], r["max_order"], r["seed"]): r for r in lookup["runs"]}
    groups = {}
    for category in ("sgns", "gru"):
        for order in (4, 8, 12):
            rows = [r for r in memory["runs"] if r["category"] == category and r["max_order"] == order]
            if sorted(r["seed"] for r in rows) != [3, 7, 11]:
                raise ValueError("missing or duplicate seed")
            for row in rows:
                key = (category, order, row["seed"])
                if lookups[key]["query_ids_sha256"] != row["query_ids_sha256"]:
                    raise ValueError("lookup trace mismatch")
            factors = ("flat_ordinary", "datrie_ordinary", "flat_compact", "datrie_compact")
            g = {"seeds": [3, 7, 11], "rows": rows[0]["rows"],
                "artifact_fraction_of_flat_ordinary": {factor: interval([
                    r["variants"]["joint"]["four_way_artifact_bytes"][factor] /
                    r["variants"]["joint"]["four_way_artifact_bytes"]["flat_ordinary"] for r in rows]) for factor in factors},
                "mark_compact_table_saving": interval([1 - r["variants"]["joint"]["metadata"]["artifact_bytes"] /
                    (r["variants"]["joint"]["four_way_artifact_bytes"]["datrie_ordinary"] -
                     r["variants"]["joint"]["index_bytes"]["datrie"]) for r in rows]),
                "total_saving_vs_flat_ordinary": interval([1 - r["variants"]["joint"]["four_way_artifact_bytes"]["datrie_compact"] /
                    r["variants"]["joint"]["four_way_artifact_bytes"]["flat_ordinary"] for r in rows]),
                "index_byte_saving": 1 - rows[0]["variants"]["layout"]["index_bytes"]["datrie"] /
                                         rows[0]["variants"]["layout"]["index_bytes"]["flat"],
                "joint_merged_rows": interval([r["variants"]["joint"]["merged_rows"] for r in rows]),
                "exact_merged_rows": [r["variants"]["exact"]["merged_rows"] for r in rows],
                "lookup_us": {method: interval([lookups[(category, order, r["seed"])]["lookup"][method]["p50_ms"] * 1000
                                                 for r in rows]) for method in ("flat", "hash", "datrie")},
                "datrie_lookup_speedup_vs_hash": interval([
                    lookups[(category, order, r["seed"])]["lookup"]["hash"]["p50_ms"] /
                    lookups[(category, order, r["seed"])]["lookup"]["datrie"]["p50_ms"] for r in rows]),
                "minimum_joint_cosine": min(r["variants"]["joint"]["fidelity"]["minimum_cosine"] for r in rows),
                "mean_relative_l2": interval([r["variants"]["joint"]["fidelity"]["relative_l2_mean"] for r in rows]),
                "max_relative_l2": max(r["variants"]["joint"]["fidelity"]["relative_l2_max"] for r in rows),
                "query_numeric_row_cache_ms": {stage: {variant: interval([
                    (r["ordinary_queries"] if variant == "ordinary" else r["variants"]["joint"]["queries"])
                    ["rows"]["4194304"][stage]["ms" if stage == "first_empty_application_cache" else "p50_ms"]
                    for r in rows]) for variant in ("ordinary", "joint")}
                    for stage in ("first_empty_application_cache", "warm_numeric_gather")}}
            if category == "gru":
                g["ppl_relative_change_percent"] = {split: [
                    r["variants"]["joint"]["quality"][split]["delta"]["perplexity_relative_change"] * 100 for r in rows]
                    for split in ("validation", "test")}
                g["maximum_top1_change_fraction"] = max(
                    r["variants"]["joint"]["quality"][split]["delta"]["top1_changed_fraction"]
                    for r in rows for split in ("validation", "test"))
            groups[f"{category}-{order}"] = g
    summary = {"memory_report_sha256": hashlib.sha256(cfg.memory_report.read_bytes()).hexdigest(),
               "lookup_report_sha256": hashlib.sha256(cfg.lookup_report.read_bytes()).hexdigest(),
               "generator": {"command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
                             "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "groups": groups, "aggregation": "Median and seed range; not confidence intervals."}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    labels = ["Flat + ordinary", "DATrie + ordinary", "Flat + joint compact", "DATrie + joint compact"]
    colors = ["#b6bec9", "#6788ab", "#e8aa62", "#26776c"]
    for ax, category, title in zip(axes, ("sgns", "gru"), ("SGNS: vector-only evidence", "Causal GRU: task-tested memory")):
        for i, (factor, label, color) in enumerate(zip(factors, labels, colors)):
            values = [groups[f"{category}-{o}"]["artifact_fraction_of_flat_ordinary"][factor]["median"] * 100 for o in (4, 8, 12)]
            ax.bar(np.arange(3) + (i-1.5)*.19, values, width=.18, color=color, label=label)
        ax.set_xticks(range(3), ["Order 4", "Order 8", "Order 12"])
        ax.set_title(title)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Serving artifact bytes (% of flat + ordinary)")
    figure.legend(labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("DATrie indexing and joint cosine/norm-ratio ≥ 0.99 compaction", fontsize=12)
    figure.tight_layout(rect=(0, .11, 1, .94))
    figure.savefig(cfg.output_dir / "storage-ablation.svg")
    figure.savefig(cfg.output_dir / "storage-ablation.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    keys = list(groups)
    for i, (method, color) in enumerate(zip(("flat", "hash", "datrie"), ("#b6bec9", "#d9a062", "#26776c"))):
        axes[0].bar(np.arange(6)+(i-1)*.24, [groups[k]["lookup_us"][method]["median"] for k in keys],
                    width=.23, label=method, color=color)
    axes[0].set_xticks(range(6), [k.replace("-", "\n").upper() for k in keys])
    axes[0].set_ylabel("512-key lookup (µs, median across 3 seeds)")
    axes[0].set_title("All three lookup implementations are C++")
    axes[0].legend(frameon=False)
    for split, shift, color in (("validation", -.08, "#6788ab"), ("test", .08, "#26776c")):
        for i, order in enumerate((4, 8, 12)):
            values = groups[f"gru-{order}"]["ppl_relative_change_percent"][split]
            axes[1].scatter(np.full(3, i+shift), values, color=color, label=split if i == 0 else None)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(range(3), ["Order 4", "Order 8", "Order 12"])
    axes[1].set_ylabel("Character perplexity change (%)")
    axes[1].set_title("Fixed joint threshold; no retraining / test selection")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(cfg.output_dir / "lookup-and-quality.svg")
    figure.savefig(cfg.output_dir / "lookup-and-quality.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
