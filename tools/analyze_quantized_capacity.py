"""Summarize measured table bytes, per-process latencies and CPU resources."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def range_summary(values):
    return dict(median=float(np.median(values)), min=float(min(values)), max=float(max(values)), values=values)


def analyze(report):
    summary = {key: report[key] for key in ("status", "source_commit", "command", "shape", "source_bytes",
                "source_sha256", "python", "numpy", "os", "cpu", "settings", "limitations",
                "native_library_sha256", "keys_bytes", "keys_sha256", "training_report_sha256")}
    summary["training"] = report["training"]
    summary["software"] = {key: report.get(key) for key in ("pytorch_installed_not_used", "thread_environment", "cpu_affinity")}
    summary.update(builds={}, queries={}, failures=[])
    for name, build in report["builds"].items():
        if build["returncode"]:
            summary["failures"].append(dict(phase="build", variant=name, error=build.get("error")))
            continue
        value = build["result"]
        m = value["metadata"]
        summary["builds"][name] = dict(rows=m["shape"][0], encoding=m["encoding"], backend=m["backend"],
            wall_seconds=value["wall_seconds"], cpu_seconds=value["cpu_seconds"],
            peak_rss_bytes=value["memory_after"]["peak_rss_bytes"], table_bytes=value["table_bytes"],
            bundle_bytes=value["bundle_bytes"], bundle_with_keys_bytes=value.get("bundle_with_keys_bytes"),
            storage=m["storage"], source_sha256=m["source_float32_sha256"], table_sha256=m["table_sha256"],
            source_bytes_preserved=m["source_bytes_preserved"])
        verified = report["verification"].get(name)
        if verified and not verified["returncode"]:
            v = verified["result"]
            summary["builds"][name]["verification"] = {key: v[key] for key in
                ("rows_verified", "all_original_ids_encoded_byte_exact", "all_decoded_rows_oracle_byte_exact",
                 "encoded_original_id_sha256", "decoded_original_id_sha256", "relative_frobenius", "row_relative_l2", "cosine_distance")}
        elif verified:
            summary["failures"].append(dict(phase="verify", variant=name, error=verified.get("error")))
    groups = {}
    for worker in report["workers"]:
        if worker["returncode"]:
            summary["failures"].append(dict(phase="query", variant=worker["variant"], case=worker["case"], error=worker.get("error")))
            continue
        groups.setdefault((worker["variant"], worker["case"]), []).append(worker["result"])
    for (variant, case), workers in groups.items():
        target = dict(processes=len(workers), query_rows=workers[0]["query_rows"], distinct_rows=workers[0]["distinct_rows"],
                      batch_rows=workers[0]["batch_rows"], query_buffer_bytes=workers[0]["query_buffer_bytes"],
                      cache_budget_bytes=workers[0]["cache_budget_bytes"],
                      all_query_bytes_exact=all(w["all_query_bytes_exact"] for w in workers))
        for phase in ("first_trace", "second_trace", "hot_repeated"):
            target[phase] = {"wall_" + key: range_summary([float(np.percentile(w[phase]["wall"]["samples_ms"], q)) for w in workers])
                             for key, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99))}
            target[phase]["rows_per_second"] = range_summary([w[phase]["rows_per_second"] for w in workers])
            target[phase]["cpu_p50_ms"] = range_summary([w[phase]["cpu"]["p50_ms"] for w in workers])
            if variant.find("native") >= 0:
                decoded, compressed, hits = [], [], []
                for w in workers:
                    work = w[phase]["codec_work"]
                    decoded.append(sum(c["last_decoded_bytes"] for c in work))
                    compressed.append(sum(c["last_compressed_bytes"] for c in work))
                    hits.append(sum(c["last_row_cache_hits"] for c in work))
                target[phase]["logical_decoded_bytes"] = range_summary(decoded)
                target[phase]["logical_compressed_bytes"] = range_summary(compressed)
                target[phase]["row_cache_hits"] = range_summary(hits)
                row_width = report["builds"][variant]["result"]["metadata"]["row_bytes"]
                output_bytes = workers[0]["batch_rows"] * row_width * len(workers[0][phase]["wall"]["samples_ms"])
                target[phase]["decoded_over_encoded_output"] = range_summary([v / output_bytes for v in decoded])
        for phase in ("memory_after_first_trace", "memory_after_second_trace", "memory_after_hot"):
            target[phase] = {key: range_summary([w[phase][key] for w in workers])
                             for key in ("peak_rss_bytes", "VmRSS_bytes") if all(key in w[phase] for w in workers)}
        summary["queries"].setdefault(variant, {})[case] = target
    baseline = summary["builds"].get("fp32-raw")
    if baseline:
        for name, v in summary["builds"].items():
            if v["rows"] != report["shape"][0]:
                continue
            v["table_saving_vs_raw_fp32"] = 1 - v["table_bytes"] / baseline["table_bytes"]
            v["bundle_saving_vs_raw_fp32"] = 1 - v["bundle_with_keys_bytes"] / baseline["bundle_with_keys_bytes"]
            matched = summary["builds"].get(v["encoding"] + "-raw")
            if matched:
                v["bundle_added_saving_vs_same_encoding_raw"] = 1 - v["bundle_with_keys_bytes"] / matched["bundle_with_keys_bytes"]
    return summary


def plot(summary, directory):
    import matplotlib.pyplot as plt
    names = ["fp32-raw", "fp32-native", "fp16-raw", "fp16-native", "int8-raw", "int8-native",
             "int8-native-rows-4", "int8-native-rows-1"]
    names = [n for n in names if n in summary["builds"]]
    labels = [n.replace("-raw", " mmap").replace("-native-rows-", " native/").replace("-native", " native") for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    colors = ["#4c78a8" if "fp32" in n else "#59a14f" if "fp16" in n else "#f28e2b" for n in names]
    axes[0].barh(labels, [summary["builds"][n]["bundle_with_keys_bytes"] / 1e9 for n in names], color=colors)
    axes[0].set(xlabel="Table + codec + common keys (GB)",
                title=f"Same {summary['shape'][0]/1e6:.3f}M × {summary['shape'][1]} trained rows")
    query_names = ["fp32-direct"] + names
    for case, marker in (("heldout-lexical", "o"), ("uniform-id-stress", "s")):
        values = [summary["queries"][n][case]["first_trace"]["wall_p50_ms"]["median"] for n in query_names]
        axes[1].plot(values, list(range(len(query_names))), marker=marker, label=case)
    axes[1].set_yticks(range(len(query_names)), [n.replace("fp32-direct", "FP32 direct mmap") for n in query_names])
    axes[1].set(xscale="log", xlabel="Fresh-process first-trace batch p50 (ms)", title="512-row CPU gather → FP32")
    axes[1].legend(fontsize=8)
    scale_names = ["int8-native-prefix-100000", "int8-native-prefix-500000", "int8-native"]
    scale_names = [n for n in scale_names if n in summary["builds"]]
    axes[2].plot([summary["builds"][n]["rows"] / 1e6 for n in scale_names],
                 [summary["builds"][n]["peak_rss_bytes"] / 2**20 for n in scale_names], "o-", color="#f28e2b")
    axes[2].set(xlabel="Original trained rows (millions)", ylabel="Fresh builder peak RSS (MiB)",
                title="INT8 streaming export, ≤8 KiB blocks")
    for ax in axes[:2]:
        ax.invert_yaxis()
    fig.suptitle("Quantization is lossy; native preserves encoded bytes. Shared CPU, warmed filesystem, no GPU.", fontsize=11)
    for ext in ("svg", "png"):
        fig.savefig(directory / ("scale-performance." + ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    opener = gzip.open if args.report.suffix == ".gz" else open
    with opener(args.report, "rt") as stream:
        summary = analyze(json.load(stream))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plots:
        plot(summary, args.output_dir)


if __name__ == "__main__":
    main()
