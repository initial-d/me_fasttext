"""Draw Chinese outreach figures from checked-in aggregate trained evidence.

No corpus, new vectors, or model inference is used. Requires matplotlib.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path,
                        help="Optional local font containing Chinese glyphs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [root / "docs/native_xor_evidence_20261020/trained-native.json",
             root / "docs/cosine_merge_evidence_20261018/report.json"]
    lossless, approximate = [json.loads(p.read_bytes()) for p in paths]
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.unicode_minus": False, "svg.fonttype": "path"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    orders = [4, 8, 12]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    summaries = []
    for layout, label, color in [("shuffled", "打乱行序 + 自适应编码", "#9aa6b2"),
                                  ("lexical", "词典行序 + 自适应 XOR", "#367bc2"),
                                  ("identical_update_group", "相同更新分组 + 自适应 XOR", "#148672")]:
        values = [[100 * (1 - r["layouts"][layout]["xor"]["artifact_to_raw_ratio"])
                   for r in lossless["runs"] if r["max_order"] == order] for order in orders]
        means = [statistics.mean(v) for v in values]
        ax.plot(orders, means, "o-", lw=2.4, color=color, label=label)
        for order, samples, mean in zip(orders, values, means):
            ax.scatter([order] * len(samples), samples, s=18, color=color)
            if layout == "identical_update_group":
                ax.annotate(f"{mean:.2f}%", (order, mean), xytext=(0, 10),
                            textcoords="offset points", ha="center", color=color)
            summaries.append({"order": order, "layout": layout,
                              "saving_mean_percent": mean, "seed_values_percent": samples})
    ax.set_xticks(orders, ["最高 4 阶", "最高 8 阶", "最高 12 阶"])
    ax.set_ylabel("相对原始 n-gram 向量字节的存储节省（%）")
    ax.set_ylim(0, 10.5)
    ax.grid(alpha=.15)
    ax.legend(loc="lower right", fontsize=11)
    fig.suptitle("训练结构能带来多少无损收益？", fontsize=20, y=.97)
    fig.text(.5, .89, "真实文本训练的字符 n-gram；C 读取逐行恢复原始字节", ha="center", color="#445064")
    fig.text(.5, .045, "3 个种子；折线为均值，圆点为各次结果。计入行映射、头部和块描述。", ha="center", fontsize=10)
    fig.text(.5, .015, "未计入键索引、训练元数据和模型其他部分；分组需要训练签名。非 Engram 权重结果。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .87))
    for ext in ("png", "svg"):
        fig.savefig(args.output_dir / f"lossless_structure_zh.{ext}", dpi=180)
    plt.close(fig)

    thresholds = sorted(approximate["thresholds"], reverse=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True)
    curve = []
    for order, color in [(4, "#367bc2"), (8, "#e09b32"), (12, "#148672")]:
        groups = [[r for r in approximate["runs"] if r["max_order"] == order
                   and r["policy"] == "cosine_and_norm_0_99" and r["threshold"] == t]
                  for t in thresholds]
        factors = [statistics.mean(r["compression_factor_raw_to_artifact"] for r in g) for g in groups]
        errors = [statistics.mean(r["word_ngram_component_fidelity"]["relative_l2_mean"] * 100 for r in g) for g in groups]
        axes[0].plot(range(len(thresholds)), factors, "o-", color=color, label=f"最高 {order} 阶", lw=2)
        axes[1].plot(range(len(thresholds)), errors, "o-", color=color, lw=2)
        for t, group, factor, error in zip(thresholds, groups, factors, errors):
            curve.append({"order": order, "cosine_threshold": t, "norm_ratio": .99,
                          "compression_mean": factor, "component_error_mean_percent": error,
                          "component_max_percent": max(r["word_ngram_component_fidelity"]["relative_l2_max"] * 100 for r in group)})
        if order == 12:
            for t, offset in [(.99, (-4, -23)), (.98, (0, 12)), (.9, (-12, 12))]:
                i = thresholds.index(t)
                axes[0].annotate(f"{factors[i]:.2f}×", (i, factors[i]), xytext=offset,
                                 textcoords="offset points", ha="center", color=color)
    axes[0].set_ylabel("原始向量字节 / 实际存储字节（倍）")
    axes[0].legend(loc="lower right")
    axes[1].set_ylabel("词的 n-gram 均值分量\n平均相对 L2 变化（%）")
    axes[1].set_xlabel("最低余弦相似度（离散设置；向右放宽，模长比始终 ≥ 0.99）")
    axes[1].set_xticks(range(len(thresholds)), [f"{t:g}" for t in thresholds])
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("多接受一点向量变化，能多省多少？", fontsize=20, y=.97)
    fig.text(.5, .92, "余弦 + 模长比联合筛选；3 个种子的均值", ha="center", color="#445064")
    fig.text(.5, .045, "12 阶、余弦 ≥ 0.99：单行平均 L2 约 4.2%，单行最大约 14.2%；分量最坏约 1.29%。", ha="center", fontsize=10)
    fig.text(.5, .015, "这里只测向量及 n-gram 均值分量；不是模型损失、准确率、困惑度或完整模型压缩倍数。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .90))
    for ext in ("png", "svg"):
        fig.savefig(args.output_dir / f"approximate_tradeoff_zh.{ext}", dpi=180)
    plt.close(fig)
    outputs = sorted(args.output_dir.glob("*.png")) + sorted(args.output_dir.glob("*.svg"))
    manifest = {"category": "outreach figures redrawn from aggregate public-text SGNS evidence",
                "command": sys.argv, "python": sys.version, "matplotlib": matplotlib.__version__,
                "platform": platform.platform(), "font": str(args.font) if args.font else None,
                "inputs": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
                "lossless": summaries, "approximate": curve,
                "note": "No new training, corpus download, weights redistribution or performance measurement."}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
