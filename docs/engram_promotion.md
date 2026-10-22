# Engram-focused article and sharing copy — 2026-10-22

**Compact n-gram memory: structural indexing, byte-exact storage, and native
batched gather for Engram-style systems.**

The completed Chinese technical article is
[Engram 之后，高阶 n-gram 记忆还能压缩多少？一次可复现的存储实验](zhihu_engram_article_20261022.md).
It provides the mechanism, trained controls, exact/approximate separation,
row-level tail errors, C/PyTorch boundary and runnable factory demo.
Repository publication and external platform publication are recorded separately
in the [distribution log](distribution_log.md).

## Figures

- [Chinese lossless structure figure](engram_launch_20261022/lossless_structure_zh.png)
- [Chinese approximation/distortion figure](engram_launch_20261022/approximate_tradeoff_zh.png)
- [Source hashes and plotted values](engram_launch_20261022/figure_manifest.json)
- [Plotting script](../tools/plot_engram_launch.py); these are aggregate evidence
  redraws, with no new vectors, corpus download or training.

## Chinese short introduction

```text
Engram 把 n-gram 记忆表的存储和查询问题重新带到大模型面前。
我在 me_fasttext 中实现了独立 C 行查询后端和 PyTorch CPU 适配器，并用真实文本
训练的字符 n-gram 测了两条路线：无损存储节省约 6.9–9.1%；最高 12 阶、余弦和
模长比都 ≥0.99 时，实际向量存储缩小约 2.15 倍。
后者是近似：n-gram 均值分量平均 L2 变化约 0.48%，单行平均约 4.2%、最大约
14.2%。这些不是 Engram 训练权重、完整模型压缩倍数或任务质量结果。
文章有两张数据图、全部报告链接和可运行接入示例，欢迎复现或提交负面结果。
代码：https://github.com/initial-d/me_fasttext
文章：https://github.com/initial-d/me_fasttext/blob/main/docs/zhihu_engram_article_20261022.md
```

## English technical introduction

```text
How much trained structure can we exploit in high-order n-gram memory?

me_fasttext now has a C-readable adaptive lossless XOR codec, original-ID
forwarding, and immutable CPU PyTorch gather adapters for Engram-style tables.
On nine public-text trained character-ngram SGNS tables (max orders 4/8/12,
three seeds), identical-update grouping saves 6.9–9.1% of raw vector bytes,
including native maps/headers/descriptors. Every original row is recovered.
The layout needs training signatures; hashed buckets do not supply them.

A separate approximate prefix/suffix row-sharing experiment combines direct
cosine and norm-ratio constraints. At max order 12, both ≥0.99 give about 2.15×
raw-vector / artifact bytes. Mean word-ngram-component relative L2 is ~0.48%,
but all-row mean is ~4.2% and maximum ~14.2%. No task-quality claim follows.

Python/native/native-XOR paths preserve full pinned official demo CPU outputs
across float32/float16/bfloat16 on tiny random fixtures (interface tests only).
No trained Engram checkpoint or GPU workload was evaluated. Upstream draft
PR #27 remains unmerged as checked on 2026-10-22; there is no adoption claim.

Code: https://github.com/initial-d/me_fasttext
Run/evaluate/cite: https://github.com/initial-d/me_fasttext/blob/main/docs/engram_quickstart.md
Evidence: https://github.com/initial-d/me_fasttext/blob/main/docs/native_xor_compression_20261020.md
Distortion: https://github.com/initial-d/me_fasttext/blob/main/docs/approximate_ngram_compression_20261018.md

A licensed frozen token-ngram table or public gather trace would be a useful
next comparison. Independent reproductions and negative results are welcome.
```

## Publication interpretation

File ratios cover n-gram vector rows, not complete models or measured RSS.
Approximate sharing changes original vectors; cosine thresholds are not task
loss percentages. Character prefix/suffix and update-signature evidence does not
automatically transfer to trained Engram hash buckets. Original paper results
and these later software experiments must be cited separately. See the
[citation guide](citation_guide.md#cite-the-native-ngram-artifact).
