# Engram-focused technical launch copy

Repository publication makes these materials reviewable. No social post or
third-party message is sent by creating this page. Results and limits link to
[trained evidence](trained_ngram_similarity_20261018.md) and the
[C ABI/model guide](native_ngram_memory.md).

## One-line positioning

**Compact high-order n-gram memory: structural indexing, byte-exact storage,
and native batched gather for Engram-style systems.**

## English technical post

```text
Engram puts a practical systems question on the table: as n-gram memory grows,
how do we store and query it efficiently while keeping original lookup results?

me_fasttext studies that boundary through exact n-gram indexing, compact row
layouts and mmap access. It now has a standalone C row-gather ABI and immutable
CPU PyTorch adapters. The native path writes directly into the output tensor,
preserving original row IDs, head offsets, dtype and every stored byte.

The interesting compression hypothesis concerns trained structure. Prefix/suffix
n-grams that share their training updates can become highly similar; independent
random vectors are useful interface controls but do not represent that mechanism.

We trained an exact character-ngram SGNS reference on public English text at
max orders 4/8/12 with three seeds. Identical-update structural pairs have median
cosine around 0.996, with 85.9–89.4% reaching 0.99. Their share of structural
pairs rises from 21.8% to 59.4% as the max order rises from 4 to 12.

High cosine is not byte equality. All trained rows remain distinct. Ordinary
native block compression saves about 5%; a separate reversible bitwise-XOR
experiment with shared-update row grouping estimates 6.9–9.1% savings including
map/descriptor overhead. Shuffling rows weakens the effect. All original bytes
are recovered, and all runs/settings are public. The predictor is not yet a
native query codec, and update-signature grouping requires training metadata.

This is a FastText-style reference, not a trained Engram/model-quality result.
Native gathers beat our Python reference in a small warm control but remain
slower than resident tensor gather. GPU/runtime integrations need further work.

Next: trained token-ngram tables, structure-aware lossless predictors, native
query scheduling and matched-cost model/serving evaluations. A licensed frozen
memory table or reproducible gather trace would make a useful comparison.

Code: https://github.com/initial-d/me_fasttext
Evidence: https://github.com/initial-d/me_fasttext/blob/main/docs/trained_ngram_similarity_20261018.md
C/PyTorch integration: https://github.com/initial-d/me_fasttext/blob/main/docs/native_ngram_memory.md
```

## 中文技术帖

```text
Engram 让 n-gram 记忆重新进入大模型视野。me_fasttext 想深入其中一个具体问题：
记忆表越做越大，怎样利用训练形成的结构，把它存得更紧凑、查得更有效？

项目从 FastText 的精确 n-gram 索引、紧凑行布局和 mmap 查询出发，现在补上了
真正的 C 行查询 ABI，以及直接写入 PyTorch 输出张量的 CPU 适配器。原始 row ID、
head 偏移、dtype 和每一行的原始字节都保持一致。

压缩实验应当研究训练过的向量。前后缀相关的 n-gram 往往一起更新，随机向量
只能验证接口和低冗余边界，不能代表这种机制。

这次用公开英文文本训练精确字符 n-gram SGNS，在最高 4/8/12 阶、三个种子下
做了 9 组实验。更新来源和次数完全相同的结构组合，相似度中位数约为 0.996，
85.9–89.4% 达到 0.99。随着最高阶数从 4 到 12，这类组合在结构组合中的占比
从 21.8% 增到 59.4%。这支持进一步研究高阶 n-gram 的训练冗余。

高相似度仍然保留各行的独立细节。直接分块压缩节省约 5%；进一步把共享更新
的行分组、做可逆的按位 XOR 差异编码，估算节省约 6.9–9.1%，包含行映射和块
描述开销。打乱行序后收益减弱；恢复时每一位都保持原样，不做近似合并。

全部实验、种子、命令、哈希和限制都已公开。这是 FastText 风格训练参考，还
没有真实 Engram 权重或大模型质量结果；差异编码目前是离线实验，尚未接入 C
查询。现有 C 查询在小型热访问对照中快于 Python 参考，仍慢于内存张量 gather。

接下来应当用训练好的 token n-gram 表，验证结构预测压缩、原生查询调度，以及
同预算下的模型质量和实际存储/RSS/延迟。欢迎有公开记忆表或查询轨迹的研究者复现。

代码：https://github.com/initial-d/me_fasttext
训练证据：https://github.com/initial-d/me_fasttext/blob/main/docs/trained_ngram_similarity_20261018.md
接入说明：https://github.com/initial-d/me_fasttext/blob/main/docs/native_ngram_memory.md
```

## Article outline and follow-up

Suggested title: **《Engram 之后：利用高阶 n-gram 的共同训练结构做无损存储》**.

1. The scaling problem: exact key identity, table storage and query fan-out.
2. Why shared training matters: structural pairs versus identical update sources.
3. All nine trained results and shuffled controls, including weaker cases.
4. Byte-exact XOR prediction versus approximate row sharing; source audit of
   the legacy norm-ratio criterion and the paper's different cosine rule.
5. C ABI → FFI → PyTorch gather; tested boundaries and remaining GPU/runtime work.
6. A concrete licensed-table benchmark request and the next reproducible experiment.

The [research plan](ngram_memory_research_plan.md) defines the paper follow-up.
Current evidence supports a structural compression research direction; it does
not establish universal compression, model quality or production speedups.
