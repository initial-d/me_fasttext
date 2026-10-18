# Engram-focused technical launch copy

These drafts are ready to share after checking the linked branch/commit is
public. No social post or third-party message is sent by creating this page.

## One-line positioning

**Compact n-gram memory: exact indexing, storage compression, and mmap lookup —
with a byte-exact row-store reference for Engram-style experiments.**

## English post

```text
Engram brings n-gram memory back into the LLM systems conversation. The next
question is practical: how should large memory tables be stored and queried?

I am developing me_fasttext around that storage problem: exact trie indexing,
compact row layouts and mmap serving, originally explored with FastText.

There is now a small runnable lossless row-store reference: deduplicate only
byte-identical rows, keep original row IDs, compress blocks and retrieve through
mmap with a bounded cache. It verifies every row and repeated-ID query ordering.

The reference is a storage experiment, not an Engram checkpoint adapter. The
legacy FastText exporter also has an implementation mismatch with the paper;
the source audit is included. Random-byte demos do not predict model savings.

Update: an immutable CPU adapter now matches the official demo's multi-head
gather interface. On random weights, all rows and hash-driven query outputs
match byte-for-byte in float32/float16/bfloat16. Grouped reads reduce repeated
block decoding; files still grow and gathers are slower than resident PyTorch
in this small test. A trained checkpoint remains the next evaluation step.

I would like to test the same-row gather contract on immutable n-gram embedding
tables, measuring storage, RSS and p50/p95 against direct mmap/tensor gathers.

Code: https://github.com/initial-d/me_fasttext
Guide: https://github.com/initial-d/me_fasttext/blob/main/docs/engram_storage.md
```

## 中文技术帖

```text
Engram 把 n-gram 记忆重新带回了大模型讨论。我更关心它接下来的工程问题：
当记忆表变得很大，怎样存得更紧凑、查得更有效，同时保持原始查询结果？

这是 me_fasttext 想深入的方向。项目从 FastText 出发，研究精确 trie 索引、
紧凑行布局和 mmap 查询；现在补了一个可以直接运行的无损行存储参考实现。

它只共享字节完全相同的行，保留每个原始 row ID，再做分块压缩和 mmap 查询。
测试检查逐行字节一致、重复 ID 顺序和缓存淘汰；不是靠“相似向量”声称无损。

也把边界公开说清楚：旧 C++ 导出实现与论文的相似度规则存在差异，已附源码
审计；新工具还没有接入训练好的 Engram。随机字节实验不能推导真实模型收益。

现在也补了官方 demo 多头 gather 接口的 CPU 适配：保持每个 head 的偏移不变，
随机权重的 float32/float16/bfloat16 存储行和 hash 查询输出都逐字节一致。
批量读按块合并能减少重复解压，但小样例文件仍增大，查询也慢于内存中的
PyTorch gather；还需要训练权重才能评估实际价值。

下一步最有价值的实验，是保持 Engram 的 hash/row ID 不变，对不可变记忆表
比较原始 gather 和压缩查询，分别测字节一致性、存储、RSS、p50/p95 和吞吐。

欢迎做 n-gram memory / embedding serving 的朋友复现或提供公开表的对比结果。
代码：https://github.com/initial-d/me_fasttext
技术说明：https://github.com/initial-d/me_fasttext/blob/main/docs/engram_storage.md
```

## Where this version fits

- A technical blog or Zhihu article titled
  **《Engram 之后：大规模 n-gram 记忆怎样存、怎样查？》**.
- Embedding-serving, model-storage and conditional-memory discussions with
  room for source links and measurements.
- A reproducible public benchmark post. Include the actual row distribution,
  checkpoint/license, exact command/commit, CPU/GPU and cold/warm conditions.

Prefer one substantive technical post and benchmark follow-up to repeated
generic list submissions. The concrete invitation is a same-ID, same-byte gather
comparison; Engram-wide speedups, universal compression ratios and strict
losslessness of the legacy exporter are not supported claims.
