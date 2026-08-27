# 我把动态内存管理用在了 FastText 的 n-gram embedding 上

`me_fasttext` 表面上是一个 memory-efficient FastText 变体，但我更愿意把它看成一次系统思想在 AI 模型存储里的重新落地。

FastText 的经典设计很漂亮：一个词的表示由 word vector 和 character n-gram vectors 平均得到。这个设计天然适合长尾词、未登录词、实体名、拼写变化和多语言文本。问题是，如果真的把大语料里的所有 character n-gram 都当作独立对象存下来，矩阵会非常大。

原始 FastText 的办法是 hash bucket：把 n-gram 哈希到固定大小的桶里。这个选择非常工程化，也非常有效。但它有一个代价：n-gram 的身份被提前抹掉了。

换句话说，两个完全无关的 n-gram 可能因为哈希碰撞共享同一行向量。这个共享不是语言现象带来的，也不是训练后发现它们语义接近才做出的压缩，而是在模型学习之前就被哈希函数决定了。

`me_fasttext` 想解决的正是这个问题：能不能先保留精确身份，训练以后再压缩？

## 核心想法

这个项目的核心不是“把 FastText 改得更省内存”这么简单，而是把 FastText 的大规模 n-gram 矩阵看成一个内存管理问题。

在传统动态内存管理里，mark-compact 的思路大致是：

1. 找到还活着的对象。
2. 给这些对象分配新的紧凑地址。
3. 把对象搬到新的连续空间。
4. 更新所有引用。

在 `me_fasttext` 里，对应关系是：

| 动态内存管理 | me_fasttext |
| --- | --- |
| object | word 或 n-gram 的向量行 |
| live object | 压缩后仍需要独立保留的向量行 |
| forwarding address | old row id 到 compact row id 的映射 |
| compact | 把保留行重排成连续矩阵 |
| update reference | 重写 trie 终止节点里的 row id |

这不是在写一个垃圾回收器，而是把垃圾回收和动态内存整理里最经典的思想，用到了 AI 模型 artifact 的导出阶段。

## 为什么是 trie

如果说 mark-compact 提供了“如何整理内存”的思想，那么 trie 提供的是“如何保留身份”的基础。

FastText 的 hash bucket 本质上是匿名化的：n-gram 一进桶，很多细节就消失了。

`me_fasttext` 先给 word 和 UTF-8 character n-gram 分配精确 id，并用 trie 记录这些 id。这样做有几个好处：

- 每个 n-gram 在压缩前都是可定位的。
- 前缀、后缀、局部字符结构仍然存在。
- 压缩可以基于结构关系和训练后的向量相似度，而不是盲目 hash。
- 最终 serving 的时候，可以把 trie 终止节点重写到 compact row id。

这就形成了一个很自然的 pipeline：

1. exact identity first
2. train with separate rows
3. compress with structure and vector similarity
4. rewrite into mmap-friendly serving layout

我觉得这个顺序是项目最有技术含量的地方。

## 为什么 n-gram 值得这样做

character n-gram 很特殊。它既不是普通字符串，也不是完全无结构的参数。

在中文、实体名、多语言文本、拼写变化、长尾词里，n-gram 会呈现大量局部共享：

- 前缀共享
- 后缀共享
- 字符片段共享
- 音译和实体名片段共享
- 罕见词之间仍然共享局部字符证据

这些共享不是随机的。FastText 之所以有效，本来就来自这些局部字符统计现象。

所以问题变成了：既然 n-gram 有结构，为什么要在训练前用 hash collision 随机决定共享？更合理的方式是先保持精确身份，等模型学完以后，再根据结构和向量相似度决定哪些行可以共享。

这就是 `me_fasttext` 的设计味道。

## mmap serving 不是附属功能

很多模型压缩工作停在“矩阵变小了”。但实际服务还会关心：

- 冷启动时间
- 进程间内存共享
- reload 成本
- page cache 行为
- artifact 是否能直接被服务加载

所以 `me_fasttext` 导出的 `.z` 文件不是单纯的压缩矩阵，而是一个面向 mmap serving 的紧凑 artifact。它包含元信息、连续向量矩阵、word trie 和 n-gram trie。训练期的大结构被整理成服务期能直接映射和查询的布局。

这也是为什么我更愿意把它称为 model storage / model serving 方向的系统实践。

## 它不是要替代大模型 embedding

今天的 NLP 当然已经不是 2018 年的样子。Transformer embedding、LLM embedding、dense retrieval 都很强。

但这不意味着小型 lexical memory layer 没有价值。

在很多实际系统里，它仍然适合作为：

- dense reranking 前的低成本召回层
- OOV-heavy retrieval 的兜底层
- 实体密集搜索的字符级特征层
- 中文长尾文本的轻量表示层
- 低延迟分类或排序特征
- 内存敏感环境里的 embedding serving 组件

它的目标不是取代现代语义 encoder，而是在需要“便宜、确定、可解释、OOV 友好”的地方，提供一个有系统设计感的选择。

## 如果当年写出来

这个项目最让我遗憾的地方，是它其实很早就有了清晰的技术内核：动态内存管理、字典树、FastText n-gram 语言统计现象，这三件事接在一起，形成的是一个非常完整的工程研究问题。

如果在 FastText 还非常热、工业 NLP 还大量依赖静态 embedding、中文大词表和端侧部署都很受关注的时候系统整理出来，它的叙事会非常强：

> Memory management for subword embedding systems.

现在回头看，它未必是最大最热的项目，但它有一种少见的深度：它不是追逐新模型，而是把经典计算机系统里的老问题，重新放进 AI 模型存储和服务里。

这也是我觉得它仍然值得被整理、被复现、被引用的原因。

## 一句话总结

`me_fasttext` 把 FastText 的 n-gram embedding 看成一个大规模结构化内存对象：先用 trie 保留精确 subword identity，再用类似 mark-compact 的方式把训练后的向量行整理成紧凑、可 mmap serving 的模型 artifact。

这个项目真正的价值，不在于“又做了一个 FastText 版本”，而在于它证明了一件事：

> 很多 AI 系统问题，仍然可以从经典数据结构和经典内存管理算法里找到优雅答案。
