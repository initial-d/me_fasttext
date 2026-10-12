# Share kit

Use this page when sharing `me_fasttext` with researchers, list maintainers,
benchmarkers, or engineers who need a compact lexical embedding baseline. The
goal is discoverability with accurate scope, not hype.

## Canonical links

- Repository: https://github.com/initial-d/me_fasttext
- Paper: https://arxiv.org/abs/2506.01254
- Artifact card: https://github.com/initial-d/me_fasttext/blob/main/docs/artifact_card.md
- Evaluator quickstart: https://github.com/initial-d/me_fasttext/blob/main/docs/evaluator_quickstart.md
- Benchmark report template: https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml
- Showcase / use-case template: https://github.com/initial-d/me_fasttext/issues/new?template=showcase.yml
- Citation guide: https://github.com/initial-d/me_fasttext/blob/main/docs/citation_guide.md

## Short descriptions

### One sentence

```text
me_fasttext is a FastText-derived C++ system that keeps word and character
n-gram identities explicit with trie-backed ids, then exports a compact
mmap-friendly lexical embedding artifact.
```

### Directory entry

```text
me_fasttext - Memory-efficient FastText-style lexical embeddings with exact
trie-backed word/subword ids, conservative n-gram row sharing, and compact
mmap serving for OOV-heavy retrieval, ranking, and feature-serving workloads.
```

### Paper-oriented summary

```text
me_fasttext explores an exact-then-compact FastText design: preserve explicit
word and UTF-8 character n-gram ids with trie-backed lookup during training,
then rewrite selected learned rows into a smaller mmap-ready serving artifact.
It is best evaluated with memory, cold-load, latency, OOV coverage, and
downstream quality metrics on a clearly described corpus.
```

## Social posts

### Short post

```text
I released me_fasttext, a FastText-derived C++ prototype for memory-efficient
lexical embeddings.

It replaces hash-bucketed subword ids with exact trie-backed ids, then exports
a compact mmap serving artifact for OOV-heavy retrieval, entity-heavy search,
and low-memory feature serving.

Code: https://github.com/initial-d/me_fasttext
Paper: https://arxiv.org/abs/2506.01254
```

### Benchmark ask

```text
If you use FastText-style lexical embeddings, I would love independent
me_fasttext benchmark reports.

Most useful: corpus manifest, original FastText or quantized baseline, .z
artifact size, cold mmap load time, p50/p95 lookup latency, OOV coverage, and a
retrieval/ranking/classification quality metric.

Quickstart: https://github.com/initial-d/me_fasttext/blob/main/docs/evaluator_quickstart.md
```

### Chinese short post

```text
我开源了 me_fasttext：一个 memory-efficient FastText 风格的 C++ 原型。

它把原本 hash bucket 里的 word/subword id 换成 trie-backed exact id，再把
训练后的 n-gram rows 做保守合并和 mark-compact layout，导出可 mmap serving
的紧凑 lexical embedding artifact。

适合 OOV-heavy retrieval、实体检索、中文/多语言长尾词、低内存 feature serving
等场景。

代码: https://github.com/initial-d/me_fasttext
论文: https://arxiv.org/abs/2506.01254
```

## Maintainer note

```markdown
Hi, thanks for maintaining this list.

Would you consider adding `me_fasttext`?

It is a FastText-derived C++ research prototype for memory-sensitive lexical
embedding serving. Instead of hashing character n-grams into fixed buckets, it
assigns exact trie-backed ids, then exports a compact mmap serving artifact via
conservative row sharing and mark-compact style layout.

Repo: https://github.com/initial-d/me_fasttext
Paper: https://arxiv.org/abs/2506.01254
Artifact card: https://github.com/initial-d/me_fasttext/blob/main/docs/artifact_card.md

Disclosure: I maintain the project. I think it fits best under word embeddings,
retrieval infrastructure, compact inference, or model compression depending on
your taxonomy. Happy to adjust or withdraw if it is out of scope.
```

## Where to share

High-fit places:

- word embedding and NLP resource lists;
- retrieval and search engineering lists where lexical prefilters fit;
- model-compression or efficient-inference lists with non-LLM sections;
- trie, double-array trie, mmap, or systems discussions about model layout;
- papers, blog posts, and benchmark reports that need a compact lexical
  embedding baseline.

Lower-fit places:

- general LLM-agent directories;
- vector database lists without lexical or prefilter sections;
- runtime garbage-collector lists unless they accept applied model-layout work;
- broad social posts that cannot include benchmark context or scope limits.

## Preferred call to action

Ask for one of these:

- a star or watch if the reader wants to follow benchmark notes;
- a benchmark result with corpus, commands, hardware, and quality metrics;
- a showcase issue for citations, list entries, integrations, or use cases;
- a narrow evaluation question before someone spends time on a full run.

## Responsible sharing checklist

- Say it is a FastText-derived compact lexical embedding system.
- Mention exact trie-backed word/subword ids and mmap serving.
- Pair memory or latency claims with corpus and hardware context.
- Invite negative results and omitted-baseline notes.
- Do not claim it replaces transformer embeddings.
- Do not call it a vector database, full RAG framework, LLM memory module, or
  runtime garbage collector.
- Do not imply the paper's compression ratio is universal.
