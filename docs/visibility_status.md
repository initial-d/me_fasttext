# Visibility status

This page tracks the public discovery surface for `me_fasttext` and turns
traffic signals into conservative next actions. It is not a popularity report:
the project is still a niche research artifact, so useful evidence matters more
than broad social metrics.

## Live entry points

- Repository: <https://github.com/initial-d/me_fasttext>
- Paper: <https://arxiv.org/abs/2506.01254>
- GitHub release: <https://github.com/initial-d/me_fasttext/releases/tag/v0.1.0>
- Benchmark intake: <https://github.com/initial-d/me_fasttext/issues/1>
- Design note: <https://github.com/initial-d/me_fasttext/issues/2>
- Artifact card: <https://github.com/initial-d/me_fasttext/blob/main/docs/artifact_card.md>
- Retrieval/OOV serving benchmark: <https://github.com/initial-d/me_fasttext/blob/main/docs/retrieval_oov_benchmark.md>
- Retrieval comparison matrix: <https://github.com/initial-d/me_fasttext/blob/main/docs/retrieval_comparison_matrix.md>
- Chinese retrospective: <https://github.com/initial-d/me_fasttext/blob/main/docs/technical_retrospective_zh.md>
- Awesome Data Science listing: <https://github.com/academic/awesome-datascience>
- Open awesome-vector-search PR: <https://github.com/currentslab/awesome-vector-search/pull/70>
- Open awesome-gc PR: <https://github.com/cch123/awesome-gc/pull/1>

## Traffic snapshot

Recorded on 2026-09-10 from GitHub's rolling 14-day traffic window
(latest daily bucket available through 2026-09-08):

- Views: 288 total, 13 unique visitors.
- Clones: 164 total, 79 unique cloners.
- Repository stars/forks/watchers at snapshot time: 1 star, 0 forks, 0 watchers.
- Clone interest accelerated after the benchmark/reporting docs landed:
  56 clones from 23 unique cloners on 2026-09-07 and 28 clones from
  14 unique cloners on 2026-09-08.
- Top referrers by visits: GitHub (48) and Zhihu (1). These are visits, not
  unique-user totals.
- High-interest paths beyond the repository root include benchmark issue #1,
  design note issue #2, `index`, `fasttext`, `Makefile`, the Chinese
  retrospective, activity, pulse, and traffic pages.

Operating takeaways:

- The gap between clones and stars widened again. Treat this as private
  evaluation or automated exploration, not as broad community adoption.
- The practical next step is to make one external public-corpus report
  complete enough to cite. The benchmark issue template now includes a
  "First public corpus benchmark" report type.
- Because `index`, `fasttext`, and `Makefile` are visible traffic paths,
  build/export instructions should stay close to the first-corpus benchmark
  page and the README entry points.

Recorded on 2026-09-09 from GitHub's rolling 14-day traffic window
(latest daily bucket available through 2026-09-07):

- Views: 268 total, 11 unique visitors.
- Clones: 136 total, 68 unique cloners.
- Repository stars/forks/watchers at snapshot time: 1 star, 0 forks, 0 watchers.
- The largest recent clone bucket was 56 clones from 23 unique cloners on
  2026-09-07.
- Top referrers by visits: GitHub (42) and Zhihu (1). These are visits, not
  unique-user totals.
- High-interest paths beyond the repository root include benchmark issue #1,
  design note issue #2, `fasttext`, `index`, the Chinese retrospective, and a
  recent commit page.

Operating takeaways:

- The September 7 clone spike makes a first public-corpus benchmark more useful
  than another broad positioning note.
- The best next conversion is a single reproducible report that an embedding,
  retrieval, or systems reader can cite.
- A dedicated first-run plan was added in
  [`first_public_corpus_benchmark.md`](first_public_corpus_benchmark.md).

Recorded on 2026-09-08 from GitHub's rolling 14-day traffic window
(latest daily bucket available through 2026-09-05):

- Views: 248 total, 9 unique visitors.
- Clones: 80 total, 48 unique cloners.
- Repository stars/forks/watchers at snapshot time: 1 star, 0 forks, 0 watchers.
- Clone activity spiked during the first outreach window: 37 clones from
  25 unique cloners on 2026-08-26 and 18 clones from 13 unique cloners on
  2026-08-27.
- Recent visible traffic is small but steady: 10-12 daily views from
  4-5 unique visitors on 2026-09-03 through 2026-09-05.
- Top referrers by visits: GitHub (37) and Zhihu (1). These are visits, not
  unique-user totals.
- High-interest paths beyond the repository root include benchmark issue #1,
  design note issue #2, the Chinese retrospective, `fasttext`, `index`,
  help-wanted labels, activity, traffic, and pulse pages.

## Operating takeaways

- `me_fasttext` is being evaluated in a narrow technical circle rather than
  spreading broadly. That matches the project: it is a specialized artifact for
  FastText-style lexical embeddings, trie-backed ids, mmap serving, and
  memory-layout work.
- Clone count is stronger than star count, so star growth should not be the
  primary target. The better conversion is one public benchmark, one issue
  discussion, or one external citation/reuse note.
- The Chinese retrospective and design note are real entry points. Keep them
  connected to the benchmark/reporting path so readers can move from story to
  evidence.
- GitHub-native discovery is currently stronger than external referrers.
  Awesome-list PRs may help, but the repository should also make direct
  visitors immediately see how to evaluate and cite the artifact.

## Current positioning

Short description:

> FastText inference optimization with trie-backed n-gram ids, mark-compact
> vector storage, and mmap-friendly retrieval.

Primary ask:

> Share a reproducible memory, load-time, retrieval/OOV, or compact-inference
> benchmark report.

## Next iterations

- Keep benchmark issue #1 as the main intake for reproducible results.
- Ask for one external corpus run using
  [`first_public_corpus_benchmark.md`](first_public_corpus_benchmark.md) rather
  than more generic promotion.
- Wait for maintainer feedback before nudging open awesome-list PRs.
- Use the artifact card when approaching maintainers who need a quick triage
  page rather than a long essay.
- Prefer benchmark evidence over claims about replacing modern dense embedding
  models, vector databases, RAG systems, or runtime garbage collectors.
- If traffic continues to concentrate on the Chinese retrospective, publish a
  shorter English technical note around the same memory-management story.
