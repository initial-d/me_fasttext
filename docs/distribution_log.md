# Distribution log

## Three additional technical answers — 2026-10-22

Published three distinct answers on the maintainer's Zhihu `initial` account,
following an explicit request for additional answers on questions with larger
existing audiences:

| Answer | Question audience before publication | Published (Asia/Shanghai) | Source |
| --- | --- | --- | --- |
| [Engram architecture and storage boundary](https://www.zhihu.com/question/1994233409871050526/answer/2096647218841260122) | 1,503,083 historical views; 1,854 followers; 202 answers | 17:00 | [Text](zhihu_engram_architecture_answer_20261022.txt) |
| [Low-VRAM inference: capacity, access and scheduling](https://www.zhihu.com/question/2090845638498042646/answer/2096647356443897859) | 142,059 historical views; 224 followers; 32 answers | 17:01 | [Text](zhihu_low_vram_answer_20261022.txt) |
| [Quantization methods and distortion evaluation](https://www.zhihu.com/question/627484732/answer/2096647525356810483) | 180,728 historical views; 311 followers; 52 answers | 17:02; citation corrected 17:03 | [Text](zhihu_quantization_answer_20261022.txt) |

The editor counted 1,821 / 1,702 / 2,074 characters at initial publication;
the quantization answer counted 2,089 after the corrected QuaRot citation was
resolved to its paper title. All three link the published Chinese article and
repository, disclose maintainer affiliation and AI assistance, and set the
platform's AI-assisted-creation declaration. Publication-success UI and each
public answer permalink were inspected. The initially mistaken QuaRot URL was
corrected to [arXiv:2404.00456](https://arxiv.org/abs/2404.00456), with its title
verified on arXiv and in the updated answer.

The Engram answer explains deterministic addressing, prefill/decode limits,
original-row-ID preservation and the C/PyTorch interface. The low-VRAM answer
explains capacity versus per-step access, CPU/GPU placement, block read
amplification and comparable serving measurements; it does not claim a Strata
reproduction or integration. The quantization answer explains PTQ/QAT,
GPTQ/AWQ/SmoothQuant/QLoRA/rotation distinctions and the cosine-plus-norm error
contract. Existing trained-character-ngram evidence is reused with its row-tail
and full-model limitations; no new benchmark was run.

Question views are **historical audience context**, not views received by these
answers. No attributed repository visits, stars, citations, adoption or future
recommendation boost is established. No private message, paid promotion or
automatic cross-post was sent for this entry.

## Chinese Engram article and figures — 2026-10-22

Prepared [a complete Chinese technical article](zhihu_engram_article_20261022.md),
two [data-derived figures](engram_launch_20261022/figure_manifest.json), runnable
factory commands and refreshed [sharing copy](engram_promotion.md). The figures
reuse checked-in aggregate trained SGNS reports; no new compression/model-quality
measurement or corpus/weights redistribution is claimed.

Published on the maintainer's Zhihu `initial` account on 2026-10-22:

- [Technical article](https://zhuanlan.zhihu.com/p/2096640222452781097):
  5,443 editor-counted characters, two trained-evidence figures, source links,
  factory commands and paper/software citation boundaries. Topics: DeepSeek,
  model compression and NLP. The UI showed publication success and the public
  article page was inspected.
- [Companion introduction](https://www.zhihu.com/pin/2096642168534348240):
  205-character introduction linking the article, with exact/approximate and
  row-error limits. Publication success and public detail page verified.

- [Storage-cost answer](https://www.zhihu.com/question/2087854646002037757/answer/2096644241111962030):
  published at 16:49 Asia/Shanghai to a directly relevant DRAM/cost question,
  with 1,551 editor-counted characters. It explains relocation versus lossless
  encoding versus approximation, prefetch limits and measured codec costs,
  discloses maintainer affiliation and links the article/code. The question
  showed 10,657 historical views and five existing answers before publication;
  that is audience context, not this answer's reach or an attributed conversion.
  Publication success and the answer permalink were verified. The
  [answer source](zhihu_engram_storage_answer_20261022.md) is retained.

[Repository PR #12](https://github.com/initial-d/me_fasttext/pull/12) merged as
`18202b4195f56e62c2114929611b59a14039ce32`; all CI checks passed. The documented
factory command was rerun locally: all nine dtype/backend cases preserve full
output bytes and avoid default table allocation. Publication is not evidence of
attributed visits, stars, citations or Engram adoption. No paid promotion or
private maintainer message is part of this entry.
Upstream Engram PR #27 remains open/draft/unmerged on 2026-10-22.

## Engram contribution and runnable artifact — 2026-10-19

[DeepSeek Engram draft PR #27](https://github.com/deepseek-ai/Engram/pull/27)
was submitted on 2026-10-18 for an optional embedding factory. Checked on
2026-10-19: open/draft, no reviews or comments, not merged. The upstream change
has no me_fasttext dependency or official-adoption claim.

The [reader quickstart](engram_quickstart.md) connects this contribution to
real mapped-backend construction, official CPU forward equality, trained
character-ngram distortion curves and software/paper citations. Six proposal
factory cases avoid default table allocation and preserve full module output;
25 unit tests pass. Real Engram weights/GPU/task quality remain open.

The traffic response checked on 2026-10-19 ends before the PR submission, so it
cannot establish PR-attributed discovery, stars or citations. This update makes
the reproduction/reuse path concrete; it does not record a measured conversion
rate. No new third-party comment or social message is sent for this update.

## Engram storage entry — 2026-10-18

Added an Engram-focused README entry, architecture/storage guide, byte-exact
row-store reference, synthetic positive/negative results and English/Chinese
technical launch drafts. The legacy-export audit records the norm-ratio/cosine
implementation mismatch and hardcoded 128-dimensional loops. This is a repository
update, not a social post, third-party submission or trained Engram benchmark.
See [the guide](engram_storage.md) and [launch copy](engram_promotion.md).

Follow-up: added an immutable CPU multi-head gather adapter and pinned official
hash/gather interface verification with random weights, plus coalesced batch
block reads. Byte equality, larger artifacts and slower gathers are recorded
in [the interface result](engram_adapter_verification_20261018.md).
No trained checkpoint or outside post/submission was used.

This file tracks public list submissions and follow-up status for `me_fasttext`.
It is intentionally factual: accepted entries first, then open submissions that may
need maintainer follow-up.

## Accepted listings

| Date | List | Placement |
| --- | --- | --- |
| 2026-08-26 | [Awesome Data Science](https://github.com/academic/awesome-datascience/pull/679) | General Machine Learning Packages |

## Open submissions

| Date | List | Link | Placement proposed |
| --- | --- | --- | --- |
| 2026-09-07 | awesome-vector-search | [PR 70](https://github.com/currentslab/awesome-vector-search/pull/70) | Library |
| 2026-09-07 | awesome-gc | [PR 1](https://github.com/cch123/awesome-gc/pull/1) | Applications |
| 2026-08-26 | awesome-embedding-models | [PR 12](https://github.com/Hironsan/awesome-embedding-models/pull/12) | Paper and implementation |
| 2026-08-26 | awesome-information-retrieval | [PR 22](https://github.com/harpribot/awesome-information-retrieval/pull/22) | Retrieval software |
| 2026-08-26 | awesome-ml-model-compression | [PR 11](https://github.com/cedrickchee/awesome-ml-model-compression/pull/11) | Compression resource |
| 2026-08-26 | awesome-semantic-search | [PR 78](https://github.com/Agrover112/awesome-semantic-search/pull/78) | Libraries and Tools |
| 2026-08-26 | awesome-NLP-resources | [PR 12](https://github.com/HanXinzi-AI/awesome-NLP-resources/pull/12) | Text retrieval and semantic matching |
| 2026-08-26 | Awesome-Text-Classification | [PR 2](https://github.com/fendouai/Awesome-Text-Classification/pull/2) | Projects |

## Blocked or unsuitable targets

| Date | Target | Status | Note |
| --- | --- | --- | --- |
| 2026-10-12 | [awesome-nlp PR 422](https://github.com/keon/awesome-nlp/pull/422) | Closed, not merged | Do not nudge again without maintainer guidance; use a better-fit NLP embeddings list instead. |
| 2026-09-22 | [awesome-machine-learning PR 1411](https://github.com/josephmisiti/awesome-machine-learning/pull/1411) | Closed, not merged | Keep the narrowed C++ NLP wording for future lists, but do not describe this as an accepted listing. |
| 2026-09-07 | Separius/awesome-sentence-embedding | Archived / read-only | Prepared fork branch [`initial-d:add-me-fasttext-word-embedding`](https://github.com/initial-d/awesome-sentence-embedding/tree/add-me-fasttext-word-embedding) adding the paper to the Word Embeddings table, but GitHub rejected PR creation because the upstream repository is archived. |
| 2026-09-07 | currentslab/awesome-vector-search | Submitted after benchmark tooling | Initially deferred because the list focuses on ANN engines, vector databases, and similarity-search libraries; submitted after adding the `bench_ftindex` retrieval/OOV serving benchmark. |

## Follow-up rule

Avoid same-day nudges. For open pull requests, wait at least one week before a
single polite follow-up, and only follow up when the list is still active or the
maintainer asks for changes.

## Follow-up log

| Date | Link | Note |
| --- | --- | --- |
| 2026-09-07 | [awesome-nlp PR 422](https://github.com/keon/awesome-nlp/pull/422#issuecomment-5570207578) | Added compact-inference and positioning context. |
| 2026-09-07 | [awesome-embedding-models PR 12](https://github.com/Hironsan/awesome-embedding-models/pull/12#issuecomment-5570207623) | Added embedding/citation context. |
| 2026-09-07 | [awesome-information-retrieval PR 22](https://github.com/harpribot/awesome-information-retrieval/pull/22#issuecomment-5570207663) | Added retrieval and benchmark context. |
| 2026-09-07 | [awesome-ml-model-compression PR 11](https://github.com/cedrickchee/awesome-ml-model-compression/pull/11#issuecomment-5570207731) | Added exact-then-compact compression context. |
| 2026-09-12 | [awesome-machine-learning PR 1411](https://github.com/josephmisiti/awesome-machine-learning/pull/1411#issuecomment-5646575114) | Added evaluator quickstart and narrowed the placement to C++ NLP / FastText-style subword embeddings. |

## Release page updates

| Date | Release | Note |
| --- | --- | --- |
| 2026-09-07 | [v0.1.0 research prototype](https://github.com/initial-d/me_fasttext/releases/tag/v0.1.0) | Added compact-inference and cross-community positioning links to the release notes. |

## Benchmark tooling updates

| Date | Commit | Note |
| --- | --- | --- |
| 2026-09-07 | [a450399](https://github.com/initial-d/me_fasttext/commit/a450399) | Added `bench_ftindex` and a retrieval/OOV serving benchmark guide for mmap load time, p50/p95 query latency, and OOV subword coverage. |
| 2026-09-07 | [a56a909](https://github.com/initial-d/me_fasttext/commit/a56a909) | Added an artifact card so external reports can triage fit, claims, evidence, comparisons, and citation hooks without overclaiming. |
| 2026-09-09 | [3226776](https://github.com/initial-d/me_fasttext/commit/3226776) | Added a first public-corpus benchmark plan to convert clone interest into one reproducible external report. |
| 2026-09-10 | [560a48f](https://github.com/initial-d/me_fasttext/commit/560a48f) | Added report-ready criteria and a dedicated first public-corpus benchmark issue type so external runs can be triaged for citation readiness. |
| 2026-09-10 | [84064df](https://github.com/initial-d/me_fasttext/commit/84064df) | Added a public benchmark preparation helper that creates a corpus manifest, deduplicated query file, and per-slice query files from a plain-text corpus. |

## Contributor conversion updates

| Date | Commit | Note |
| --- | --- | --- |
| 2026-09-13 | This update | Added `CONTRIBUTING.md` and a pull request template so silent evaluators have a clear path for benchmark reports, negative results, documentation fixes, and code patches. |
| 2026-09-14 | This update | Added a code of conduct, an evaluation-question issue template, and issue-template contact links so evaluators can ask focused questions before writing a full benchmark report. |
| 2026-10-12 | This update | Added a `showcase` GitHub label so citations, integrations, list entries, and use-case notes can be triaged separately from full benchmark reports. |

## Positioning updates

| Date | Commit | Note |
| --- | --- | --- |
| 2026-09-14 | This update | Added a restrained conditional-memory context note connecting `me_fasttext` to Engram-style indexed memory as adjacent design pressure, without claiming lineage or equivalence. |
| 2026-09-15 | This update | Added English and Chinese conditional-memory promotion snippets that keep the Engram framing cautious and point back to the context note. |
| 2026-09-21 | This update | Added a model-compression and Engram-style lookup bridge that frames `me_fasttext` as a compact lexical-memory baseline with explicit benchmark requirements. |

## Visibility updates

| Date | Commit | Note |
| --- | --- | --- |
| 2026-09-08 | [3f78130](https://github.com/initial-d/me_fasttext/commit/3f78130) | Recorded the first visibility snapshot: 248 views, 80 clones, 48 unique cloners, and a next-action plan focused on benchmark evidence rather than generic star requests. |
| 2026-09-10 | [560a48f](https://github.com/initial-d/me_fasttext/commit/560a48f) | Recorded the latest snapshot: 288 views, 164 clones, 79 unique cloners, with the clone/star gap pointing toward private evaluation and benchmark-report conversion. |
| 2026-09-11 | This update | Recorded the latest snapshot: 295 views, 139 clones, 65 unique cloners, and added an evaluator quickstart to convert silent clones into benchmark reports. |
| 2026-09-12 | This update | Recorded the latest snapshot: 236 views, 156 clones, 66 unique cloners, and added first-screen trust badges for paper, CI, release, benchmark intake, and evaluator quickstart. |
| 2026-09-24 | This update | Added `llms.txt` and an answer-engine context page so search and AI summaries can extract the canonical description, citation target, evaluation boundaries, and "do not claim" list. |
| 2026-09-28 | This update | Added a citation evidence ladder so benchmark and compression reports can be classified from setup notes through citable independent reproductions. |
| 2026-10-12 | This update | Added a share kit with copy-ready summaries, social posts, maintainer notes, benchmark calls to action, and responsible sharing boundaries. |
| 2026-10-13 | This update | Added a listing follow-up page and moved closed-not-merged external PRs out of the open-submission table. |

## Next high-fit targets

Prioritize targets where one of these phrases is naturally true:

- FastText-style word/subword embeddings;
- OOV-heavy lexical embedding model;
- double-array trie used as model-index infrastructure;
- mark-compact layout transformation outside runtime GC;
- compact lexical inference or mmap serving artifact.

Avoid generic vector database, LLM-agent, or runtime-GC lists unless there is a
specific section for retrieval features, model layout, or applied memory
management.
