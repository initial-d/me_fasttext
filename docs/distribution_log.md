# Distribution log

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
| 2026-09-07 | awesome-gc | [PR 1](https://github.com/cch123/awesome-gc/pull/1) | Applications |
| 2026-08-26 | awesome-nlp | [PR 422](https://github.com/keon/awesome-nlp/pull/422) | Text embeddings |
| 2026-08-26 | awesome-machine-learning | [PR 1411](https://github.com/josephmisiti/awesome-machine-learning/pull/1411) | C++ NLP resources |
| 2026-08-26 | awesome-embedding-models | [PR 12](https://github.com/Hironsan/awesome-embedding-models/pull/12) | Paper and implementation |
| 2026-08-26 | awesome-information-retrieval | [PR 22](https://github.com/harpribot/awesome-information-retrieval/pull/22) | Retrieval software |
| 2026-08-26 | awesome-ml-model-compression | [PR 11](https://github.com/cedrickchee/awesome-ml-model-compression/pull/11) | Compression resource |
| 2026-08-26 | awesome-semantic-search | [PR 78](https://github.com/Agrover112/awesome-semantic-search/pull/78) | Libraries and Tools |
| 2026-08-26 | awesome-NLP-resources | [PR 12](https://github.com/HanXinzi-AI/awesome-NLP-resources/pull/12) | Text retrieval and semantic matching |
| 2026-08-26 | Awesome-Text-Classification | [PR 2](https://github.com/fendouai/Awesome-Text-Classification/pull/2) | Projects |

## Blocked or unsuitable targets

| Date | Target | Status | Note |
| --- | --- | --- | --- |
| 2026-09-07 | Separius/awesome-sentence-embedding | Archived / read-only | Prepared fork branch [`initial-d:add-me-fasttext-word-embedding`](https://github.com/initial-d/awesome-sentence-embedding/tree/add-me-fasttext-word-embedding) adding the paper to the Word Embeddings table, but GitHub rejected PR creation because the upstream repository is archived. |
| 2026-09-07 | currentslab/awesome-vector-search | Deferred | Strong list, but current taxonomy focuses on ANN engines, vector databases, and similarity-search libraries; `me_fasttext` would need a clearer vector-search benchmark before submission. |

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

## Release page updates

| Date | Release | Note |
| --- | --- | --- |
| 2026-09-07 | [v0.1.0 research prototype](https://github.com/initial-d/me_fasttext/releases/tag/v0.1.0) | Added compact-inference and cross-community positioning links to the release notes. |

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
