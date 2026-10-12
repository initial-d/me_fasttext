# Listing follow-up

This page turns external list-submission status into concrete next actions.
Keep it factual: accepted entries and maintainer decisions are evidence; open
pull requests are only pending distribution attempts.

## Current status

Checked on 2026-10-13.

| Target | Status | Link | Next action |
| --- | --- | --- | --- |
| Awesome Data Science | Merged | https://github.com/academic/awesome-datascience/pull/679 | Keep README listing; no follow-up needed. |
| awesome-nlp | Closed, not merged | https://github.com/keon/awesome-nlp/pull/422 | Do not nudge this PR. Reuse the feedback only if a better-fit NLP list appears. |
| awesome-machine-learning | Closed, not merged | https://github.com/josephmisiti/awesome-machine-learning/pull/1411 | Do not nudge unless maintainers explicitly invite a narrower placement. |
| awesome-vector-search | Open | https://github.com/currentslab/awesome-vector-search/pull/70 | Follow up only with benchmark or retrieval-specific evidence. |
| awesome-gc | Open | https://github.com/cch123/awesome-gc/pull/1 | Follow up only with the model-layout / mark-compact framing. |
| awesome-embedding-models | Open | https://github.com/Hironsan/awesome-embedding-models/pull/12 | Good fit for paper plus implementation; use citation-oriented wording. |
| awesome-information-retrieval | Open | https://github.com/harpribot/awesome-information-retrieval/pull/22 | Good fit only if framed as lexical retrieval infrastructure. |
| awesome-ml-model-compression | Open | https://github.com/cedrickchee/awesome-ml-model-compression/pull/11 | Good fit only with exact-then-compact compression evidence. |
| awesome-semantic-search | Open | https://github.com/Agrover112/awesome-semantic-search/pull/78 | Follow up with lexical prefilter / OOV retrieval wording. |
| awesome-NLP-resources | Open | https://github.com/HanXinzi-AI/awesome-NLP-resources/pull/12 | Follow up only if the list remains active. |
| Awesome-Text-Classification | Open | https://github.com/fendouai/Awesome-Text-Classification/pull/2 | Low priority unless supervised FastText-style classification evidence is added. |

## Follow-up priority

1. Retrieval and semantic-search lists, if there is a public retrieval/OOV
   benchmark to cite.
2. Embedding and model-compression lists, if the maintainer accepts research
   prototypes and paper implementations.
3. Systems or memory-management lists, only with the narrow model-layout
   explanation from [`engineering_note.md`](engineering_note.md).
4. Broad ML/NLP lists only when they have a clear word-embedding, C++ NLP, or
   efficient-inference section.

## Polite follow-up template

```markdown
Hi, checking whether this still fits the list.

Since opening the PR, I added a shorter artifact card and evaluator quickstart:

- Artifact card: https://github.com/initial-d/me_fasttext/blob/main/docs/artifact_card.md
- Evaluator quickstart: https://github.com/initial-d/me_fasttext/blob/main/docs/evaluator_quickstart.md
- Share kit: https://github.com/initial-d/me_fasttext/blob/main/docs/share_kit.md

The intended scope is narrow: FastText-style lexical embeddings with exact
trie-backed subword ids and compact mmap serving. If that is out of scope for
this list, no worries; happy to close or adjust the wording.
```

## Do not do

- Do not reopen or re-nudge closed PRs without new maintainer guidance.
- Do not describe open PRs as accepted listings.
- Do not submit the same generic blurb to many lists on the same day.
- Do not pursue lists where the project would need to be framed as a vector
  database, full RAG framework, LLM memory module, or runtime garbage
  collector.
