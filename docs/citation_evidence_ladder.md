# Citation evidence ladder

This page classifies `me_fasttext` reports by how much evidence they provide
for reuse, comparison, or citation. The goal is to make weak reports still
useful while keeping README-level claims conservative.

## Why this exists

`me_fasttext` is easiest to overstate when a report shows one attractive number:
for example a smaller `.z` artifact or a faster mmap load. Those numbers are
useful, but they are not enough to claim the method transfers to a new corpus.

Use this ladder when deciding whether a benchmark issue, note, or external
writeup should be linked from project docs.

## Evidence levels

| Level | Name | What it proves | Link from README? |
| --- | --- | --- | --- |
| 0 | Interest signal | Someone cloned, built, or inspected the project. | No |
| 1 | Setup report | The project builds or fails on a documented machine. | No |
| 2 | Serving-only report | The compact `.z` artifact has measured size, load time, RSS, or latency. | Usually no |
| 3 | Comparable benchmark | Serving metrics are paired with a named corpus, baseline, and commands. | Sometimes |
| 4 | Citable report | Comparable benchmark plus quality guardrail, caveats, and reproducible metadata. | Yes |
| 5 | Independent reproduction | A report by someone outside the maintainer path reproduces or challenges a paper claim. | Yes |

Negative results can reach level 4 or 5. A clear failure on a public corpus is
more useful than an incomplete positive result.

## Level 1: setup report

A setup report is useful for portability, but it is not citation evidence yet.

Minimum fields:

- commit SHA;
- OS, compiler, Python version, and shell;
- build command;
- whether `make opt`, `make test`, or both passed;
- exact error if the build failed.

Use this level for issues about Linux distribution differences, compiler
versions, bundled static libraries, or filesystem assumptions.

## Level 2: serving-only report

A serving-only report measures the compact artifact, but does not establish task
quality.

Minimum fields:

- commit SHA;
- corpus summary or synthetic data description;
- `.bin`, `.vec`, and `.z` sizes where available;
- `bench_ftindex` table or equivalent latency output;
- hardware, OS, filesystem, and storage type;
- explicit caveat: `serving-only; no task-quality claim`.

This level can support engineering discussion about load time or mmap behavior.
It should not be used to claim retrieval or classification quality is preserved.

## Level 3: comparable benchmark

A comparable benchmark has enough context for another reader to rerun or compare
the serving trade-off.

Minimum fields:

- all level-2 fields;
- public corpus name, license status, language, and preprocessing;
- training command and model settings;
- at least one baseline, usually original FastText or quantized FastText;
- fixed query file or query-slice description;
- explanation for omitted baselines.

This level can be linked from issue discussions and protocol docs when the
report is useful but still missing a quality guardrail.

## Level 4: citable report

A citable report ties a claim to a reproducible setting. It can support a paper,
related-work paragraph, or README-level evidence link.

Minimum fields:

- all level-3 fields;
- one retrieval, ranking, classification, or feature-serving quality metric;
- OOV rate and OOV subword coverage for the reported query slice;
- clear positive, negative, or mixed conclusion;
- known caveats, failures, and omitted comparisons;
- public link to scripts, generated manifest, or enough command text to rerun.

Recommended summary:

```markdown
Claim tested:
Corpus:
Baseline:
me_fasttext commit:
Serving result:
Quality result:
Caveat:
```

## Level 5: independent reproduction

An independent reproduction is a level-4 report produced outside the maintainer
path, or a report that challenges a paper claim with enough detail to audit.

Examples:

- reproduces the memory/load-time direction on a public corpus;
- shows a corpus where compact export saves memory but hurts quality;
- compares `me_fasttext` with quantized FastText under a shared protocol;
- reports a build or portability blocker with a precise fix or environment.

These reports should be preserved even when the result is unfavorable. They are
the strongest evidence for where the method transfers and where it does not.

## Maintainer linking rule

Use this rule before adding a report to README-level docs:

1. Link level-4 or level-5 reports from README, release notes, or citation docs.
2. Keep level-2 and level-3 reports inside issue threads or benchmark protocol
   docs until quality, baseline, or corpus metadata is added.
3. Label negative level-4 reports as evidence, not as failures to hide.
4. Do not turn a private or redacted report into a public claim unless the
   corpus summary and metric definitions are still understandable.

For the report form, use the
[benchmark result template](https://github.com/initial-d/me_fasttext/issues/new?template=benchmark-result.yml)
or the
[compression comparison template](https://github.com/initial-d/me_fasttext/issues/new?template=compression-comparison.yml).

