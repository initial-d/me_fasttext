# Trained n-gram similarity and lossless storage reference — 2026-10-18

**Category: public-text trained SGNS reference.** This is neither original C++
FastText reproduction nor trained Engram evaluation. Independent random rows are
initialization/control only, not representative compression evidence.

The shared-update mechanism is visible: structurally related rows with identical
host-word/multiplicity signatures have median cosine about **0.996**. Across nine
runs, **85.9–89.4%** of those pairs reach 0.99. Higher maximum n-gram order
increases the proportion of these signatures in the structural pair population.
This supports studying learned structural redundancy; it does not establish that
all prefix/suffix neighbors have cosine ≥0.99.

Raw evidence: [all nine trained runs](native_ngram_evidence_20261018/trained-reference.json),
[residual/layout runs](native_ngram_evidence_20261018/trained-residuals.json),
[environment and source hashes](native_ngram_evidence_20261018/provenance.json).

## Learned distributions

Pairs are a child and its immediate prefix/suffix parent (one character removed,
minimum length two); duplicate prefix/suffix pairs are counted once. “Identical
updates” is the subset with equal `(host word ID, multiplicity)` signatures,
not mere spelling similarity. All structural pairs in this corpus share at least
one host word, but they often have different other host words.

| Max order | Seed | Rows | Structural pairs | Structural cosine median | Structural ≥0.99 | Identical-update pairs | Their cosine median | Their ≥0.99 | Native bytes/raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 3 | 5,333 | 9,865 | 0.5814 | 19.1% | 2,146 | 0.9964 | 87.5% | 0.9501 |
| 4 | 7 | 5,333 | 9,865 | 0.5843 | 19.4% | 2,146 | 0.9965 | 89.4% | 0.9499 |
| 4 | 11 | 5,333 | 9,865 | 0.5787 | 18.9% | 2,146 | 0.9965 | 86.6% | 0.9500 |
| 8 | 3 | 13,424 | 26,047 | 0.9896 | 49.3% | 14,938 | 0.9961 | 85.9% | 0.9485 |
| 8 | 7 | 13,424 | 26,047 | 0.9899 | 49.9% | 14,938 | 0.9961 | 87.0% | 0.9488 |
| 8 | 11 | 13,424 | 26,047 | 0.9898 | 49.7% | 14,938 | 0.9959 | 86.6% | 0.9487 |
| 12 | 3 | 14,132 | 27,463 | 0.9910 | 52.0% | 16,317 | 0.9962 | 87.4% | 0.9482 |
| 12 | 7 | 14,132 | 27,463 | 0.9914 | 52.3% | 16,317 | 0.9963 | 87.9% | 0.9484 |
| 12 | 11 | 14,132 | 27,463 | 0.9911 | 51.9% | 16,317 | 0.9961 | 87.3% | 0.9484 |

The identical-update share of structural pairs rises from **21.8%** at max
order 4 to **57.4%** at 8 and **59.4%** at 12. This small word vocabulary
largely saturates by order 12; it is not evidence of indefinite growth or a
matched-capacity model improvement. Seed variation is shown, not a confidence
interval over languages or checkpoints.

Prefix, suffix, occurrence strata (5/50/200/1000), ≥0.999, initialization
distributions and examples are all in the raw report. Initial structural cosine
medians are near zero. Training loss per pair falls from 4.1589 to 2.29–2.46;
these are sampled SGNS losses, not held-out quality or convergence evidence.

## Byte-exact storage and structural prediction

**All learned rows remain byte-distinct.** Native `MEFTNAT1` storage therefore
gets no exact-row dedup savings here. Its ordinary zlib blocks save about 5.0–5.2%
including maps/header/descriptors, with all rows recovered byte-for-byte.
Lexical versus shuffled ordinary zlib differs by less than 0.05% of raw bytes.
High cosine alone is not enough for strong generic byte-compression savings.

A separate offline experiment predicts each row with the previous row using
**bitwise XOR**, then zlib. It preserves every float bit, including the independent
initialization differences; no floating subtraction or approximate sharing occurs.
Blocks reset prediction at their first row. Three layouts isolate locality:
lexical, seeded shuffled, and grouping rows by identical update signatures.

| Max order | Seed | XOR lexical estimate/raw | XOR shuffled estimate/raw | XOR update-group estimate/raw |
| --- | --- | --- | --- | --- |
| 4 | 3 | 0.9557 | 0.9656 | 0.9310 |
| 4 | 7 | 0.9557 | 0.9655 | 0.9308 |
| 4 | 11 | 0.9557 | 0.9656 | 0.9309 |
| 8 | 3 | 0.9322 | 0.9646 | 0.9108 |
| 8 | 7 | 0.9324 | 0.9646 | 0.9111 |
| 8 | 11 | 0.9326 | 0.9647 | 0.9115 |
| 12 | 3 | 0.9304 | 0.9643 | 0.9095 |
| 12 | 7 | 0.9305 | 0.9644 | 0.9095 |
| 12 | 11 | 0.9307 | 0.9643 | 0.9099 |

Update-group XOR estimates save **6.9%** at order 4 and **8.9–9.1%** at
orders 8/12, including an estimated 8-byte row map and native-width header/
descriptors. This is a promising structural lossless-codec direction. The XOR
codec and these construction layouts are **not implemented in the C reader**,
and estimates are not actual `.men` file sizes or latency results. Update
signatures require training metadata; spelling-only grouping is weaker. Every
offline block and the mapping back to original IDs were checked byte-exact.

## Mechanism and legacy criterion

If two rows receive the same additive gradients, their initial offset remains
in real arithmetic. A large shared learned component can make their cosine high
while their bytes differ. Finite-precision SGD can add further drift. Prefix or
suffix membership alone does not imply identical gradients: e.g. `play` appears
in `play`, `played`, and `playing`, while `playi` only appears in `playing`.

The legacy norm-ratio criterion is also insufficient: in these runs, roughly
8–11% of accepted structural pairs have cosine below 0.99 (see `norm_ratio_audit`).
This is a separate source-implementation issue, not a refutation of the shared
training mechanism. The [source audit](engram_storage.md#legacy-export-audit)
remains applicable. Approximate similarity merging needs reconstruction-error
and task-quality studies separate from strict lossless storage.

## Reproduce

Training commit: `ba564539489c44ff2582a7d83039a57c7dde7b9e`. Residual experiment commit:
`62fa3907c81df69a68e760a54c67867d481708f7`. Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4;
Apple M5, 10 logical CPUs, 32 GiB, macOS 26.6.2 arm64, one PyTorch/interop thread.
Native compiler: Apple clang 21.0.0. CPU only; local compute, no paid API or rented
compute. No monetary/energy estimate. Corpus source license was not established
in this run; raw corpus and learned weights are not redistributed.

```bash
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
mkdir -p artifacts/native-ngram-20261018/corpus
curl -fL https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt \
  -o artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt
python tools/train_ngram_similarity.py \
  --corpus artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt \
  --native-library build/libmeft_rows.dylib \
  --output-dir artifacts/native-ngram-20261018/trained-reference \
  --max-orders 4,8,12 --seeds 3,7,11 --steps 2000
python tools/benchmark_trained_residuals.py \
  --training-dir artifacts/native-ngram-20261018/trained-reference \
  --corpus artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt \
  --output artifacts/native-ngram-20261018/trained-residuals.json
```

Use `.so` and `make native` on Linux. Exact executed absolute-path commands,
source hashes, row hashes, per-run losses and settings are in the JSON reports.
Use fresh output directories to reproduce experiments. The training report is
written after each completed run; exceptions still fail the command.

Training uses exact 2..N character n-grams including word boundaries, top 1,200
English words (minCount 5), a separate word row, adjacent original token positions
within ±2, dim128 float32, 512,000 sampled pairs/run, batch256, five negatives,
sparse SGD lr0.05. Hidden values are averaged but input-row gradients follow
the legacy unscaled convention; negative weights use counts^0.5. This reference
uses batched SGD without original C++ subsampling or a learning-rate schedule.

Development [pilot one](native_ngram_evidence_20261018/development-pilot.json)
and [pilot two](native_ngram_evidence_20261018/development-pilot2.json) are retained
with their different step/optimization settings, excluded from final comparisons.

## Next evidence needed

Repeat on licensed multilingual and trained token-ngram/Engram tables, with
longer training, matched total memory/compute, trained baselines and task quality.
Test structural predictors with corpus-independent grouping, then implement a
versioned native codec and measure build memory, cold/warm I/O, RSS, full request
fan-out and CPU/GPU transfer. See the [research plan](ngram_memory_research_plan.md)
and [native model integration guide](native_ngram_memory.md).
