# Legacy export: structure + cosine + norm ratio

The current `.z` exporter accepts approximate sharing only when the original row
and its final representative have a whole-key prefix/suffix relationship,
cosine similarity >= 0.99 and min(norm)/max(norm) >= 0.99. It no longer merges
equal-norm opposite vectors. This implements the joint criterion used in the
[trained-row experiments](trained_ngram_similarity_20261018.md) at the legacy
export boundary; it does not reproduce the paper's large-corpus numbers.

## Implementation and evidence map

| Claim | Code | Check / limitation |
| --- | --- | --- |
| Direction and magnitude constrain sharing | `src/row_merge.h` | Reject opposite vectors and same-direction vectors with different norms |
| Keys must be structurally related | `src/legacy_merge.h` | Complete-key prefix or suffix ancestry; common first/last characters alone are insufficient |
| First trie key need not be row zero | `LegacyMergeState.initialized` | First visited key retains its own ID |
| No drift across prefix/suffix passes | `FinalMergeGuard` | Compare every original row directly with its final representative; restore its original ID on failure |
| Export stride/copy follows model dimension | Both `FastText::saveIndex` overloads | Callback tests cover 1, 2, 64, 128 and 256; legacy serving helpers still contain 128-element buffers |
| Byte-preserving Engram gather | Independent `native/meft_rows.c` | Different storage path; approximate `.z` merging is not lossless |

Zero vectors merge only with zero vectors under the structural restriction.
Nonfinite vectors are not merged. Threshold checks use double accumulation over
stored float rows. These are row-level constraints, not a downstream quality
guarantee. Thresholds remain fixed at 0.99 in the legacy exporter.

Each traversal keeps one current representative; the implementation does not
search all structural ancestors for maximal compression. The final guard adds
an original row-ID-to-key map during export, requiring O(number of rows + total
key bytes) extra builder memory and another traversal. Large-corpus construction
memory, artifact size, latency and downstream quality require fresh measurement.
The file layout is unchanged, but regenerated row counts and vectors can change.
Previously exported files keep their previous approximate content.

## Run checks

```bash
python3 -m unittest discover -s tests -p 'test_legacy_merge.py'
make test
# Linux: check the complete exporter translation unit with bundled headers.
c++ -std=c++11 -I. -Imylib/include -fsyntax-only src/fasttext.cc
```

The portable regression uses the production merge callbacks and a map-backed
trie test double under AddressSanitizer and UndefinedBehaviorSanitizer. It tests
threshold, structural, ID and final-representative semantics. It does not verify
the bundled trie traversal, file serialization, mmap serving or task quality.
Linux CI also checks the complete exporter translation unit. Full legacy builds
depend on bundled platform libraries; the macOS header failure is recorded in
the [run note](legacy_merge_check_20261023.md).

The mathematical vectors in regression cases are adversarial correctness
controls. They are not a claim about the distribution of trained embeddings.
Use the existing trained-row experiments for empirical similarity evidence and
keep a new full FastText reproduction separate from the original paper results.
