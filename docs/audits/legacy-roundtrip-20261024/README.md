# Real legacy artifact roundtrip — 2026-10-24

Source commit: `10d1898c1f8ad4d2d2a89443e8cf8ecf78fd5931`.
[Successful Linux CI run](https://github.com/initial-d/me_fasttext/actions/runs/40042470352).
[Metadata, commands and artifact/library hashes](report.json); [raw output](run.txt).
The workflow preserves the full build log as the `legacy-roundtrip-report` artifact.

```bash
python tools/check_legacy_roundtrip.py --output-dir artifacts/legacy-roundtrip
```

The checker builds the complete exporter and reader against the bundled Linux
x86-64 trie/pub libraries. It invokes both `FastText::saveIndex` overloads,
serializes actual `.z` files and reopens them through `ftindex_mmap`.

The six cases (64/128/256 dimensions × two overloads) passed all original-key,
known-word, OOV, direction/norm/structure and final-representative checks.
The driver checks tail components above 128 and non-lexical original row IDs.
Repeated export and close passed AddressSanitizer, UndefinedBehaviorSanitizer
and LeakSanitizer, with bundled libraries linked but not instrumented.

Inputs are 14 deterministic mathematical n-gram controls and one word. They
include exact duplicates, a near-angle pair, opposite vectors, norm mismatch,
unrelated equal rows, suffix ancestry, UTF-8 keys and zero rows. These are
synthetic engineering fixtures, not trained FastText or Engram weights and not
paper-scale memory, latency or downstream-quality evidence. CPU only; no market
or corpus downloads, GPU or paid API. Cost/performance estimates are not applicable.

Environment: Ubuntu Linux x86-64, AMD EPYC 9V74 CPU, GCC 13.3.0, Python 3.12.15;
full OS/kernel and exact flags are in the report. Local portable tests also
passed all 36 cases on macOS/Apple M5, Python 3.12.14/PyTorch 2.14.1.

## Preserved initial failure

[Initial run](https://github.com/initial-d/me_fasttext/actions/runs/40042285757)
compiled successfully but failed the suffix-sharing assertion on the 64-dim
fixture. [Initial metadata](initial-report.json) and [raw failure](initial-run.txt)
are retained. The one-current-representative traversal lost a valid shorter
ancestor after a mismatching branch. The corrected callback retains the visited
prefix chain; the final original-to-representative guard remains in place.

The change also frees both serialized word-trie buffers and both mmap trie
views, fixing omissions found during lifecycle inspection. Valid-file serving
is covered here; malformed legacy-file hardening and sentence helpers with
128-element buffers remain separate work.
