# Legacy merge checks — 2026-10-23

Code commit: `2a1b39801b3810193e97c1106d6df359c878d669`. [Implementation contract](legacy_merge_contract.md).

Environment: macOS 26.6.2 (arm64), Apple M5, Apple Clang 21.0.0;
Python 3.12.14, PyTorch 2.14.1; CPU only. Inputs are mathematical regression
controls and existing unit fixtures, not a real Engram checkpoint or a new
trained-corpus benchmark. No storage-ratio, latency or task-quality claim.

```bash
python -m unittest discover -s tests
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
```

Result: 36 tests passed, including the production merge callbacks compiled with
`-std=c++11 -Wall -Wextra -Werror -fsanitize=address,undefined
-fno-omit-frame-pointer -g`. Native C gather library and example compiled with
`-std=c11 -O2 -Wall -Wextra -Werror`. Full unit output:

```text
....................................
----------------------------------------------------------------------
Ran 36 tests in 5.057s

OK

```

The first run in an old Python 3.9.6 / PyTorch 2.8.0 environment passed 35 tests
and failed the missing-input CLI test because a broken environment `.pth` file
printed `_distutils_hack` startup tracebacks. All 36 passed after a clean install.

The complete legacy exporter translation unit was checked separately:

```bash
c++ -std=c++11 -I. -Imylib/include -fsyntax-only src/fasttext.cc
```

On macOS this fails in bundled `pub/pub_malloc.h:5` with `fatal error:
'malloc.h' file not found`. A temporary compatibility header containing
`#include <malloc/malloc.h>` and added via `-I/tmp/meft-legacy-check-20261023`
allowed the same syntax check to pass. That is only a translation-unit check,
not a linked legacy exporter/reader run. Linux CI checks the unmodified headers.
The portable callback regression uses a trie test double and does not validate
bundled library traversal, serialization or mmap serving.
