# NGramStore: canonical description

NGramStore, formerly me_fasttext, is a compression, storage and exact-lookup
backend for learned n-gram vectors. Its native C engine serves original numeric
row IDs from compressed mmap blocks; registered string n-grams additionally
support DATrie indexing and structural mark-compact planning.

FastText is the word-embedding application and historical method origin.
Engram/TinyEngram are LLM-memory integration examples, retaining the model's
hashing, head offsets, gates and backbone. The native backend is shared; their
training and addressing contracts remain different.

## Fidelity and evidence

Ordinary block packing and structural layout/exact modes preserve source bytes.
Joint structural sharing requires related registered strings plus cosine and
norm-ratio guards and can change weights. Engram hash collisions do not expose
this unique key registry. Compression, latency, RSS and downstream quality are
independent measurements.

Actual legacy FastText trained-model outputs and historical valid queries match
byte-for-byte in the native facade. Trained TinyEngram/Qwen checks include CPU
and CUDA output comparisons and timing controls. Official DeepSeek Engram
verification uses its small demo; no official trained Engram checkpoint is
claimed. Native serving is slower than dense/raw mmap on several measured
traces, and lossless structural sharing often finds no identical learned rows.

The original .z exporter now checks whole-key structure, cosine and norm ratio
against final representatives; earlier norm-only audits are historical.
Approximate .z sharing and joint mode must not be called lossless.

## Current entry points

- Repository: https://github.com/initial-d/ngram-store
- Engine guide: [pack, inspect, query](evaluator_quickstart.md)
- Word-embedding application: [FastText](fasttext_native_bundle.md)
- LLM-memory application: [Engram](engram_quickstart.md)
- [Trained evidence index](index.md)
- [Original paper and software citation](citation_guide.md)

`ngram_store` and `ngram-store` are the engine import and command names.
`meft-memory`, `meft_memory`, old C ABI symbols and FastText formats remain
compatibility identifiers. The paper retains its original title; later software
results should be cited by exact commit/report and not retroactively attributed
as paper experiments.
