# Research plan: exact high-order n-gram memory storage and serving

## Research question

Can exact indexing and byte-preserving storage keep high-order n-gram memory
tractable while preserving a frozen model's lookup results and reducing actual
serving cost? The existing C ABI and PyTorch adapters provide a reproducible
engineering starting point, not a demonstrated LLM performance improvement.

## Hypotheses and competing explanations

| Hypothesis | What would support it | What could refute or explain it |
| --- | --- | --- |
| Longer n-gram keys benefit more from shared-prefix indexing | Same-source key storage/query comparisons by order; all identities preserved | Diverse keys share little structure; index metadata outweighs savings |
| Larger trained memory tables offer exact-byte redundancy | Frozen learned-row hashes and dedup rates by order/layer/head | Random/learned rows remain unique; only quantization creates duplicates |
| Native batched decoding reduces serving cost | Same-ID byte-exact outputs, latency/throughput/RSS versus raw mmap and tensor gather | Resident tensor gathers remain faster; sorting/decode/transfer dominates |
| Higher-order memory improves model task quality | Trained architecture ablation with equal capacity/compute and strong baselines | Additional capacity/training causes apparent gain; sparse keys harm generalization |

Key compression, exact row deduplication, block codecs and approximate vector
sharing are different mechanisms. Approximate sharing must have its own error
and downstream-quality study and cannot be advertised as lossless.

## Experimental design before learned-weight claims

Freeze checkpoint/source hash, tokenizer normalization/hash settings, layers,
head sizes, dtype, row dimensions and test queries. Compare both fixed-total-
memory and fixed-per-head-capacity protocols across max orders 2/3/4/6/8/12;
the current synthetic sweep uses only the latter and cannot isolate all effects.
Include identity-order shuffling to separate lexical/index locality from accidental
row placement, and matched request counts as well as full per-token fan-out.

Baselines: ordinary tensor gather; raw mmap; block codec without dedup;
exact-byte dedup plus block codec; exact trie/front-coded key variants when a
working query implementation exists. Report maps/descriptors, metadata,
construction peak memory/disk, cold and warm load/RSS, p50/p95/p99, throughput,
batch size, worker count, hit/locality distribution and CPU/GPU transfers.
Record repeated runs and variation rather than choosing the fastest run.

Acceptance for storage: all rows and arbitrary queries preserve IDs/dtype/bytes,
including boundaries/unseen keys/hash collisions/repetition. Acceptance for a
model: controlled deterministic embedding outputs, gated states/logits and
generation/quality checks. Then measure accuracy/perplexity/task metrics and
cost on the trained model, with retained negative results.

## Paper and project positioning

The current paper is about FastText. Describe the native high-order memory
direction as a follow-up systems artifact; do not rewrite its old reported
benchmarks as Engram evidence. The legacy norm-ratio/cosine mismatch remains
documented and should be corrected/reproduced in a dedicated change.

A credible follow-up contribution would combine exact high-order key storage,
scalable immutable-table construction, trace-aware batch/prefetch scheduling,
and matched-cost learned-model results. A C wrapper and synthetic order curve
alone are insufficient for a model/research superiority claim.

## Distribution and reusable evidence

Publish a source-linked technical article and one-command reproduction with the
current limits. Update each result only after new evidence; avoid generic
keyword/listing churn. Offer a specific benchmark request: a licensed frozen
table or serving trace with original hash IDs, shapes and dtype. Keep the
project name/ABI stable and make model adapters discoverable by tested boundary.
External outreach and social posting require explicit destination/content
authorization; the repository's technical materials can be shared now.
