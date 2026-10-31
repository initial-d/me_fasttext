# NGramStore engine roadmap and research plan

## Historical background: n-gram embeddings to LLM memory

Early learned n-gram embeddings represent local patterns as trainable vectors.
FastText's subword vectors contribute to word representations. Newer Engram-style
LLM architectures retrieve learned token-n-gram features from tables and fuse
them with hidden states. This gives the same addressable-vector abstraction a
role as an explicit learned memory module alongside the LLM backbone.

Here, n-gram parametric memory means learned lookup-table weights. This is one
branch of model memory, with a common storage/query problem across application
generations. The continuity is the table abstraction; character versus token
keys, hash versus exact addressing, training and context fusion remain distinct.

NGramStore's historical development follows that abstraction: the original
me_fasttext method optimizes n-gram indexing and memory management for word
embeddings; the shared engine extends storage and queries to frozen learned
tables used by LLM memory modules. The later software evidence keeps its own
scope and does not rewrite the original paper's experiments.

## Product contract and history

NGramStore is a general-purpose backend for **learned n-gram embeddings**:
compression, storage and exact lookup. The reusable unit is a frozen embedding
table, independent of the model architecture that trained or consumes it.

1. **Shared table engine:** preserve original IDs, shape, dtype and row bytes;
   encode blocks, forward IDs, mmap, cache and gather.
2. **Optional n-gram structure:** registered unique string keys enable DATrie
   indexing, prefix/suffix selection and mark-compact export. Numeric/hash IDs
   work without this layer; colliding buckets do not certify string relations.
3. **Model applications:** FastText is the historical word-embedding use case;
   Engram/TinyEngram is the LLM-memory use case. Models keep training,
   normalization, addressing, aggregation, gates and downstream state.

This contract covers immutable inference tables. The [CLI](evaluator_quickstart.md)
currently packs little-endian float32; lower-level dtype support does not imply
every exporter supports every dtype. [Runtime boundaries](native_ngram_memory.md)
remain part of the contract.

NGramStore continues me_fasttext's repository history. The original paper's
DATrie/mark-compact method is the origin; native codecs, query scheduling and
LLM adapters are subsequent software contributions. Preserve paper attribution,
file readers, C symbols and Python aliases; cite later results with their own
source commits. The [legacy audit](audits/legacy-roundtrip-20261024/README.md) and
[structural ablation](structural_native_evidence_20261030/README.md) cover the
corrected cosine/norm-ratio guards. Historical reports retain their run names.

## Why the backend matters

Growing n-gram coverage forces separate choices about **identity**, **capacity**
and **vector fidelity**. These choices interact but solve different problems.

| Approach | What it buys | Cost or risk | NGramStore boundary |
| --- | --- | --- | --- |
| Fixed-size hash buckets, common in n-gram embedding designs | Bounded row count and cheap addressing | Different n-grams can share one row and its updates, losing independent identities; actual task impact depends on occupancy, distribution and model design | Existing hash IDs remain unchanged. A unique key registry with distinct source rows can instead use exact DATrie lookup |
| Full distinct-key vector table | Independent rows without bucket collisions | Key/index, vector and metadata storage grow with covered types; construction RAM and query working set also matter | Compress keys and row payloads separately; count complete artifacts and process costs |
| Quantization or approximate vector sharing | Smaller payloads, sometimes substantially | Quantization rounds source values; joint sharing replaces rows. Vector errors and downstream task effects require measurement | Byte-preserving modes retain the exported source bytes; approximation is explicit and separately evaluated |

Hashing is not universally the default and collisions are not automatically
harmful in every trained architecture. Their impact must be measured, rather
than inferred from the mere presence of a hash. Increasing bucket capacity
reduces collision pressure at a storage cost. Higher n-gram order can enlarge
key diversity and query fan-out, but the number of observed types and bucket
occupancy depend on the corpus and allocation policy.

A collision shares parameters rather than merely causing a lookup failure:
updates from distinct keys can affect the same vector. Multiple independent
hash heads may mitigate this, so report the actual addressing design. A plain
distinct-row table costs N × d × bytes-per-value before keys, maps and metadata;
covering more observed types or using wider vectors raises this baseline.

Exact indexing can avoid **additional addressing collisions** for independently
represented registered keys. It cannot unmix information already combined in a
trained hash bucket or manufacture missing per-key vectors. Changing a model
from bucket sharing to independent rows is a model/training change, not a
byte-preserving storage conversion.

Likewise, packing an already quantized table preserves that table's bytes, not
the original pre-quantization weights. Exact compression offers a storage
option without introducing new vector error, but unique learned float rows may
compress only modestly, and decoding may cost more than dense/raw mmap lookup.

## Recommended training-to-serving lifecycle

Train independent rows for a **selected, affordable n-gram vocabulary**, then
optimize frozen storage and queries. Full coverage means the chosen vocabulary,
not enumeration of every possible n-gram. Select observed types, normalization,
order limits and frequency/coverage policy according to training resources.
Exact post-training compression does not remove the cost of training those rows.

| Starting point | Integration route | What it establishes |
| --- | --- | --- |
| Existing trained table with distinct rows and registered keys | Export the original key-to-ID registry and rows, pack, verify bytes, replace only frozen lookup | Storage/query optimization without retraining; already-shared bucket rows remain shared |
| Stable vocabulary for the next training run | Fix unique keys and original IDs at preprocessing; train with a mutable embedding table; freeze, export and query through NGramStore using the same IDs | Exact addressing can be part of the model design from the start; useful quality/cost differences still require trained controls |

The stable-vocabulary route is a model-side integration recipe, not a claim
that the native backend supports training. Its current handles are frozen;
there is no native gradient/update contract. DATrie can provide an exact key
index for a registered vocabulary, while the model's trainable table owns
parameter updates. Expanding the vocabulary or reallocating rows requires an
explicit new registry/checkpoint version and rebuilt frozen artifact.

Keep a dense or raw-mmap option for tables where exact compression adds cost
without meaningful savings. Approximation remains a separately chosen quality
budget, not a prerequisite for the full-vocabulary route.

## Adoption goal

Make NGramStore a routine evaluation option when a team serves a large n-gram
embedding table. A user should export existing rows, verify unchanged addressing
and outputs, measure the storage/serving tradeoff, and replace only the table
backend. Industry-wide adoption remains a goal, not a measured result.

Track independent integrations/reproductions, retained use after evaluation and
source-linked software/paper citations. Stars and article views cannot establish
those outcomes. Failed or no-benefit evaluations identify where the engine needs
work.

## Current baseline

| Capability | Existing evidence | Unresolved constraint |
| --- | --- | --- |
| Original-ID/byte-preserving storage and adapters | [Actual FastText outputs](fasttext_native_evidence_20261031/README.md), [trained TinyEngram/Qwen](tinyengram_qwen_evidence_20261030/README.md), Linux/macOS wheel CI | Frozen handles; reduction contracts differ; official Engram verification uses a demo, not trained weights |
| Sparse native queries | [1.85M trained rows](cpu_capacity_evidence_20261030/README.md) and [smaller blocks](sparse_blocks_evidence_20261030/README.md) | Misses improved 16–27× over the prior layout; raw mmap remains faster |
| DATrie / mark-compact | [Trained SGNS/GRU ablations](structural_native_evidence_20261030/README.md) | Exact key indexing helps these traces; no identical structural rows found; extra joint row savings are approximate |
| Quantization + structural sharing | [18 trained tables, 486 controls](quantized_structural_evidence_20261031/README.md) | Additional savings remain after FP16/INT8 on these references; low-order INT8 can regress; validation-selected test failures are retained |
| Quantized scale and CPU serving | [1.85M rows, 54 fresh workers](quantized_capacity_evidence_20261031/README.md) | FP16 native saves storage with broad-query cost; INT8 native expands versus raw INT8; vector error is measured without large-table task quality |
| End-to-end artifact cost | [FastText matched controls](fasttext_native_evidence_20261031/README.md) | Whole bundle saves only 0.46%; queries are slower; structural order adds no benefit on this model |
| Construction/runtime resources | [Streaming quantized files and resource controls](quantized_capacity_evidence_20261031/README.md), [implementation boundaries](native_ngram_memory.md) | File codecs/hashes use chunks; native descriptors and optional planner indexes still grow; structural planning copies full rows/maps; cache budgets do not bound RSS; FastText retains an unused 200 MB hash allocation |

Correctness is a starting point on supported paths. Useful compression and
serving gains still depend on the trained table and workload.

## Ordered engineering priorities

Stable export/query contracts and historical compatibility are release gates
throughout. Optimize measured bottlenecks before adding another named-model adapter.

| Priority | Concrete deliverable | Acceptance gate |
| --- | --- | --- |
| 1. Sparse-query serving cost | Trace-driven block/cache comparison, then improve miss decoding and batch scheduling at the measured bottleneck | Same trained rows and ID trace versus dense, raw mmap and current native. Report cold/first-query and warm p50/p95/p99, throughput, full artifact bytes and RSS. Include any storage/resource regression; a gain over prior native does not imply a win over raw mmap |
| 2. N-gram structural benefit | Matched DATrie/mark-compact ablation and a reversible structural predictor/layout candidate | Compare identity, shuffled and structural layouts under the same codec/cache settings. Count keys, maps and residual metadata. Exact mode reconstructs every byte; joint mode reports row error and held-out task quality. Retain candidates for workloads where complete costs improve |
| 3. Large-table construction and process memory | Profile staging/maps and runtime allocations; remove avoidable copies/legacy allocations and implement chunked construction where feasible | Sweep table size at fixed row width and a declared RAM budget. Record build time, peak RSS, temporary disk and final bytes; verify rows and failure cleanup. Describe remaining O(N) maps; cache budgets alone do not prove bounded RSS |
| 4. Independent reuse | Third-party trained-table export and query integration through the common engine API | An independent user reproduces with a pinned environment and licensed/public artifact or documented manifest. Model internals need not change; original IDs/dtypes/outputs and dense/raw-mmap controls stay inspectable |

CPU traces suffice for decoding and resource comparisons. GPU measurements are
required only for device-transfer or whole-model serving claims. Separate key
indexing, row layout, exact residual compression and approximate sharing gains.

Show tradeoffs across full artifact size, latency and memory. Declare workload
and resource/quality budgets before tuning; retain a separate held-out trace.
Keeping the original dense/raw-mmap path is valid when compression does not help.

## High-order n-gram research

Higher order is a testable structural opportunity, not a universal monotonic
gain. Keep key length, dimension, total capacity and query fan-out separate.

| Hypothesis | Supporting evidence to seek | Competing explanation |
| --- | --- | --- |
| Longer keys benefit from shared-prefix indexing | Same-source key storage/query comparisons by order with identities preserved | Diverse keys share little structure; index metadata outweighs savings |
| Shared training creates high-cosine structural rows | Trained prefix/suffix and identical-update distributions across orders, languages, seeds and training duration | Spelling alone does not imply shared gradients; sparse training and initialization matter |
| Structural prediction compresses similar rows without losing bits | Reversible residual codecs beat ordinary blocks/shuffled layouts including predictor overhead | Low-bit differences or decode work erase benefits |
| Larger memories contain exact-byte redundancy | Frozen row hashes/dedup rates by order/layer/head | Learned rows stay unique; only quantization creates duplicates |
| Batched decoding reduces serving cost | Byte-exact outputs and better latency/throughput/RSS tradeoffs | Tensor/raw-mmap gathers remain faster; scheduling or transfer dominates |
| Exact key addressing avoids harmful bucket interference | Matched trained key-index versus bucket baselines with occupancy/collision statistics and held-out quality, at fixed capacity/compute and separately fixed coverage | Larger capacity, changed training or multiple heads explain task differences; replacing existing buckets alone cannot restore per-key rows |

The [trained similarity reference](trained_ngram_similarity_20261018.md)
includes all nine runs and frequency strata: identical-update structural pairs
have median cosine about 0.996. Random vectors establish initialization/interface
controls only. High cosine does not establish identical bytes or a universal
ratio. The [native XOR codec](native_xor_compression_20261020.md) is a reversible
baseline; structural grouping must not depend on unavailable training signatures.

The [approximate curves](approximate_ngram_compression_20261018.md) constrain
representative cosine and norm ratio. At max order 12, joint 0.99 gives
2.14–2.16× n-gram table compression and 0.98 gives about 2.35× on that reference.
These are approximate table results, not whole-model gains or loss bounds.
The [SGNS/GRU structural study](structural_native_evidence_20261030/README.md)
finds a plateau from order 8 to 12 for SGNS and continued improvement for GRU.
Maximum per-row relative L2 error reaches 14.24% despite the 0.99 joint guard.
Retain tail errors and held-out model quality next to savings.

Whether higher-order memory improves task quality is a separate model experiment
requiring equal-capacity/compute trained ablations. The engine supports that
experiment without claiming its outcome.

## Common evidence and release gates

Freeze source/checkpoint hashes, preprocessing/tokenizer/hash rules, key
registry, layer/head layout, dimensions, dtype and query traces. For order sweeps,
compare fixed-total-memory and fixed-per-head-capacity across orders 2/3/4/6/8/12.
Include matched request counts and actual per-token fan-out. For addressing
comparisons, report distinct key count, allocated bucket count, occupancy and
collision distribution; do not treat exact storage as a free increase in capacity.

Use working dense/raw-mmap and ordinary-codec controls. Include exact-byte dedup,
quantization and alternative key indices when implemented, with actual query
costs and their own fidelity contracts. Report repeated-run variation, block/cache
budgets, workers, locality, load time, anonymous/file-backed RSS, build peak
memory/disk and transfers. Record commit, commands, toolchain, hardware,
data/license provenance and omissions with raw results. Preserve failed/slow runs.

Verify original IDs, shape/dtype and every reconstructed row byte, including
signed zero/nonfinite payloads, boundaries, repetition and registered keys.
Retain existing model collision semantics. Adapters also verify ordered
aggregation, embeddings, gated states/logits and controlled generation/task
metrics as applicable. Row identity does not prove a changed reduction order
preserves model outputs.

Release gates include installed C/Python/PyTorch paths on their declared platforms,
legacy imports/readers and explicit format versions. Do not promise mutable
weights, shared-worker handles or portable legacy serialization before validation.
Distribution/compatibility aliases stay in the [quickstart](evaluator_quickstart.md).

Use the [contribution guide](../CONTRIBUTING.md) for integrations or measurements.
A follow-up paper needs a specific verified algorithm/resource contribution and
matched trained-model evidence; adapters and visibility alone do not prove
research superiority.
