# NGramStore guides and evidence

## Engine and application entry points

- [Engine: pack, inspect and query](evaluator_quickstart.md)
- [C ABI, native construction and runtime boundaries](native_ngram_memory.md)
- [Engine roadmap, priorities and acceptance gates](ngram_memory_research_plan.md)
- [Word embeddings: FastText prepare/train/export/query](fasttext_native_bundle.md)
- [LLM memory: official Engram demo and factory](engram_quickstart.md)
- [TinyEngram/Qwen checkpoint integration](tinyengram_qwen_evidence_20261030/README.md)
- [Historical me_fasttext compatibility](legacy_fasttext.md)
- [Citation guide](citation_guide.md) and [artifact card](artifact_card.md)

## Source-linked trained results

- [Actual legacy FastText model and matched layout controls](fasttext_native_evidence_20261031/README.md)
- [Original DATrie / mark-compact contribution ablations](structural_native_evidence_20261030/README.md)
- [Same-trained-table FP16/INT8 and structural-sharing controls](quantized_structural_evidence_20261031/README.md)
- [1.85M trained-row CPU capacity](cpu_capacity_evidence_20261030/README.md)
- [Large FP16/INT8 capacity, construction RSS and fresh CPU traces](quantized_capacity_evidence_20261031/README.md)
- [Large-table structural applicability, shared updates and initialization](large_structural_evidence_20261031/README.md)
- [Sparse-block native miss latency](sparse_blocks_evidence_20261030/README.md)
- [Native query/cache controls](native_query_evidence_20261030/README.md)
- [Causal language-model quality controls](causal_lm_evidence_20261024/README.md)
- [Held-out lossless compression](heldout_compression_evidence_20261024/README.md)
- [Native embedding-bag memory and latency](embedding_bag_evidence_20261025/README.md)
- [Trained n-gram similarity](trained_ngram_similarity_20261018.md)
- [Approximate vector distortion curves](approximate_ngram_compression_20261018.md)
- [Qwen CUDA inference timing controls](qwen_cuda_performance_20261030/README.md)
- [A100 output correctness](cuda_a100_evidence_20261030/README.md)

These reports keep their original run dates, source commits, commands and
historical project names. A rename does not change their evidence or scope.
Official Engram demo fixtures, trained references and TinyEngram/Qwen results
are separate categories. Raw mmap/dense controls and no-benefit results remain
close to claimed savings. [Contribution/reporting guide](../CONTRIBUTING.md).
