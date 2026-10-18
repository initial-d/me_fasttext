"""Offline comparison using only pinned official hash and gather definitions.

Requires torch/numpy and a locally saved official engram_demo_v1.py. It extracts
two verified definitions rather than running tokenizer downloads or the model.
Weights and normalized token IDs are random synthetic input, not a checkpoint.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import List
from unittest import mock

import numpy as np
import torch
from torch import nn

try:
    from .engram_row_adapter import MappedMultiHeadEmbedding, export_demo_embedding
    from . import lossless_row_store
except ImportError:
    from engram_row_adapter import MappedMultiHeadEmbedding, export_demo_embedding
    import lossless_row_store

SOURCE_COMMIT = "fb7f84a21f91223715394a33a1dc24bbfb7f788e"
SOURCE_SHA256 = "9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044"


def official_definitions(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("official source bytes differ from the audited pinned demo")
    tree = ast.parse(raw)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MultiHeadEmbedding":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "NgramHashMapping":
            selected.extend(child for child in node.body
                            if isinstance(child, ast.FunctionDef) and child.name == "_get_ngram_hashes")
    if len(selected) != 2:
        raise ValueError("expected official hash and gather definitions not found")
    namespace = {"np": np, "torch": torch, "nn": nn, "List": List}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_get_ngram_hashes"], namespace["MultiHeadEmbedding"]


def measure(call, repeats=30):
    for _ in range(5):
        call()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        elapsed.append((time.perf_counter_ns() - start) / 1000)
    return {"p50_us": statistics.median(elapsed),
            "p95_us": sorted(elapsed)[int(0.95 * (repeats - 1))], "repeats": repeats}


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    get_hashes, Original = official_definitions(args.official_source)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    # Tiny synthetic head vocabularies; normalized IDs bypass the tokenizer.
    sizes = [257, 263, 269, 271]
    config = SimpleNamespace(
        max_ngram_size=3, n_head_per_ngram=2, pad_id=0,
        layer_multipliers={1: np.array([17, 31, 47], dtype=np.int64)},
        vocab_size_across_layers={1: [[257, 263], [269, 271]]},
    )
    tokens = rng.integers(0, 128, size=(2, 32), dtype=np.int64)
    tokens[:, 8:16] = tokens[:, :8]  # repeated contexts, not duplicated weight rows
    hashes = get_hashes(config, tokens, layer_id=1)
    ids = torch.from_numpy(hashes)
    result = {
        "category": "official-demo-interface verification on random weights; no trained checkpoint",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "exact_command": " ".join([sys.executable, *sys.argv]),
        "official_commit": SOURCE_COMMIT, "official_source_sha256": SOURCE_SHA256,
        "definitions_executed": ["NgramHashMapping._get_ngram_hashes", "MultiHeadEmbedding"],
        "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(), "device": "cpu",
        "torch_threads": 1, "torch_interop_threads": 1, "seed": 7,
        "head_sizes": sizes, "embedding_dim": 16, "batch_shape": list(ids.shape),
        "block_rows": 32, "cache_blocks": 4,
        "normalized_token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "head_ids_sha256": hashlib.sha256(hashes.tobytes()).hexdigest(),
        "cases": {},
        "limitations": [
            "Random initialized weights and synthetic normalized token IDs, not a trained model.",
            "Only the official hash function and embedding gather are executed; tokenizer/backbone/gates are not run.",
            "CPU-only immutable gather, no GPU transfer, logits, task quality, cold-cache or RSS measurements.",
            "Small Python experiment; offline export memory usage does not scale to large tables.",
            "Full-file checksum validation is included in opening and is O(artifact bytes), not lazy startup.",
        ],
    }
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            name = str(dtype).split(".")[-1]
            source = Original(sizes, 16).to(dtype=dtype).eval()
            original_bytes = tensor_bytes(source.embedding.weight)
            out = args.output_dir / name
            meta = export_demo_embedding(source, sizes, out, 32)
            started = time.perf_counter()
            mapped = MappedMultiHeadEmbedding(out, 4)
            open_seconds = time.perf_counter() - started
            try:
                for row_id in range(sum(sizes)):
                    assert mapped.table.row(row_id) == tensor_bytes(source.embedding.weight[row_id])
                query_cases = {
                    "official_hash_ids": ids,
                    "repeated_batches": ids.repeat(2, 1, 1),
                    "noncontiguous": ids.transpose(0, 1),
                    "head_boundaries": torch.tensor([[0, 0, 0, 0], [n - 1 for n in sizes]]),
                    "empty": ids[:0],
                }
                checks = {}
                for label, query in query_cases.items():
                    expected = source(query)
                    actual = mapped(query)
                    assert tensor_bytes(expected) == tensor_bytes(actual)
                    assert expected.shape == actual.shape and expected.dtype == actual.dtype
                    checks[label] = {"byte_exact": True, "shape": list(actual.shape)}
                assert original_bytes == tensor_bytes(source.embedding.weight)
                assert np.array_equal(hashes, get_hashes(config, tokens, layer_id=1))
                global_ids = (ids + source.offsets).reshape(-1).tolist()
                with lossless_row_store.MappedRows(out / "table.mer", 0) as uncached:
                    with mock.patch.object(lossless_row_store.zlib, "decompress",
                                           wraps=lossless_row_store.zlib.decompress) as decode:
                        coalesced = uncached.rows(global_ids)
                        grouped_reads = decode.call_count
                    with mock.patch.object(lossless_row_store.zlib, "decompress",
                                           wraps=lossless_row_store.zlib.decompress) as decode:
                        sequential = b"".join(uncached.row(i) for i in global_ids)
                        individual_reads = decode.call_count
                    assert coalesced == sequential
                    grouped_timing = measure(lambda table=uncached, query=global_ids: table.rows(query))
                    sequential_timing = measure(lambda table=uncached, query=global_ids:
                                                b"".join(table.row(i) for i in query))
                case = copy.deepcopy(meta)
                case.update({
                    "metadata_bytes": (out / "embedding.json").stat().st_size,
                    "total_artifact_to_raw_ratio": ((out / "embedding.json").stat().st_size
                                                    + meta["artifact_bytes"]) / meta["input_bytes"],
                    "query_checks": checks, "all_source_rows_byte_exact": True,
                    "source_weight_and_hash_unchanged": True,
                    "gather_output_sha256": hashlib.sha256(tensor_bytes(source(ids))).hexdigest(),
                    "verified_open_seconds": open_seconds,
                    "official_gather_warm": measure(lambda original=source: original(ids)),
                    "adapter_gather_warm": measure(lambda adapter=mapped: adapter(ids)),
                    "uncached_coalesced_bytes": grouped_timing,
                    "uncached_sequential_bytes": sequential_timing,
                    "uncached_batch_block_decodes": grouped_reads,
                    "uncached_sequential_decodes": individual_reads,
                })
                result["cases"][name] = case
            finally:
                mapped.close()
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
