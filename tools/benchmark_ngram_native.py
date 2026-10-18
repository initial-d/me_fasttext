"""Controlled n-gram order/key-storage/native-gather experiment, synthetic only.

Uses pinned official hash/gather definitions. Separate key encoding from random
weight storage: key repetition does not imply byte-identical learned vectors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import zlib

import numpy as np
import torch

try:
    from .demo_engram_adapter import official_definitions, measure, tensor_bytes, SOURCE_COMMIT
    from .engram_row_adapter import export_demo_embedding, MappedMultiHeadEmbedding
except ImportError:
    from demo_engram_adapter import official_definitions, measure, tensor_bytes, SOURCE_COMMIT
    from engram_row_adapter import export_demo_embedding, MappedMultiHeadEmbedding
from types import SimpleNamespace


def key_metrics(tokens, order):
    keys = sorted({tuple(int(x) for x in row[t:t + order])
                   for row in tokens for t in range(len(row) - order + 1)})
    flat = b"".join(struct.pack("<" + "I" * order, *key) for key in keys)
    prefix_encoded = bytearray()
    previous = ()
    decoded = []
    for key in keys:
        common = 0
        while common < min(len(previous), len(key)) and previous[common] == key[common]:
            common += 1
        suffix = key[common:]
        prefix_encoded.extend(struct.pack("<HH", common, len(suffix)))
        prefix_encoded.extend(struct.pack("<" + "I" * len(suffix), *suffix))
        previous = key
    previous, cursor = (), 0
    while cursor < len(prefix_encoded):
        common, suffix_len = struct.unpack_from("<HH", prefix_encoded, cursor)
        cursor += 4
        suffix = struct.unpack_from("<" + "I" * suffix_len, prefix_encoded, cursor)
        cursor += 4 * suffix_len
        previous = previous[:common] + suffix
        decoded.append(previous)
    assert decoded == keys
    windows = tokens.shape[0] * (tokens.shape[1] - order + 1)
    return {"windows": windows, "unique_keys": len(keys), "flat_key_bytes": len(flat),
            "prefix_key_bytes": len(prefix_encoded), "prefix_to_flat_ratio": len(prefix_encoded) / len(flat),
            "zlib_flat_key_bytes": len(zlib.compress(flat)), "keys_roundtrip_exact": True,
            "key_sha256": hashlib.sha256(flat).hexdigest(),
            "note": "Sorted front-coded key bytes, not a random-access trie or trained weight compression."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hasher, Original = official_definitions(args.official_source)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(7)
    # Key workloads: diverse independent tokens and repeated template prefixes.
    diverse = rng.integers(1, 512, size=(16, 128), dtype=np.int64)
    templates = rng.integers(1, 512, size=(8, 32), dtype=np.int64)
    structured = np.vstack([np.concatenate([templates[(i + j) % 8] for j in range(4)])
                            for i in range(16)])
    # Alter phrase tails; retain long shared prefixes but create different keys.
    structured[:, 31::32] = rng.integers(512, 1024, size=(16, 4))
    result = {
        "category": "controlled synthetic n-gram order and native gather experiment; not trained model evidence",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "official_commit": SOURCE_COMMIT, "exact_command": " ".join([sys.executable, *sys.argv]),
        "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "device": "cpu", "seed": 7,
        "threads": {"torch": 1, "interop": 1},
        "library_sha256": hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
        "order_maxima": [2, 3, 4, 6, 8, 12], "heads_per_order": 2,
        "embedding_dim": 16, "dtype": "torch.float32", "block_rows": 64,
        "fixed_head_sizes": [257, 263], "cases": [],
        "limitations": [
            "Untrained random embedding rows and synthetic tokens; no accuracy, trained checkpoint or logits.",
            "Higher max order adds heads/tables linearly here; fixed per-head capacity is a control, not optimal model allocation.",
            "Key prefix encoding is an offline index-footprint comparison, not a query backend.",
            "C gather has no persistent decoded-block cache; Python reference uses cache_blocks=0 for byte-gather comparison.",
            "All latencies are warm 30-call CPU microbenchmarks, not cold I/O, RSS or GPU-transfer measurements.",
        ],
    }
    with torch.inference_mode():
        for maximum in result["order_maxima"]:
            sizes = [257, 263] * (maximum - 1)
            torch.manual_seed(100 + maximum)
            original = Original(sizes, 16).eval()
            directory = args.output_dir / f"order-{maximum}"
            native_meta = export_demo_embedding(original, sizes, directory / "native", 64, backend="native")
            py_meta = export_demo_embedding(original, sizes, directory / "python", 64)
            with MappedMultiHeadEmbedding(directory / "native", native_library=args.native_library) as native, \
                 MappedMultiHeadEmbedding(directory / "python", cache_blocks=0) as python:
                config = SimpleNamespace(max_ngram_size=maximum, n_head_per_ngram=2, pad_id=0,
                                         layer_multipliers={1: np.arange(17, 17 + maximum * 2, 2, dtype=np.int64)},
                                         vocab_size_across_layers={1: [[257, 263]] * (maximum - 1)})
                for label, tokens in (("diverse", diverse), ("shared_prefix", structured)):
                    ids = torch.from_numpy(hasher(config, tokens[:2, :64], layer_id=1))
                    expected = original(ids)
                    assert tensor_bytes(native(ids)) == tensor_bytes(expected)
                    assert tensor_bytes(python(ids)) == tensor_bytes(expected)
                    flat_ids = (ids + original.offsets).reshape(-1).tolist()
                    native.table.rows(flat_ids)
                    block_reads = native.table.last_block_reads
                    decodes = native.table.last_decompressions
                    key = key_metrics(tokens, maximum)
                    nbytes = native_meta["artifact_bytes"] + (directory / "native/embedding.json").stat().st_size
                    pbytes = py_meta["artifact_bytes"] + (directory / "python/embedding.json").stat().st_size
                    result["cases"].append({
                        "max_order": maximum, "workload": label, "keys": key,
                        "weight_input_bytes": native_meta["input_bytes"],
                        "native_artifact_bytes": nbytes, "native_ratio": nbytes / native_meta["input_bytes"],
                        "python_artifact_bytes": pbytes, "native_unique_rows": native_meta["unique_rows"],
                        "weight_sha256": native_meta["weight_sha256"],
                        "head_ids_sha256": hashlib.sha256(tensor_bytes(ids)).hexdigest(),
                        "query_row_count": ids.numel(), "output_sha256": hashlib.sha256(tensor_bytes(expected)).hexdigest(),
                        "native_and_python_gather_byte_exact": True,
                        "blocks_per_batch": block_reads, "decodes_per_batch": decodes,
                        "official_gather": measure(lambda model=original, query=ids: model(query)),
                        "native_gather": measure(lambda model=native, query=ids: model(query)),
                        "python_gather": measure(lambda model=python, query=ids: model(query)),
                    })
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
