"""Offline full official Engram forward with exact CPU storage replacement.

Loads the hashed official source, replaces only tokenizer construction with a
tiny identity fixture, and runs original hashes/gates/projections/ShortConv.
Requires torch, numpy, sympy, transformers and tokenizers; downloads no model.
Random weights verify integration only, not compression or trained-model gains.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
import transformers
import tokenizers
import sympy

try:
    from .demo_engram_adapter import SOURCE_COMMIT, SOURCE_SHA256, tensor_bytes
    from .engram_row_adapter import export_demo_embedding, MappedMultiHeadEmbedding
except ImportError:
    from demo_engram_adapter import SOURCE_COMMIT, SOURCE_SHA256, tensor_bytes
    from engram_row_adapter import export_demo_embedding, MappedMultiHeadEmbedding


class IdentityTokenizer:
    """Local normalized-token fixture; not a tokenizer equivalence test."""
    def __init__(self, **_):
        self.lookup_table = np.arange(32, dtype=np.int64)

    def __len__(self):
        return len(self.lookup_table)

    def __call__(self, ids):
        return np.asarray(ids, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.official_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("official source differs from the pinned audited source")
    spec = importlib.util.spec_from_file_location("meft_official_engram", args.official_source)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = official.EngramConfig(engram_vocab_size=[11, 19], max_ngram_size=3,
                                   n_embed_per_ngram=8, n_head_per_ngram=2,
                                   layer_ids=[1], pad_id=0, kernel_size=3)
    backbone = official.BackBoneConfig(hidden_size=8, hc_mult=2, vocab_size=32, num_layers=2)
    ids = np.array([[0, 1, 2, 3, 0], [3, 3, 2, 1, 0]], dtype=np.int64)
    report = {"category": "full official Engram CPU forward integration on random weights, local normalized-token fixture",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "exact_command": " ".join([sys.executable, *sys.argv]),
              "official_commit": SOURCE_COMMIT, "official_source_sha256": SOURCE_SHA256,
              "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
              "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "sympy": sympy.__version__,
              "platform": platform.platform(), "device": "CPU", "threads": 1, "seed": 7,
              "native_library_sha256": hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
              "token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
              "executed": ["NgramHashMapping.__init__ and hash", "Engram.forward", "key/value projections", "RMSNorm gates", "ShortConv"],
              "limitations": ["Random small weights and local normalized tokens, not trained Engram weights or tokenizer fidelity.",
                              "No backbone/LM logits, held-out task quality, model-sized memory, latency or GPU measurements.",
                              "CPU inference only; complete checkpoint loading/lifecycle is caller responsibility."], "cases": []}
    with patch.object(official, "engram_cfg", config), patch.object(official, "backbone_config", backbone), \
            patch.object(official, "CompressedTokenizer", IdentityTokenizer), torch.inference_mode():
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for backend in ("python", "native"):
                torch.manual_seed(7)
                model = official.Engram(layer_id=1).to(dtype=dtype).eval()
                hidden = torch.randn(2, 5, 2, 8, dtype=dtype)
                before = model(hidden, ids)
                original = model.multi_head_embedding
                sizes = [n for group in model.hash_mapping.vocab_size_across_layers[1] for n in group]
                directory = args.output_dir / (str(dtype).split(".")[-1] + "-" + backend)
                meta = export_demo_embedding(original, sizes, directory, 8, backend=backend)
                hashes = model.hash_mapping.hash(ids)[1]
                with MappedMultiHeadEmbedding(directory, native_library=args.native_library) as mapped:
                    model.multi_head_embedding = mapped
                    after = model(hidden, ids)
                    exact = tensor_bytes(before) == tensor_bytes(after)
                    if not exact:
                        raise AssertionError("full gated output differs after replacement")
                    if mapped.table.rows(range(sum(sizes))) != tensor_bytes(original.embedding.weight):
                        raise AssertionError("source row mismatch")
                    report["cases"].append({"dtype": str(dtype), "backend": backend, "head_sizes": sizes,
                                            "full_output_byte_exact": exact, "all_rows_byte_exact": True,
                                            "shape": list(after.shape), "output_sha256": hashlib.sha256(tensor_bytes(after)).hexdigest(),
                                            "head_ids_sha256": hashlib.sha256(hashes.tobytes()).hexdigest(),
                                            "artifact_bytes": meta["artifact_bytes"], "input_bytes": meta["input_bytes"]})
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
