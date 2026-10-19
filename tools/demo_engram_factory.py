"""Verify the proposed Engram factory with a real mapped backend, offline.

The source must match the published draft-PR commit, not official main. Only
tokenizer construction is replaced by local normalized IDs. Random weights are
integration fixtures, not trained Engram compression evidence.
"""
import argparse
from functools import partial
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

try:
    from .demo_engram_full_adapter import IdentityTokenizer
    from .demo_engram_adapter import tensor_bytes
    from .engram_row_adapter import export_demo_embedding, open_demo_embedding
except ImportError:
    from demo_engram_full_adapter import IdentityTokenizer
    from demo_engram_adapter import tensor_bytes
    from engram_row_adapter import export_demo_embedding, open_demo_embedding

PROPOSAL_COMMIT = "8b4155c2f8fc886ef27973d1164b798126f90c0d"
PROPOSAL_SHA256 = "f536d39f229d6352828709960593b7fcc9b5a3ca1025e5c5c785e78660d2a041"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-source", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.proposal_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROPOSAL_SHA256:
        raise ValueError("source differs from the published draft-PR commit")
    spec = importlib.util.spec_from_file_location("meft_engram_factory", args.proposal_source)
    demo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = demo
    spec.loader.exec_module(demo)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = demo.EngramConfig(engram_vocab_size=[11, 19], max_ngram_size=3,
                              n_embed_per_ngram=8, n_head_per_ngram=2,
                              layer_ids=[1], pad_id=0, kernel_size=3)
    backbone = demo.BackBoneConfig(hidden_size=8, hc_mult=2, vocab_size=32, num_layers=2)
    ids = np.array([[0, 1, 2, 3, 0], [3, 3, 2, 1, 0]], dtype=np.int64)
    report = {"category": "draft-PR factory integration, tiny random CPU weights, no tokenizer/checkpoint download",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "exact_command": " ".join([sys.executable, *sys.argv]),
              "upstream_pr": "https://github.com/deepseek-ai/Engram/pull/27",
              "proposal_commit": PROPOSAL_COMMIT, "proposal_source_sha256": PROPOSAL_SHA256,
              "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": "CPU", "threads": 1, "seed": 7,
              "native_library_sha256": hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
              "limitations": ["Proposed fork API, not merged official main.", "Tiny random weights verify integration only.",
                              "No tokenizer fidelity, trained Engram compression, logits, GPU or task-quality result.",
                              "Default allocation is checked, but total process memory/serving speed is not measured."], "cases": []}
    with patch.object(demo, "engram_cfg", config), patch.object(demo, "backbone_config", backbone), \
            patch.object(demo, "CompressedTokenizer", IdentityTokenizer), torch.inference_mode():
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for backend, predictor in (("python", "none"), ("native", "none"), ("native", "xor")):
                torch.manual_seed(7)
                baseline = demo.Engram(layer_id=1).to(dtype=dtype).eval()
                hidden = torch.randn(2, 5, 2, 8, dtype=dtype)
                sizes = [n for group in baseline.hash_mapping.vocab_size_across_layers[1] for n in group]
                directory = args.output_dir / (str(dtype).split(".")[-1] + "-" + backend + "-" + predictor)
                export_demo_embedding(baseline.multi_head_embedding, sizes, directory, 8,
                                      backend=backend, predictor=predictor)
                factory = partial(open_demo_embedding, directory, native_library=args.native_library)
                mapped = factory(list_of_N=sizes, D=4)
                try:
                    def recorded_factory(list_of_N, D, expected_sizes=sizes, opened=mapped):
                        if list_of_N != expected_sizes or D != opened.embedding_dim:
                            raise ValueError("factory parameters differ")
                        return opened
                    with patch.object(demo, "MultiHeadEmbedding", side_effect=AssertionError("default table allocated")):
                        candidate = demo.Engram(layer_id=1, embedding_factory=recorded_factory).to(dtype=dtype).eval()
                    downstream = {k: v for k, v in baseline.state_dict().items() if not k.startswith("multi_head_embedding.")}
                    status = candidate.load_state_dict(downstream, strict=False)
                    if status.unexpected_keys or any(not k.startswith("multi_head_embedding.") for k in status.missing_keys):
                        raise AssertionError("downstream weights not fully restored")
                    before, after = baseline(hidden, ids), candidate(hidden, ids)
                    if tensor_bytes(before) != tensor_bytes(after):
                        raise AssertionError("full Engram output changed")
                    report["cases"].append({"dtype": str(dtype), "backend": backend, "predictor": predictor,
                                            "default_table_allocation_avoided": True, "full_output_byte_exact": True,
                                            "head_sizes": sizes, "shape": list(after.shape),
                                            "output_sha256": hashlib.sha256(tensor_bytes(after)).hexdigest()})
                finally:
                    mapped.close()
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
