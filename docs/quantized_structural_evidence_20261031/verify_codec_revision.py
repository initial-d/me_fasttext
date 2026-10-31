"""Check the finite-range codec fix against all cached study inputs.

Run from this trusted repository. Executes the pinned study's own codec source
from Git to compare encoded bytes; never loads arbitrary network code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import quantized_memory as current


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sgns-dir", type=Path, required=True)
    p.add_argument("--gru-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raw = args.report.read_bytes()
    study = json.loads(raw)
    source = subprocess.check_output(["git", "show", study["commit"] + ":tools/quantized_memory.py"], cwd=ROOT)
    if sha(source) != study["source_sha256"]["tools/quantized_memory.py"]:
        raise ValueError("pinned codec source differs from study")
    reference = {"__name__": "tools._pinned_study_codec", "__package__": "tools"}
    exec(compile(source, "pinned study codec", "exec"), reference)
    checks = []
    for run in study["runs"]:
        folder = f"order-{run['max_order']}-seed-{run['seed']}"
        if run["category"] == "sgns":
            rows = np.frombuffer((args.sgns_dir / folder / "trained-ngram.f32").read_bytes(), dtype="<f4").reshape(-1, run["dim"])
        else:
            state = torch.load(args.gru_dir / folder / "reference.pt", map_location="cpu", weights_only=True)
            rows = state["model_state"]["memory.weight"].numpy().astype("<f4")
        if sha(rows.tobytes()) != run["source_float32_sha256"]:
            raise ValueError("cached trained rows differ from study")
        variants = {v["name"]: v for v in run["variants"]}
        for encoding in ("fp32", "fp16", "int8"):
            old = reference["encode_rows"](rows, encoding)
            new = current.encode_rows(rows, encoding)
            decoded = current.decode_rows(new, run["dim"], encoding)
            expected_sha = variants[encoding + "-raw"]["metadata"]["decoded_float32_sha256"]
            if old.tobytes() != new.tobytes() or sha(decoded.tobytes()) != expected_sha:
                raise ValueError("codec revision changed a trained study table")
            checks.append({"category": run["category"], "max_order": run["max_order"], "seed": run["seed"],
                           "encoding": encoding, "source_float32_sha256": run["source_float32_sha256"],
                           "encoded_sha256": sha(new.tobytes()), "decoded_sha256": expected_sha,
                           "encoded_bytes_identical": True, "decoded_bytes_identical": True})
    result = {"study_commit": study["commit"], "report_sha256": sha(raw),
              "revision_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "revision_codec_sha256": sha((ROOT / "tools/quantized_memory.py").read_bytes()),
              "verification_script_sha256": sha(Path(__file__).read_bytes()),
              "command": shlex.join([sys.executable, *sys.argv]), "working_directory": str(ROOT),
              "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": "cpu", "gpu": "unused",
              "all_trained_encodings_unchanged": True, "checks": checks,
              "scope": "byte compatibility of a finite-range guard; benchmark timings remain tied to the original study commit"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
