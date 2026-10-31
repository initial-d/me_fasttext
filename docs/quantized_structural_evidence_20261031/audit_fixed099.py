"""Post-main-run fixed-default quality audit, separate from model selection.

Uses the already declared 0.99 bundles; does not replace failed validation
choices. This additional test audit is not preregistered confirmatory evidence.
Run inside the pinned study checkout with the original locally cached inputs.
"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import torch

from tools.benchmark_quantized_structural import score_bundle, file_sha
from tools.benchmark_causal_lm_compression import text_splits, paired_ppl_interval
from tools.causal_ngram_lm import CausalNgramLM, encode_segment, evaluate, metric_summary, prediction_delta


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.study_root
    report_path = root / "results/report.json"
    report = json.loads(report_path.read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != report["commit"] or report["failures"] or len(report["runs"]) != 18:
        raise ValueError("complete pinned primary study required")
    corpus = Path("artifacts/native-ngram-20261018/corpus/tinyshakespeare.txt")
    if file_sha(corpus) != report["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    training = json.loads(Path("artifacts/causal-lm-compression-20261024/report.json").read_text())
    text = text_splits(corpus.read_text(), training["settings"]["max_characters"])["test"]
    cfg = Namespace(**report["settings"], native_library=Path("build/libmeft_rows.so"))
    checks = []
    for run in report["runs"]:
        if run["category"] != "gru":
            continue
        checkpoint = Path("artifacts/causal-lm-compression-20261024") / f"order-{run['max_order']}-seed-{run['seed']}/reference.pt"
        if file_sha(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = CausalNgramLM(len(state["registry"]["characters"]) + 1, len(state["registry"]["grams"]), state["dim"])
        model.load_state_dict(state["model_state"])
        model.eval()
        original = model.memory
        segment = encode_segment(text, state["registry"])
        baseline = evaluate(model, segment, cfg.burn_in, cfg.eval_chunk)
        variants = {v["name"]: v for v in run["variants"]}
        for encoding in ("fp32", "fp16", "int8"):
            name = encoding + "-joint-0.99"
            directory = root / "results" / f"gru-order-{run['max_order']}-seed-{run['seed']}" / name
            if file_sha(directory / "table.men") != variants[name]["metadata"]["table_sha256"]:
                raise ValueError("frozen bundle hash mismatch")
            scores = score_bundle(model, original, directory, segment, cfg)
            delta = prediction_delta(scores, baseline)
            checks.append({"max_order": run["max_order"], "seed": run["seed"], "encoding": encoding,
                "name": name, "checkpoint_sha256": run["checkpoint_sha256"],
                "test": metric_summary(scores), "test_delta": delta,
                "test_paired_block_interval": paired_ppl_interval(scores, baseline, seed=cfg.eval_seed),
                "test_all_budgets_passed": delta["perplexity_relative_change"] <= cfg.max_ppl_increase
                    and delta["mean_prediction_kl_nats"] <= cfg.max_kl and delta["top1_changed_fraction"] <= cfg.max_top1_flip})
    result = {"category": "post-main-run fixed-default test audit; not preregistered confirmatory evidence",
              "primary_report_sha256": file_sha(report_path), "commit": commit,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
              "pytorch": torch.__version__, "platform": platform.platform(), "device": "cpu", "gpu": "unused",
              "threads": 1, "interop_threads": 1, "primary_validation_choices_unchanged": True,
              "quality_budgets": {"ppl_relative": cfg.max_ppl_increase, "mean_kl_nats": cfg.max_kl, "top1_flip_fraction": cfg.max_top1_flip},
              "checks": checks}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
