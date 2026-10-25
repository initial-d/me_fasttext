"""Native bags on existing trained causal memory: serving costs and task delta.

No random embedding benchmark and no retraining. Performance uses cyclic replay
of held-out suffix IDs at a larger batch size; task quality uses the original
complete chronological test split in the trained GRU. Not trained Engram evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.causal_ngram_lm import CausalNgramLM, encode_segment, evaluate, metric_summary, prediction_delta
from tools.benchmark_causal_lm_compression import text_splits
from tools.engram_row_adapter import MappedEmbedding, file_sha256
from tools.mapped_embedding_bag import MappedEmbeddingBag


def sha(data):
    return hashlib.sha256(data).hexdigest()


def peak_rss():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_reference(args, order):
    report = json.loads((args.training_dir / "report.json").read_text())
    run = next(r for r in report["runs"] if r["max_order"] == order and r["seed"] == args.seed)
    directory = args.training_dir / f"order-{order}-seed-{args.seed}"
    if file_sha256(directory / "reference.pt") != run["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from training evidence")
    corpus = args.corpus.read_bytes()
    if sha(corpus) != report["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    split = text_splits(corpus.decode("utf-8"), report["settings"]["max_characters"])
    if sha(split["test"].encode()) != report["split"]["test"]["sha256"]:
        raise ValueError("chronological test split changed")
    checkpoint = torch.load(directory / "reference.pt", weights_only=False, map_location="cpu")
    segment = encode_segment(split["test"], checkpoint["registry"])
    if sha(segment["memory_ids"].numpy().astype("<i8").tobytes()) != run["evaluation"]["test"]["memory_ids_sha256"]:
        raise ValueError("held-out original memory IDs changed")
    return report, run, directory, checkpoint, segment


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, run, directory, checkpoint, segment = load_reference(args, args.order)
    weight = checkpoint["model_state"]["memory.weight"]
    if not weight[0].eq(0).all():
        raise ValueError("reference zero-padding memory is not zero")
    ids = segment["memory_ids"].repeat((args.bags + len(segment["memory_ids"]) - 1) // len(segment["memory_ids"]), 1)[:args.bags].contiguous()
    weight_digest = sha(weight.view(torch.uint8).numpy().tobytes())
    artifact = directory / "lossless"
    table = None
    if args.worker == "resident-bag":
        module = torch.nn.EmbeddingBag.from_pretrained(weight, freeze=True, mode="mean", padding_idx=0).eval()
        def forward():
            return module(ids)
    elif args.worker == "mapped-gather":
        module = MappedEmbedding(artifact, native_library=args.native_library, native_cache_bytes=args.cache_bytes)
        table = module.gather.table
        denominator = ids.ne(0).sum(-1, keepdim=True).clamp_min(1)
        def forward():
            return module(ids).sum(-2) / denominator
    else:
        module = MappedEmbeddingBag(artifact, mode="mean", padding_idx=0, chunk_rows=args.chunk_rows,
                                    native_library=args.native_library, native_cache_bytes=args.cache_bytes)
        table = module.gather.table
        def forward():
            return module(ids)
    # Each mode has the same checkpoint/input-loading baseline. The original
    # weight is retained for diagnostics in all modes, including mapped modes.
    baseline = peak_rss()
    times = []
    try:
        with torch.no_grad():
            for i in range(args.warmup + args.repeat):
                start = time.perf_counter()
                output = forward()
                elapsed = time.perf_counter() - start
                if i >= args.warmup:
                    times.append(elapsed)
                if i < args.warmup + args.repeat - 1:
                    del output
            rss = peak_rss()
            counters = ({"block_reads": table.last_block_reads, "decompressions": table.last_decompressions,
                         "cache_hits": table.last_cache_hits, "cache_payload_bytes": table.cache_bytes} if table else None)
            sample = torch.unique(torch.linspace(0, len(ids) - 1, 256).long())
            oracle = weight[ids[sample]].double().sum(-2) / ids[sample].ne(0).sum(-1, keepdim=True).clamp_min(1)
            error = (output[sample].double() - oracle).abs()
            byte_exact = None
            if table:
                digest = hashlib.sha256()
                for start in range(0, len(weight), 1024):
                    digest.update(table.rows(range(start, min(start + 1024, len(weight)))))
                byte_exact = digest.hexdigest() == weight_digest
                if not byte_exact:
                    raise ValueError("lossless native memory row bytes changed")
        result = {"mode": args.worker, "order": args.order, "seed": args.seed,
                  "chunk_rows": args.chunk_rows if args.worker == "mapped-bag" else None,
                  "seconds": times, "median_seconds": statistics.median(times),
                  "p95_seconds": float(np.percentile(times, 95)),
                  "input_peak_rss_bytes": baseline, "query_peak_rss_bytes": rss,
                  "input_shape": list(ids.shape), "output_shape": list(output.shape),
                  "full_gather_tensor_bytes": ids.numel() * weight.shape[1] * weight.element_size(),
                  "output_tensor_bytes": output.numel() * output.element_size(),
                  "source_weight_sha256": weight_digest, "query_ids_sha256": sha(ids.numpy().tobytes()),
                  "checkpoint_sha256": run["checkpoint_sha256"], "all_native_rows_byte_exact": byte_exact,
                  "sample_double_oracle_max_abs_error": float(error.max()),
                  "output_sha256": sha(output.numpy().tobytes()), "last_query_native_counters": counters}
        print(json.dumps(result), flush=True)
    finally:
        if hasattr(module, "close"):
            module.close()


def task_delta(args, order):
    report, run, directory, checkpoint, segment = load_reference(args, order)
    model = CausalNgramLM(len(checkpoint["registry"]["characters"]) + 1,
                          len(checkpoint["registry"]["grams"]), checkpoint["dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    reference = evaluate(model, segment, report["settings"]["burn_in"], report["settings"]["eval_chunk"])
    original = model.memory
    with MappedEmbeddingBag(directory / "lossless", mode="mean", padding_idx=0,
                            native_library=args.native_library, native_cache_bytes=args.cache_bytes,
                            chunk_rows=max(args.chunks)) as mapped:
        model.memory = mapped
        candidate = evaluate(model, segment, report["settings"]["burn_in"], report["settings"]["eval_chunk"])
        model.memory = original
    return {"order": order, "seed": args.seed, "checkpoint_sha256": run["checkpoint_sha256"],
            "trained_state_sha256": run["trained_state_sha256"],
            "reference_test": metric_summary(reference), "native_bag_test": metric_summary(candidate),
            "prediction_delta": prediction_delta(candidate, reference),
            "target_ids_identical": bool(torch.equal(reference["targets"], candidate["targets"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--orders", default="4,8,12")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--bags", type=int, default=32768)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--chunks", default="1024,16384")
    parser.add_argument("--cache-bytes", type=int, default=4194304)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--worker", choices=["resident-bag", "mapped-gather", "mapped-bag"], help=argparse.SUPPRESS)
    parser.add_argument("--order", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.chunks = [int(v) for v in args.chunks.split(",")]
    if min(args.bags, args.chunk_rows, args.repeat, args.timeout, *args.chunks) < 1 or min(args.cache_bytes, args.warmup) < 0:
        parser.error("positive workload/chunks/repeat, nonnegative cache/warmup required")
    if args.worker:
        worker(args)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    training = json.loads((args.training_dir / "report.json").read_text())
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"category": "trained causal character-memory CPU serving; cyclic held-out ID replay and original test task",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version, "pytorch": torch.__version__,
              "numpy": np.__version__, "os": platform.platform(), "cpu": cpu,
              "device": "cpu", "gpu": "not used", "threads": 1, "interop_threads": 1,
              "native_library_sha256": file_sha256(args.native_library),
              "training_report_sha256": file_sha256(args.training_dir / "report.json"),
              "training_commit": training["commit"], "corpus_source": training["corpus_source"],
              "corpus_sha256": training["corpus_sha256"], "split": training["split"],
              "cost_assumptions": "Existing local trained checkpoints/cached public corpus; no retraining or paid services.",
              "settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path) and k not in ("worker", "order")},
              "limitations": ["Three previously trained orders, one seed/corpus; not trained Engram or a large Transformer.",
                              "Performance repeats chronological test IDs cyclically to fill large batches; not full-model throughput or actual production traces.",
                              "Fresh subprocesses, fixed mode order and warm allocator/page cache; RSS includes checkpoints, framework, inputs, tables and cache.",
                              "Checkpoint weight retained as a common diagnostic baseline in every mode; mapped RSS is not a minimal deployment estimate.",
                              "Native chunks can decode more blocks than a whole-batch gather; report memory/latency tradeoffs and resident baseline.",
                              "Byte-exact storage does not imply bit-exact reductions. Native FP64 sums can change logits; task delta is measured separately.",
                              "Single-worker CPU; no CUDA, training, weighted or max bags. Corpus/weights remain local."],
              "cases": [], "task_quality": [], "failures": []}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    for order in [int(v) for v in args.orders.split(",")]:
        for mode, chunk in [("resident-bag", args.chunk_rows), ("mapped-gather", args.chunk_rows)] + [("mapped-bag", n) for n in args.chunks]:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--order", str(order),
                   "--training-dir", str(args.training_dir), "--corpus", str(args.corpus),
                   "--native-library", str(args.native_library), "--seed", str(args.seed),
                   "--bags", str(args.bags), "--chunk-rows", str(chunk), "--cache-bytes", str(args.cache_bytes),
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--json-out", str(args.json_out)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                if result.returncode:
                    raise RuntimeError(f"exit {result.returncode}: {result.stderr}")
                case = json.loads(result.stdout)
                case["command"] = shlex.join(cmd)
                report["cases"].append(case)
                if case["sample_double_oracle_max_abs_error"] > 1e-6:
                    report["failures"].append({"order": order, "mode": mode, "error": "sample oracle error exceeds 1e-6"})
                print(json.dumps({k: v for k, v in case.items() if k in ("order", "mode", "chunk_rows", "median_seconds", "query_peak_rss_bytes")}), flush=True)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                report["failures"].append({"order": order, "mode": mode, "error": str(error), "command": shlex.join(cmd)})
            args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        order_cases = [case for case in report["cases"] if case["order"] == order]
        if any(len({case[key] for case in order_cases}) > 1 for key in ("source_weight_sha256", "query_ids_sha256")):
            report["failures"].append({"order": order, "error": "mode inputs differ"})
        try:
            quality = task_delta(args, order)
            report["task_quality"].append(quality)
            if (not quality["target_ids_identical"]
                    or quality["prediction_delta"]["mean_prediction_kl_nats"] > 1e-8
                    or quality["prediction_delta"]["maximum_absolute_logit_change"] > 1e-5):
                report["failures"].append({"order": order, "stage": "task quality", "error": "targets or reduction deltas exceed reference tolerance"})
        except Exception as error:
            report["failures"].append({"order": order, "stage": "task quality", "error": repr(error)})
        args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit("failures preserved in JSON")


if __name__ == "__main__":
    main()
