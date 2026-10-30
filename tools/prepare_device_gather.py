"""CPU-only fixtures from original trained causal memory and held-out suffix IDs.

No retraining, approximate rows, row tiling or generated performance vectors.
This reference is a character GRU, not trained TinyEngram/DeepSeek evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import torch

from .causal_ngram_lm import encode_segment
from .engram_row_adapter import export_multihead, file_sha256, MappedMultiHeadEmbedding

ROOT = Path(__file__).resolve().parents[1]


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare(training_dir, corpus, output_dir, *, order=12, seed=3,
            positions=(1, 32, 128, 512), queries=8, native_library=None):
    training_dir, corpus, output_dir = map(Path, (training_dir, corpus, output_dir))
    if not positions or min(*positions, queries) < 1 or len(set(positions)) != len(positions):
        raise ValueError("distinct positive position counts and positive queries required")
    training = json.loads((training_dir / "report.json").read_text())
    run = next(r for r in training["runs"] if r["max_order"] == order and r["seed"] == seed)
    checkpoint = training_dir / f"order-{order}-seed-{seed}" / "reference.pt"
    if file_sha256(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError("trained checkpoint checksum changed")
    if file_sha256(corpus) != training["corpus_sha256"]:
        raise ValueError("corpus checksum changed")
    text = corpus.read_text(encoding="utf-8")[:training["settings"]["max_characters"]]
    heldout = text[int(len(text) * .9):]
    if hashlib.sha256(heldout.encode()).hexdigest() != training["split"]["test"]["sha256"]:
        raise ValueError("chronological held-out split changed")
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weight = source["model_state"]["memory.weight"].detach().contiguous()
    ids = encode_segment(heldout, source["registry"])["memory_ids"]
    if tensor_sha256(ids) != run["evaluation"]["test"]["memory_ids_sha256"]:
        raise ValueError("original held-out memory IDs changed")
    start = training["settings"]["burn_in"]
    if start + max(positions) * queries > len(ids):
        raise ValueError("not enough held-out positions; query fixtures are never tiled")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(weight, output_dir / "dense.pt")
    native = export_multihead(weight, [len(weight)], output_dir / "native",
                              block_rows=64, backend="native", predictor="xor")
    replay, cases = {}, []
    for count in positions:
        keys = []
        for index in range(queries):
            key = f"positions-{count}/query-{index}"
            query = ids[start + index * count:start + (index + 1) * count].reshape(-1, 1).contiguous()
            replay[key] = query
            keys.append({"key": key, "ids_sha256": tensor_sha256(query),
                         "output_sha256": tensor_sha256(weight[query]),
                         "shape": list(query.shape), "distinct_rows": len(torch.unique(query))})
        cases.append({"positions": count, "suffix_slots_per_position": ids.shape[1], "queries": keys})
    torch.save(replay, output_dir / "queries.pt")
    # Check all rows without retaining a second dense table. Benchmark workers
    # can subsequently compare output digests without opening dense.pt.
    digest = hashlib.sha256()
    with MappedMultiHeadEmbedding(output_dir / "native", native_library=native_library) as mapped:
        for offset in range(0, len(weight), 1024):
            digest.update(mapped.table.rows(range(offset, min(offset + 1024, len(weight)))))
    if digest.hexdigest() != native["weight_sha256"]:
        raise AssertionError("native rows changed source bytes")
    files = ["dense.pt", "queries.pt", "native/embedding.json", "native/table.men"]
    manifest = {"schema_version": 1, "category": "trained causal character-memory held-out gather replay",
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "preparer_sha256": file_sha256(__file__),
                "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
                "pytorch": torch.__version__, "os": platform.platform(), "device": "cpu",
                "training_report_sha256": file_sha256(training_dir / "report.json"),
                "training_commit": training["commit"], "checkpoint_sha256": run["checkpoint_sha256"],
                "corpus_source": training["corpus_source"], "corpus_sha256": training["corpus_sha256"],
                "test_split": training["split"]["test"], "full_test_ids_sha256": tensor_sha256(ids),
                "order": order, "seed": seed, "start_position": start,
                "shape": list(weight.shape), "dtype": str(weight.dtype),
                "weight_sha256": native["weight_sha256"], "raw_table_bytes": native["input_bytes"],
                "native_table_and_metadata_bytes": sum((output_dir / f).stat().st_size for f in files[2:]),
                "all_native_rows_byte_exact": True, "files": {f: file_sha256(output_dir / f) for f in files},
                "cases": cases,
                "limitations": ["Original lossless memory, not approximate cosine/norm variants.",
                                "Distinct chronological windows; no manufactured larger table or repeated rows.",
                                "Timed replay repeats these windows; gather microbenchmark, not model throughput.",
                                "One small character GRU/corpus, not TinyEngram or DeepSeek trained weights.",
                                "Corpus and weights remain local; no quality or compression claim from this preparation." ]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--positions", default="1,32,128,512")
    parser.add_argument("--queries", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = prepare(args.training_dir, args.corpus, args.output_dir, order=args.order, seed=args.seed,
                     positions=tuple(int(n) for n in args.positions.split(",")), queries=args.queries,
                     native_library=args.native_library)
    print(json.dumps({k: report[k] for k in ("order", "shape", "all_native_rows_byte_exact")}))


if __name__ == "__main__":
    main()
