"""Synthetic byte-exact lookup check; no Engram checkpoint or speedup claim."""
import argparse
import hashlib
import json
import mmap
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from lossless_row_store import MappedRows, pack_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    width, count = 256, 2048
    result = {
        "category": "synthetic engineering; not an Engram model benchmark",
        "command": " ".join(sys.argv),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "seed": 7, "row_bytes": width, "n_rows": count,
        "codec": "zlib", "block_rows": 32, "cache_blocks": 4, "cases": {},
        "limitations": ["Random bytes, not trained embedding weights.",
                        "Offline builder keeps rows and compressed payloads in RAM.",
                        "Python timing includes dispatch and bytes copies; warm cache, no RSS or GPU test.",
                        "No Engram integration or end-to-end model equivalence measured."],
    }
    queries = [rng.randrange(count) for _ in range(4096)]
    for name, pool_size in (("exact_duplicates", 128), ("unique_random", count)):
        pool = [rng.getrandbits(width * 8).to_bytes(width, "little") for _ in range(pool_size)]
        rows = [pool[i % pool_size] for i in range(count)]
        source, packed = args.output_dir / f"{name}.raw", args.output_dir / f"{name}.mer"
        source.write_bytes(b"".join(rows))
        start = time.perf_counter()
        stats = pack_rows(source, packed, width, 32)
        build_seconds = time.perf_counter() - start
        with MappedRows(packed, cache_blocks=4) as table:
            for row_id, original in enumerate(rows):
                assert table.row(row_id) == original
            assert table.rows(queries) == b"".join(rows[i] for i in queries)
            latencies = []
            for row_id in queries:
                start = time.perf_counter_ns()
                table.row(row_id)
                latencies.append((time.perf_counter_ns() - start) / 1000)
        with source.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            raw_times = []
            for row_id in queries:
                start = time.perf_counter_ns()
                raw[row_id * width:(row_id + 1) * width]
                raw_times.append((time.perf_counter_ns() - start) / 1000)
        result["cases"][name] = {
            "unique_rows": stats["unique_rows"], "input_bytes": stats["input_bytes"],
            "artifact_bytes": stats["artifact_bytes"], "row_map_bytes": stats["row_map_bytes"],
            "artifact_to_raw_ratio": stats["artifact_bytes"] / stats["input_bytes"],
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
            "all_rows_byte_exact": True, "batch_order_byte_exact": True,
            "build_seconds": build_seconds,
            "warm_lookup_p50_us": statistics.median(latencies),
            "warm_lookup_p95_us": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "raw_mmap_p50_us": statistics.median(raw_times),
            "raw_mmap_p95_us": sorted(raw_times)[int(0.95 * (len(raw_times) - 1))],
        }
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
