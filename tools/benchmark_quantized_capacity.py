"""CPU scale/latency evidence on unchanged genuinely trained FP32 rows.

No training, downloads or GPU calls. Construction, verification and each query
trace use separate processes. Uniform IDs are a synthetic access pattern on
real vectors. Prefix subsets are storage scaling controls, not new trained
models. Native compression is exact to encoded bytes; FP16/INT8 are lossy.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from .benchmark_memory_capacity import environment, memory, samples
from .prepare_public_benchmark import file_sha256
from .quantized_memory import QuantizedRows, encode_rows, decode_rows, pack_quantized_file


def software_environment():
    result = environment()
    try:
        result["pytorch_installed_not_used"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        result["pytorch_installed_not_used"] = None
    result["thread_environment"] = {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_chunks(source, dim, chunk_rows):
    with source.open("rb") as stream:
        while True:
            part = stream.read(chunk_rows * dim * 4)
            if not part:
                break
            if len(part) % (dim * 4):
                raise ValueError("partial FP32 source row")
            yield np.frombuffer(part, dtype="<f4").reshape(-1, dim)


def build(args):
    report = dict(status="running", **software_environment(), memory_before=memory())
    started, cpu = time.perf_counter(), time.process_time()
    metadata = pack_quantized_file(args.source, args.bundle, args.dim, args.encoding,
                                   native=args.backend == "native", deduplicate=False,
                                   chunk_rows=args.chunk_rows, max_block_bytes=args.max_block_bytes)
    report.update(status="passed", wall_seconds=time.perf_counter() - started,
                  cpu_seconds=time.process_time() - cpu, memory_after=memory(), metadata=metadata,
                  table_bytes=(args.bundle / metadata["table_file"]).stat().st_size,
                  bundle_bytes=sum(p.stat().st_size for p in args.bundle.iterdir()))
    write_json(args.worker_report, report)


def verify(args):
    started = time.perf_counter()
    encoded_digest, decoded_digest, source_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    square_error = square_source = 0.0
    relative_errors, cosine_errors = [], []
    row = 0
    with QuantizedRows(args.bundle, args.native_library, cache_bytes=0) as table:
        for chunk in read_chunks(args.source, args.dim, args.chunk_rows):
            ids = np.arange(row, row + len(chunk), dtype="<u8")
            expected_encoded = encode_rows(chunk, table.encoding)
            expected_decoded = decode_rows(expected_encoded, args.dim, table.encoding)
            actual = table.encoded_rows(ids)
            if actual.tobytes() != expected_encoded.tobytes():
                raise AssertionError("original-ID encoded bytes differ from the codec oracle")
            out = np.empty_like(chunk)
            table.gather_into(ids, out)
            if out.tobytes() != expected_decoded.tobytes():
                raise AssertionError("preallocated FP32 reconstruction differs from the codec oracle")
            source_digest.update(chunk.tobytes())
            encoded_digest.update(actual.tobytes())
            decoded_digest.update(out.tobytes())
            x, y = chunk.astype(np.float64), out.astype(np.float64)
            difference = np.sum((y - x)**2, axis=1)
            norm = np.sum(x*x, axis=1)
            other = np.sum(y*y, axis=1)
            square_error += float(np.sum(difference))
            square_source += float(np.sum(norm))
            relative_errors.append(np.sqrt(difference / np.maximum(norm, np.finfo(np.float64).tiny)))
            cosine = np.divide(np.sum(x*y, axis=1), np.sqrt(norm*other),
                               out=np.zeros(len(x)), where=(norm*other) > 0)
            cosine[(norm == 0) & (other == 0)] = 1
            cosine_errors.append(1 - np.clip(cosine, -1, 1))
            row += len(chunk)
        if row != table.nrows or source_digest.hexdigest() != table.metadata["source_float32_sha256"]:
            raise AssertionError("source provenance differs from the full bundle")
        if decoded_digest.hexdigest() != table.metadata["decoded_float32_sha256"]:
            raise AssertionError("full original-ID decoded hash differs from export audit")
    relative = np.concatenate(relative_errors)
    cosine = np.concatenate(cosine_errors)
    write_json(args.worker_report, dict(status="passed", **software_environment(), rows_verified=row,
        all_original_ids_encoded_byte_exact=True, all_decoded_rows_oracle_byte_exact=True,
        source_sha256=source_digest.hexdigest(), encoded_original_id_sha256=encoded_digest.hexdigest(),
        decoded_original_id_sha256=decoded_digest.hexdigest(), verify_seconds=time.perf_counter() - started,
        relative_frobenius=float(np.sqrt(square_error / max(square_source, np.finfo(np.float64).tiny))),
        row_relative_l2=dict(p50=float(np.median(relative)), p99=float(np.percentile(relative, 99)), max=float(np.max(relative))),
        cosine_distance=dict(p50=float(np.median(cosine)), p99=float(np.percentile(cosine, 99)), max=float(np.max(cosine))),
        memory_after=memory()))


def query(args):
    manifest = json.loads(args.query_manifest.read_text())
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    report = dict(status="running", **software_environment(), variant=args.variant_label or args.backend + "-" + args.encoding,
                  case=args.case, cache_budget_bytes=args.cache_bytes if args.backend == "native" else 0,
                  memory_before_open=memory(), batch_rows=len(case["queries"][0]["ids"]))
    started = time.perf_counter()
    if args.backend == "direct":
        table = np.memmap(args.source, mode="r", dtype="<f4", shape=tuple(manifest["shape"]))
        width = args.dim * 4
    else:
        table = QuantizedRows(args.bundle, args.native_library, cache_bytes=args.cache_bytes)
        width = table.width
    report.update(open_ms=(time.perf_counter() - started)*1000, memory_after_open=memory())
    queries = [(np.asarray(q["ids"], dtype="<u8"), q["sha256_by_encoding"][args.encoding]) for q in case["queries"]]
    output = np.empty((report["batch_rows"], args.dim), dtype="<f4")
    scratch = np.empty((report["batch_rows"], width), dtype="u1")
    report["query_buffer_bytes"] = output.nbytes + scratch.nbytes
    report["active_query_buffer_bytes"] = output.nbytes + (0 if args.backend == "direct" else scratch.nbytes)

    def gather(item):
        ids, expected = item
        started, cpu = time.perf_counter_ns(), time.process_time_ns()
        if args.backend == "direct":
            np.take(table, ids.view(np.int64), axis=0, out=output, mode="clip")
        else:
            table.gather_into(ids, output, encoded_scratch=scratch)
        wall_ms, cpu_ms = (time.perf_counter_ns() - started)/1e6, (time.process_time_ns() - cpu)/1e6
        work = None
        if args.backend == "native":
            native = table.table
            work = {name: getattr(native, name) for name in ("last_block_reads", "last_decompressions",
                    "last_row_cache_hits", "last_compressed_bytes", "last_decoded_bytes")}
        if hashlib.sha256(output.tobytes()).hexdigest() != expected:
            raise AssertionError("query output differs from the independent original-ID codec oracle")
        return wall_ms, cpu_ms, work

    def measure(items):
        values = [gather(q) for q in items]
        return dict(wall=samples([v[0] for v in values]), cpu=samples([v[1] for v in values]),
                    codec_work=[v[2] for v in values],
                    rows_per_second=sum(len(q[0]) for q in items) / (sum(v[0] for v in values)/1000))

    try:
        report["memory_before_trace"] = memory()
        report["first_trace"] = measure(queries)
        report["memory_after_first_trace"] = memory()
        # Repeat the entire trace to reveal cache churn beyond the hot subset.
        report["second_trace"] = measure(queries)
        report["memory_after_second_trace"] = memory()
        hot = queries[:min(8, len(queries))]
        for item in hot:
            gather(item)
        report["hot_repeated"] = measure(hot * args.repeat)
        report.update(status="passed", memory_after_hot=memory(), query_rows=sum(len(q[0]) for q in queries),
                      distinct_rows=len(set(int(i) for q in queries for i in q[0])), all_query_bytes_exact=True)
    finally:
        if args.backend == "direct":
            table._mmap.close()
        else:
            table.close()
    write_json(args.worker_report, report)


def prepare_oracle(source, manifest, dim):
    result = dict(shape=manifest["shape"], cases=[])
    with source.open("rb") as stream:
        for case in manifest["cases"]:
            queries = []
            for query in case["queries"]:
                ids = query["ids"]
                if len(ids) != 512 or any(type(i) is not int or not 0 <= i < manifest["shape"][0] for i in ids):
                    raise ValueError("expected valid fixed 512-row original-ID trace")
                rows = []
                for i in ids:
                    stream.seek(i * dim * 4)
                    rows.append(stream.read(dim * 4))
                chunk = np.frombuffer(b"".join(rows), dtype="<f4").reshape(len(ids), dim)
                hashes = {e: hashlib.sha256(decode_rows(encode_rows(chunk, e), dim, e).tobytes()).hexdigest()
                          for e in ("fp32", "fp16", "int8")}
                if hashes["fp32"] != query["sha256"]:
                    raise AssertionError("old FP32 query oracle disagrees with unchanged source")
                queries.append(dict(ids=ids, sha256_by_encoding=hashes))
            result["cases"].append(dict(name=case["name"], queries=queries))
    return result


def run(args):
    source, training_path = args.training_dir / "trained.f32", args.training_dir / "training.json"
    training = json.loads(training_path.read_text())
    manifest = json.loads(args.query_manifest.read_text())
    if (training["status"] != "passed" or not training["all_rows_changed"]
            or file_sha256(source) != training["trained_sha256"]
            or manifest["shape"] != [training["ngram_rows"], training["dim"]]):
        raise ValueError("require unchanged, genuinely trained finite rows and matched query oracle")
    rows, dim = manifest["shape"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "report.json"
    report = dict(status="running", **software_environment(), training=training,
                  source_sha256=file_sha256(source), shape=manifest["shape"],
                  source_bytes=source.stat().st_size, training_report_sha256=file_sha256(training_path),
                  input_query_manifest_sha256=file_sha256(args.query_manifest),
                  native_library_sha256=file_sha256(args.native_library),
                  keys_bytes=(args.training_dir / "keys.json").stat().st_size,
                  keys_sha256=file_sha256(args.training_dir / "keys.json"),
                  settings=dict(chunk_rows=args.chunk_rows, max_block_bytes=args.max_block_bytes,
                                cache_bytes=args.cache_bytes, repeat=args.repeat, process_repeats=args.process_repeats,
                                seed=args.seed, scale_rows=args.scale_rows), builds={}, verification={}, workers=[],
                  limitations=["Shared CPU and filesystem; timing rounds randomize variant/trace order, no isolation guarantee.",
                    "Filesystem warmed by build/verification; first trace means fresh process/empty application cache, not disk-cold.",
                    "Uniform IDs are synthetic accesses on genuine vectors; original held-out lexical trace is reused.",
                    "FP16/INT8 are lossy relative to FP32; byte-exact claims apply to their encoded rows only.",
                    "No downstream task quality, GPU, cgroup memory cap, physical disk I/O or monetary cost measured.",
                    "Smaller prefixes reuse original trained rows without retraining and are storage scaling controls.",
                    "Chunking bounds codec working buffers, not native O(block count) descriptors or arbitrary optional planner indexes.",
                    "Complete bundle is table + codec metadata + common key registry; training source-ID provenance is separate."])
    write_json(report_path, report)

    def dispatch(phase, label, source_file, bundle, encoding, backend, case=None, block_bytes=None, variant=None):
        target = args.output_dir / (phase + "-" + label + ".json")
        command = [sys.executable, "-m", "tools.benchmark_quantized_capacity", "--phase", phase,
                   "--source", str(source_file), "--bundle", str(bundle), "--dim", str(dim),
                   "--encoding", encoding, "--backend", backend, "--worker-report", str(target),
                   "--native-library", str(args.native_library), "--chunk-rows", str(args.chunk_rows),
                   "--max-block-bytes", str(block_bytes or args.max_block_bytes), "--cache-bytes", str(args.cache_bytes),
                   "--repeat", str(args.repeat), "--query-manifest", str(args.output_dir / "queries.json")]
        if case:
            command += ["--case", case]
        if variant:
            command += ["--variant-label", variant]
        process = subprocess.run(command, capture_output=True, text=True)
        (args.output_dir / (phase + "-" + label + ".log")).write_text(process.stdout + process.stderr)
        value = dict(command=command, returncode=process.returncode, report_file=target.name)
        if target.exists():
            value["result"] = json.loads(target.read_text())
        else:
            value["error"] = process.stderr[-4000:]
        return value

    try:
        write_json(args.output_dir / "queries.json", prepare_oracle(source, manifest, dim))
        report["query_oracle_sha256"] = file_sha256(args.output_dir / "queries.json")
        variants = [(e + "-" + b, e, b, args.max_block_bytes)
                    for e in ("fp32", "fp16", "int8") for b in ("raw", "native")]
        variants += [("int8-native-rows-" + str(n), "int8", "native", n * (dim + 4))
                     for n in args.int8_extra_block_rows]
        report["settings"]["int8_extra_block_rows"] = args.int8_extra_block_rows
        for label, encoding, backend, block_bytes in variants:
            bundle = args.output_dir / label
            value = dispatch("build", label, source, bundle, encoding, backend, block_bytes=block_bytes)
            report["builds"][label] = value
            write_json(report_path, report)
            if value["returncode"]:
                raise RuntimeError("build failed: " + label)
            value["result"]["bundle_with_keys_bytes"] = value["result"]["bundle_bytes"] + report["keys_bytes"]
            value = dispatch("verify", label, source, bundle, encoding, backend)
            report["verification"][label] = value
            write_json(report_path, report)
            if value["returncode"]:
                raise RuntimeError("verification failed: " + label)
            print(json.dumps(dict(phase="verified", variant=label)), flush=True)
        for count in args.scale_rows:
            if not 0 < count < rows:
                raise ValueError("scale prefix must be smaller than the full trained source")
            prefix = args.output_dir / ("prefix-" + str(count) + ".f32")
            with source.open("rb") as src, prefix.open("wb") as dest:
                remaining = count * dim * 4
                while remaining:
                    part = src.read(min(1048576, remaining))
                    if not part:
                        raise ValueError("source truncated")
                    dest.write(part)
                    remaining -= len(part)
            label = "int8-native-prefix-" + str(count)
            value = dispatch("build", label, prefix, args.output_dir / label, "int8", "native")
            report["builds"][label] = value
            write_json(report_path, report)
            if value["returncode"]:
                raise RuntimeError("scale build failed: " + label)
            prefix.unlink()
        cases = [c["name"] for c in manifest["cases"]]
        for cycle in range(args.process_repeats):
            jobs = [(label, e, b, c) for label, e, b, _ in variants + [("fp32-direct", "fp32", "direct", 0)] for c in cases]
            random.Random(args.seed + cycle).shuffle(jobs)
            for variant, encoding, backend, case in jobs:
                label = f"{variant}-{case}-{cycle}"
                value = dispatch("query", label, source, args.output_dir / variant, encoding, backend, case, variant=variant)
                value.update(variant=variant, encoding=encoding, backend=backend, case=case, cycle=cycle)
                report["workers"].append(value)
                write_json(report_path, report)
                if value["returncode"]:
                    raise RuntimeError("query failed: " + label)
                print(json.dumps(dict(phase="query-passed", variant=label)), flush=True)
        report["status"] = "passed"
        write_json(report_path, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        write_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--max-block-bytes", type=int, default=8192)
    parser.add_argument("--cache-bytes", type=int, default=4194304)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--process-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-rows", type=int, nargs="*", default=[100000, 500000])
    parser.add_argument("--int8-extra-block-rows", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--phase", choices=("build", "verify", "query"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--encoding", choices=("fp32", "fp16", "int8"))
    parser.add_argument("--backend", choices=("raw", "native", "direct"))
    parser.add_argument("--worker-report", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--variant-label")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("set CUDA_VISIBLE_DEVICES to empty")
    if min(args.chunk_rows, args.max_block_bytes, args.repeat, args.process_repeats) < 1 or args.cache_bytes < 0:
        parser.error("invalid chunk/block/repeat/cache setting")
    if any(n < 1 or n > 256 for n in args.int8_extra_block_rows) or len(set(args.int8_extra_block_rows)) != len(args.int8_extra_block_rows):
        parser.error("distinct extra block rows between 1 and 256 required")
    if args.phase:
        if any(getattr(args, name) is None for name in ("source", "bundle", "dim", "encoding", "backend", "worker_report")):
            parser.error("worker source/bundle/dim/encoding/backend/report are required")
        if args.backend == "direct" and (args.phase != "query" or args.encoding != "fp32"):
            parser.error("direct backend is the FP32 query control only")
        {"build": build, "verify": verify, "query": query}[args.phase](args)
    else:
        if args.training_dir is None or args.output_dir is None:
            parser.error("training-dir and output-dir required")
        run(args)


if __name__ == "__main__":
    main()
