"""Separate-process resident / dense-offload / native device gather comparison.

CPU mode validates the protocol. CUDA performance requires an idle selected GPU.
The offload baselines share the production pinned-slot/copy/event implementation.
No full source table is loaded in the native worker. No model throughput claims.
"""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from .device_row_adapter import StagedMultiHeadEmbedding
from .engram_row_adapter import file_sha256
from .prepare_device_gather import tensor_sha256

ROOT = Path(__file__).resolve().parents[1]
MODES = ("resident", "dense-offload", "native")


class DenseStagedBaseline(StagedMultiHeadEmbedding):
    """Benchmark-only dense CPU reader with the native adapter's copy scheduler."""
    def __init__(self, fixture, **kwargs):
        super().__init__(Path(fixture) / "native", **kwargs)
        # Retain original addressing metadata, release mmap/cache immediately.
        self.gather.close()
        self.weight = torch.load(Path(fixture) / "dense.pt", map_location="cpu", weights_only=True)
        if (list(self.weight.shape) != self.gather.metadata["shape"]
                or self.weight.dtype != self.dtype
                or tensor_sha256(self.weight) != self.gather.metadata["weight_sha256"]):
            raise ValueError("dense baseline differs from native source bytes")

    def _read(self, ids, output):
        torch.index_select(self.weight, 0, ids, out=output.reshape(-1, self.embedding_dim))


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = None
    if sys.platform == "linux":
        current = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {"current_bytes": current, "process_peak_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}


def gpu_snapshot(uuid, *, own_pid=None):
    """No process names/commands; report competition without disturbing jobs."""
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    own_pid = os.getpid() if own_pid is None else own_pid
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    clients = sorted({int(pid.strip()) for line in raw.splitlines() if "," in line
                      for gpu, pid in [line.split(",")] if gpu.strip() == uuid})
    others = [pid for pid in clients if pid != own_pid]
    raw = subprocess.check_output(["nvidia-smi", "--id=" + uuid,
                                   "--query-gpu=uuid,name,driver_version,utilization.gpu,memory.free",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    gpu, name, driver, utilization, free = [v.strip() for v in raw.strip().split(",")]
    if gpu != uuid:
        raise ValueError("GPU monitoring selected a different device")
    return {"gpu_uuid": gpu, "gpu": name, "driver": driver, "utilization_percent": int(utilization),
            "free_mib": int(free), "compute_pids": clients, "external_compute_pids": others}


def monitor_client_pid(snapshot):
    """Resolve driver PID after an empty-device gate and our context creation.

    NVML may report a host PID while os.getpid() is in a container namespace.
    Require one client; never ignore an arbitrary pre-existing driver process.
    """
    if len(snapshot["compute_pids"]) != 1:
        raise ValueError("cannot identify worker CUDA client after idle-device gate")
    return snapshot["compute_pids"][0]


def fixture_inputs(fixture, mode):
    manifest = json.loads((fixture / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or not manifest["all_native_rows_byte_exact"]:
        raise ValueError("unsupported or unverified fixture")
    # Native does not open dense.pt, even for integrity/quality diagnostics.
    for filename, digest in manifest["files"].items():
        if filename == "dense.pt" and mode == "native":
            continue
        if file_sha256(fixture / filename) != digest:
            raise ValueError("fixture checksum changed: " + filename)
    replay = torch.load(fixture / "queries.pt", map_location="cpu", weights_only=True)
    for case in manifest["cases"]:
        for query in case["queries"]:
            if tensor_sha256(replay[query["key"]]) != query["ids_sha256"]:
                raise ValueError("query IDs changed")
    return manifest, replay


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {"status": "failed", "mode": args.worker, "device": args.device,
              "command": shlex.join([sys.executable, *sys.argv]),
              "python": sys.version, "torch": torch.__version__, "os": platform.platform(),
              "cpu": platform.processor() or platform.machine(), "threads": 1, "interop_threads": 1,
              "cases": []}
    module, uuid, monitor_pid = None, None, None
    try:
        manifest, replay = fixture_inputs(args.fixture, args.worker)
        report.update(fixture_sha256=file_sha256(args.fixture / "manifest.json"),
                      weight_sha256=manifest["weight_sha256"], source_shape=manifest["shape"],
                      dtype=manifest["dtype"], raw_table_bytes=manifest["raw_table_bytes"],
                      native_table_and_metadata_bytes=manifest["native_table_and_metadata_bytes"],
                      native_library_sha256=file_sha256(args.native_library))
        if args.device == "cuda":
            selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not selector or "," in selector:
                report.update(status="blocked", error="expose exactly one selected GPU with CUDA_VISIBLE_DEVICES")
                return 2
            # Observe competition before creating any worker CUDA context, so
            # the driver/client PID can subsequently be resolved in containers.
            uuid = subprocess.check_output(["nvidia-smi", "--id=" + selector, "--query-gpu=uuid",
                                            "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
            report["gpu_before"] = gpu_snapshot(uuid, own_pid=0)
            if report["gpu_before"]["external_compute_pids"] or report["gpu_before"]["utilization_percent"] > 5:
                report.update(status="blocked", error="selected GPU is contended; performance run deferred")
                return 2
            if not torch.cuda.is_available():
                report.update(status="blocked", error="CUDA unavailable")
                return 2
            if torch.cuda.device_count() != 1:
                report.update(status="blocked", error="expose exactly one selected GPU with CUDA_VISIBLE_DEVICES")
                return 2
            properties = torch.cuda.get_device_properties(0)
            runtime_uuid = str(properties.uuid)
            runtime_uuid = runtime_uuid if runtime_uuid.startswith("GPU-") else "GPU-" + runtime_uuid
            if runtime_uuid != uuid:
                raise ValueError("CUDA runtime and GPU monitoring selected different devices")
            free, total = torch.cuda.mem_get_info()
            cap = int(total * args.allocator_fraction)
            report.update(cuda_runtime=torch.version.cuda, allocator_limit_bytes=cap,
                          reserve_bytes=args.reserve_bytes, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
            if free < cap + args.reserve_bytes:
                report.update(status="blocked", error="insufficient free memory for allocator cap plus reserve")
                return 2
            torch.cuda.set_per_process_memory_fraction(args.allocator_fraction)
            probe = torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            snapshot = gpu_snapshot(uuid, own_pid=0)
            try:
                monitor_pid = monitor_client_pid(snapshot)
            except ValueError as error:
                report.update(status="blocked", error=str(error), gpu_after_context=snapshot)
                return 2
            report.update(monitor_client_pid=monitor_pid, worker_namespace_pid=os.getpid(),
                          monitor_pid_method="sole driver client after empty-device gate and worker context creation")
            del probe
        sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
        report["rss_before_open"] = rss()
        started = time.perf_counter()
        kwargs = dict(device=args.device, native_library=args.native_library, native_cache_bytes=args.cache_bytes,
                      query_chunk_rows=args.query_chunk_rows, staging_rows=args.staging_rows,
                      staging_slots=args.staging_slots, native_cache_policy=args.cache_policy)
        if args.worker == "native":
            module = StagedMultiHeadEmbedding(args.fixture / "native", **kwargs)
            forward = module
        else:
            module = DenseStagedBaseline(args.fixture, **kwargs)
            if args.worker == "resident":
                module.weight = module.weight.to(args.device)
                gc.collect()
                def forward(ids):
                    global_ids = module.gather._global_ids(ids).reshape(-1).to(args.device)
                    return module.weight.index_select(0, global_ids).reshape(*ids.shape, module.embedding_dim)
            else:
                forward = module
        sync()
        report["open_seconds"] = time.perf_counter() - started
        report["rss_after_open"] = rss()
        with torch.inference_mode():
            for case in manifest["cases"]:
                items = case["queries"]
                # First pass includes CUDA output readback for correctness only.
                # It is not disk-cold: integrity checks have read artifact files.
                first = time.perf_counter()
                for item in items:
                    output = forward(replay[item["key"]])
                    sync()
                    if tensor_sha256(output) != item["output_sha256"]:
                        raise AssertionError("gather output changed source bytes: " + item["key"])
                    del output
                first_seconds = time.perf_counter() - first
                for _ in range(args.warmup):
                    for item in items:
                        output = forward(replay[item["key"]])
                        sync()
                        del output
                if uuid:
                    before_timing = gpu_snapshot(uuid, own_pid=monitor_pid)
                    if before_timing["external_compute_pids"] or monitor_pid not in before_timing["compute_pids"]:
                        report.update(status="blocked", error="competition appeared before timing", gpu_during=before_timing)
                        return 2
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.repeat):
                    for item in items:
                        sync()
                        started = time.perf_counter()
                        output = forward(replay[item["key"]])
                        sync()  # includes CPU gather/H2D/ID transfer; excludes D2H hashing
                        times.append(time.perf_counter() - started)
                        del output  # one live output, no retained layer/prefill outputs
                sync()
                memory = ({"allocated_bytes": torch.cuda.memory_allocated(),
                           "reserved_bytes": torch.cuda.memory_reserved(),
                           "warm_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "warm_peak_reserved_bytes": torch.cuda.max_memory_reserved()} if uuid else None)
                if uuid:
                    after_timing = gpu_snapshot(uuid, own_pid=monitor_pid)
                    if after_timing["external_compute_pids"] or monitor_pid not in after_timing["compute_pids"]:
                        report.update(status="blocked", error="competition appeared during timing", gpu_after=after_timing)
                        return 2
                report["cases"].append({"positions": case["positions"], "query_shape": items[0]["shape"],
                    "query_id_bytes": replay[items[0]["key"]].numel() * 8,
                    "output_bytes": replay[items[0]["key"]].numel() * module.embedding_dim * torch.empty((), dtype=module.dtype).element_size(),
                    "query_ids_sha256": [item["ids_sha256"] for item in items],
                    "outputs_byte_exact": True, "first_pass_including_correctness_readback_seconds": first_seconds,
                    "seconds": times, "median_seconds": statistics.median(times),
                    "p95_seconds": float(np.percentile(times, 95)),
                    "gather_rows_per_second": sum(replay[item["key"]].numel() for item in items) * args.repeat / sum(times),
                    "gpu_memory": memory, "rss": rss(),
                    "staging": module.stats if args.worker != "resident" else None})
        report["status"] = "passed"
        return 0
    except Exception as error:
        report.update(status="failed", error=repr(error))
        return 1
    finally:
        if module is not None:
            try:
                module.close()
            except Exception as error:
                report.update(status="failed", cleanup_error=repr(error))
        args.worker_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staging-rows", type=int, default=256)
    parser.add_argument("--staging-slots", type=int, default=2)
    parser.add_argument("--query-chunk-rows", type=int, default=65536)
    parser.add_argument("--cache-bytes", type=int, default=4194304)
    parser.add_argument("--cache-policy", choices=("blocks", "rows"), default="blocks")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--allocator-fraction", type=float, default=.04)
    parser.add_argument("--reserve-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (min(args.staging_rows, args.staging_slots, args.query_chunk_rows, args.repeat, args.timeout) < 1
            or min(args.cache_bytes, args.warmup, args.reserve_bytes) < 0 or not 0 < args.allocator_fraction <= 1):
        parser.error("positive sizes/repeat/timeout, nonnegative budgets/warmup and fraction in (0,1] required")
    if args.worker:
        return worker(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "category": "CPU protocol validation" if args.device == "cpu" else "CUDA gather microbenchmark",
              "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]),
              "source_hashes": {p: file_sha256(ROOT / "tools" / p) for p in
                                ("benchmark_device_gather.py", "prepare_device_gather.py", "device_row_adapter.py")},
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("worker", "worker_report")},
              "limitations": ["Small trained character-GRU tables; no TinyEngram/DeepSeek compatibility or model throughput conclusion.",
                              "CPU mode validates bytes/protocol only; no accelerator speed or staging evidence.",
                              "Fresh processes, fixed baseline order; warm OS page cache is not flushed.",
                              "First-pass diagnostics include readback; warm timings exclude correctness D2H/hash.",
                              "One live gather output; full-model prefetch outputs/KV/backbone memory are excluded.",
                              "Dense offload shares native copy scheduling; generic offload savings are not compression gains.",
                              "GPU allocator metrics exclude driver/context allocations; RSS peak includes startup/transient tensors.",
                              "Idle gate samples external processes before/after timing, cannot reserve the GPU or rule out short interference.",
                              "Driver PID is the sole client appearing after an empty-device gate and worker context creation; ambiguity blocks the run.",
                              "Existing local checkpoints and corpus; no paid services, retraining or downloads."], "workers": []}
    for mode in MODES:
        path = args.output_dir / (mode + ".json")
        cmd = [sys.executable, "-m", "tools.benchmark_device_gather", "--worker", mode, "--worker-report", str(path)]
        for key, value in vars(args).items():
            if key not in ("worker", "worker_report"):
                cmd += ["--" + key.replace("_", "-"), str(value)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            entry = json.loads(path.read_text()) if path.exists() else {"status": "failed", "mode": mode}
            entry.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as error:
            entry = {"status": "failed", "mode": mode, "error": "worker timeout",
                     "command": shlex.join(cmd), "timeout_seconds": error.timeout, "exit_code": 1}
        report["workers"].append(entry)
        if entry["exit_code"]:
            break
    statuses = [v["status"] for v in report["workers"]]
    report["status"] = "blocked" if "blocked" in statuses else ("passed" if len(statuses) == 3 and set(statuses) == {"passed"} else "failed")
    if report["status"] == "passed":
        if (len({v["fixture_sha256"] for v in report["workers"]}) != 1
                or any(v["cases"] and [c["query_ids_sha256"] for c in v["cases"]] !=
                       [c["query_ids_sha256"] for c in report["workers"][0]["cases"]] for v in report["workers"])):
            report.update(status="failed", error="baseline inputs differ")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Report:", (args.output_dir / "report.json").resolve(), report["status"])
    return {"passed": 0, "blocked": 2, "failed": 1}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
