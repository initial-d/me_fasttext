"""Protocol/bit fixtures only; these vectors are not trained compression evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None and shutil.which("cc"), "PyTorch and C compiler required")
class DeviceBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.library = cls.root / "librows.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-fPIC", "-shared", str(ROOT / "native/meft_rows.c"),
                        "-lz", "-o", str(cls.library)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fixture(self):
        from tools.engram_row_adapter import export_multihead, file_sha256
        from tools.prepare_device_gather import tensor_sha256
        directory = self.root / str(len(list(self.root.iterdir())))
        directory.mkdir()
        weight = torch.arange(91, dtype=torch.float32).reshape(13, 7)
        weight[0, 0] = -0.0
        weight.view(torch.int32)[4, 1] = 0x7FC00001
        weight[10] = weight[0]
        torch.save(weight, directory / "dense.pt")
        meta = export_multihead(weight, [13], directory / "native", backend="native", block_rows=3, predictor="xor")
        ids = torch.tensor([[0], [4], [10], [12], [4]])
        torch.save({"q": ids}, directory / "queries.pt")
        files = ["dense.pt", "queries.pt", "native/table.men", "native/embedding.json"]
        manifest = {"schema_version": 1, "all_native_rows_byte_exact": True,
                    "files": {f: file_sha256(directory / f) for f in files},
                    "weight_sha256": meta["weight_sha256"], "shape": list(weight.shape), "dtype": str(weight.dtype),
                    "raw_table_bytes": weight.numel() * weight.element_size(),
                    "native_table_and_metadata_bytes": meta["artifact_bytes"],
                    "cases": [{"positions": 5, "queries": [{"key": "q", "shape": list(ids.shape),
                        "ids_sha256": tensor_sha256(ids), "output_sha256": tensor_sha256(weight[ids])}]}]}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory, weight, ids

    def test_dense_reader_preserves_special_bytes_and_rejects_bad_addressing(self):
        from tools.benchmark_device_gather import DenseStagedBaseline
        from tools.prepare_device_gather import tensor_sha256
        directory, weight, ids = self.fixture()
        with DenseStagedBaseline(directory, device="cpu", native_library=self.library) as dense:
            result = dense(ids)
            self.assertEqual(tensor_sha256(result), tensor_sha256(weight[ids]))
            result.zero_()
            self.assertEqual(tensor_sha256(dense(ids)), tensor_sha256(weight[ids]))
            with self.assertRaises(IndexError):
                dense(torch.tensor([[13]]))

    def test_native_worker_does_not_read_or_retain_dense_table(self):
        from tools.benchmark_device_gather import worker
        directory, _, _ = self.fixture()
        # Missing source weights are a stronger guard than recording load calls.
        (directory / "dense.pt").unlink()
        args = argparse.Namespace(worker="native", device="cpu", fixture=directory,
            native_library=self.library, cache_bytes=0, query_chunk_rows=2, staging_rows=3,
            staging_slots=2, warmup=1, repeat=2, worker_report=directory / "result.json")
        with patch("torch.cuda.is_available", side_effect=AssertionError("CPU mode probed GPU")):
            self.assertEqual(worker(args), 0)
        report = json.loads(args.worker_report.read_text())
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["cases"][0]["outputs_byte_exact"])

    def test_corrupt_queries_and_dense_source_are_rejected(self):
        from tools.benchmark_device_gather import fixture_inputs, DenseStagedBaseline
        directory, _, _ = self.fixture()
        with (directory / "queries.pt").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            fixture_inputs(directory, "native")
        directory, weight, _ = self.fixture()
        weight[2, 0] += 1
        torch.save(weight, directory / "dense.pt")
        with self.assertRaisesRegex(ValueError, "source bytes"):
            DenseStagedBaseline(directory, device="cpu", native_library=self.library)

    def test_gpu_competition_snapshot_identifies_only_other_processes(self):
        import os
        from tools.benchmark_device_gather import gpu_snapshot
        with patch("subprocess.check_output", side_effect=[
            f"GPU-a, {os.getpid()}\nGPU-a, 7\nGPU-b, 9\n",
            "GPU-a, NVIDIA A100, 535.54.03, 99, 6277\n"]):
            self.assertEqual(gpu_snapshot("GPU-a")["external_compute_pids"], [7])

    def test_gpu_uuid_prefix_and_explicit_host_pid_mapping(self):
        from tools.benchmark_device_gather import gpu_snapshot, monitor_client_pid
        with patch("subprocess.check_output", side_effect=[
            "GPU-a, 17\nGPU-a, 19\n", "GPU-a, NVIDIA A100, 535.54.03, 1, 75000\n"]) as command:
            snapshot = gpu_snapshot("a", own_pid=17)
            self.assertEqual(snapshot["compute_pids"], [17, 19])
            self.assertEqual(snapshot["external_compute_pids"], [19])
            self.assertIn("--id=GPU-a", command.call_args.args[0])
            with self.assertRaisesRegex(ValueError, "cannot identify"):
                monitor_client_pid(snapshot)
        self.assertEqual(monitor_client_pid({"compute_pids": [17]}), 17)
        with self.assertRaises(ValueError):
            monitor_client_pid({"compute_pids": []})


if __name__ == "__main__":
    unittest.main()
