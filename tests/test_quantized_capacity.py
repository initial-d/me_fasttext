"""Small deterministic correctness fixtures, never compression/quality evidence."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None and shutil.which("cc"), "NumPy and C compiler required")
class CapacityTest(unittest.TestCase):
    def test_fresh_worker_oracles_full_verification_and_direct_control(self):
        from tools.benchmark_quantized_capacity import build, verify, query, prepare_oracle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / ("rows.dylib" if sys.platform == "darwin" else "rows.so")
            native = Path(__file__).resolve().parents[1] / "native/meft_rows.c"
            subprocess.run(["cc", "-std=c11", "-O2", "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared",
                            str(native), "-lz", "-o", str(library)], check=True, capture_output=True)
            rows = np.array([[127., .5, -.5], [1., .3, -.2], [0., -0., 0.], [2., .6, -.4]], dtype="<f4")
            source = root / "source.f32"
            source.write_bytes(rows.tobytes())
            ids = [3, 1, 0, 2] * 128
            manifest = dict(shape=list(rows.shape), cases=[dict(name="fixture", queries=[dict(
                ids=ids, sha256=hashlib.sha256(rows[ids].tobytes()).hexdigest())])])
            oracle = prepare_oracle(source, manifest, 3)
            oracle_file = root / "oracle.json"
            oracle_file.write_text(json.dumps(oracle))
            cfg = Namespace(source=source, dim=3, native_library=library, chunk_rows=2,
                            cache_bytes=128, max_block_bytes=12, repeat=2, case="fixture",
                            query_manifest=oracle_file, variant_label=None)
            for encoding in ("fp32", "fp16", "int8"):
                for backend in ("raw", "native"):
                    cfg.encoding, cfg.backend = encoding, backend
                    cfg.bundle = root / (encoding + backend)
                    cfg.worker_report = root / "worker.json"
                    build(cfg)
                    verify(cfg)
                    verified = json.loads(cfg.worker_report.read_text())
                    self.assertEqual(verified["rows_verified"], len(rows))
                    self.assertTrue(verified["all_original_ids_encoded_byte_exact"])
                    if encoding == "fp32":
                        self.assertEqual(verified["relative_frobenius"], 0)
                    query(cfg)
                    checked = json.loads(cfg.worker_report.read_text())
                    self.assertTrue(checked["all_query_bytes_exact"])
                    self.assertEqual(checked["query_rows"], 512)
                    self.assertEqual(len(checked["first_trace"]["wall"]["samples_ms"]), 1)
            cfg.backend, cfg.encoding = "direct", "fp32"
            query(cfg)
            self.assertTrue(json.loads(cfg.worker_report.read_text())["all_query_bytes_exact"])
            manifest["cases"][0]["queries"][0]["sha256"] = "bad"
            with self.assertRaisesRegex(AssertionError, "old FP32"):
                prepare_oracle(source, manifest, 3)


if __name__ == "__main__":
    unittest.main()
