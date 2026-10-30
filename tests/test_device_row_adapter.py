"""Storage/stream lifetime fixtures, not trained-table compression evidence."""
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
class DeviceGatherTest(unittest.TestCase):
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

    def fixture(self, dtype=torch.float32 if torch is not None else None):
        from tools.engram_row_adapter import export_multihead
        directory = self.root / str(len(list(self.root.iterdir())))
        weight = torch.arange(91, dtype=torch.float32).reshape(13, 7).to(dtype)
        weight[0, 0] = -0.0
        weight[10] = weight[0]
        bits = weight.view(torch.int32 if dtype == torch.float32 else torch.int16)
        bits[4, 1] = 0x7FC00001 if dtype == torch.float32 else 0x7FFF
        export_multihead(weight, [5, 8], directory, backend="native", block_rows=3, predictor="xor")
        return directory, weight

    def adapter(self, directory, device, **kwargs):
        from tools.device_row_adapter import StagedMultiHeadEmbedding
        return StagedMultiHeadEmbedding(directory, device=device, native_library=self.library,
                                        query_chunk_rows=2, staging_rows=3, staging_slots=2, **kwargs)

    def assert_bytes(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertTrue(torch.equal(a.cpu().contiguous().view(torch.uint8),
                                    b.cpu().contiguous().view(torch.uint8)))

    def check_queries(self, device):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(device=device, dtype=dtype):
                directory, weight = self.fixture(dtype)
                with self.adapter(directory, device) as mapped:
                    ids = torch.tensor([[[0, 0], [4, 7]], [[1, 3], [2, 5]]])
                    handles, expected = [], []
                    # Delay consumption until slots have been reused many times.
                    for query in (ids, ids.transpose(0, 1), ids.to(torch.int32), ids[0, 0], ids[:0]) * 3:
                        handles.append(mapped.prefetch(query))
                        expected.append(torch.nn.functional.embedding(query.long() + torch.tensor([0, 5]), weight))
                    for ticket, reference in zip(handles, expected):
                        self.assert_bytes(ticket.wait(), reference)
                        self.assert_bytes(ticket.wait(), reference)
                    output = mapped(ids)
                    output.zero_()
                    self.assert_bytes(mapped(ids), expected[0])
                    self.assertEqual(mapped.gather.offsets.device.type, "cpu")
                    self.assertFalse(any(p.requires_grad for p in mapped.parameters()))
                    self.assertLessEqual(mapped.stats["staging_bytes"], mapped.stats["staging_byte_limit"])
                    if device != "cpu":
                        self.assertEqual(mapped.stats["slot_allocations"], 2)
                        self.assertGreater(mapped.stats["slot_reuses"], 0)
                self.assert_bytes(handles[0].wait(), expected[0])
                self.assertEqual(mapped.stats["staging_bytes"], 0)
                mapped.close()
                with self.assertRaisesRegex(ValueError, "closed"):
                    mapped.prefetch(ids)

    def test_cpu_owned_results_and_original_bytes(self):
        self.check_queries("cpu")

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_synchronous_staging_and_reuse(self):
        self.check_queries("mps")

    def test_reject_invalid_ids_placement_dtype_training_and_concurrency(self):
        directory, _ = self.fixture()
        with self.adapter(directory, "cpu") as mapped:
            for ids in (torch.tensor([[5, 0]]), torch.tensor([[-1, 0]]), torch.tensor([[0, 8]])):
                with self.assertRaises(IndexError):
                    mapped.prefetch(ids)
            with self.assertRaises(TypeError):
                mapped(torch.zeros(1, 2))
            with self.assertRaises(ValueError):
                mapped(torch.zeros(1, 3, dtype=torch.int64))
            self.assertIs(mapped.to("cpu"), mapped)
            with self.assertRaisesRegex(ValueError, "fixed"):
                mapped.half()
            with self.assertRaises(RuntimeError):
                mapped.train()
            with self.assertRaises(TypeError):
                import pickle
                pickle.dumps(mapped)
            self.assertIs(mapped.eval(), mapped)
            mapped._lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "single-worker"):
                    mapped(torch.zeros(1, 2, dtype=torch.int64))
                with self.assertRaisesRegex(RuntimeError, "single-worker"):
                    mapped.close()
            finally:
                mapped._lock.release()
            with self.assertRaises(ValueError):
                mapped.prefetch(torch.zeros(1, 2, dtype=torch.int64)).wait(object())
            ids = torch.tensor([[0, 0], [4, 7]], device="cpu")
            # The C output pointer must remain host memory even if a surrounding
            # model uses a non-CPU default allocation context.
            with torch.device("meta"):
                self.assertEqual(mapped(ids).device.type, "cpu")

    @unittest.skipUnless(torch is not None and not torch.cuda.is_available(), "requires host without CUDA")
    def test_cuda_request_fails_explicitly_without_hardware(self):
        directory, _ = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "not a GPU validation"):
            self.adapter(directory, "cuda")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "real CUDA required")
    def test_cuda_owned_outputs_pinned_staging_and_stream_waits(self):
        self.check_queries("cuda")
        directory, weight = self.fixture()
        mapped = self.adapter(directory, "cuda")
        consumer = torch.cuda.Stream()
        ids = torch.tensor([[0, 0], [4, 7]] * 40)
        reference = torch.nn.functional.embedding(ids + torch.tensor([0, 5]), weight)
        with torch.cuda.stream(mapped._copy_stream):
            if torch.version.hip is None:
                torch.cuda._sleep(5000000)
        ticket = mapped.prefetch(ids)
        self.assertTrue(all(s.tensor.is_pinned() for s in mapped._slots))
        with torch.cuda.stream(consumer):
            output = ticket.wait(consumer)
            if torch.version.hip is None:
                torch.cuda._sleep(5000000)
            result = output.clone()
            del output, ticket
        with torch.cuda.stream(mapped._copy_stream):
            for _ in range(10):
                pressure = torch.empty(reference.shape, dtype=reference.dtype, device="cuda")
                pressure.fill_(-1)
        # Close only waits producer copies. Consumer ownership is recorded on
        # the result tensor and remains valid after the adapter is gone.
        mapped.close()
        consumer.synchronize()
        self.assert_bytes(result, reference)
        with self.adapter(directory, "cuda") as reopened:
            pending = reopened.prefetch(ids)
        self.assert_bytes(pending.wait(), reference)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "real CUDA required")
    def test_cuda_partial_failure_drains_copies_before_reusing_slots(self):
        directory, weight = self.fixture()
        with self.adapter(directory, "cuda") as mapped:
            ids = torch.tensor([[0, 0], [4, 7]] * 40)
            original = mapped.gather.table.gather_into
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError("injected partial read failure")
                return original(*args, **kwargs)

            with patch.object(mapped.gather.table, "gather_into", fail_second):
                with self.assertRaisesRegex(ValueError, "partial read"):
                    mapped.prefetch(ids)
            reference = torch.nn.functional.embedding(ids + torch.tensor([0, 5]), weight)
            self.assert_bytes(mapped(ids), reference)


if __name__ == "__main__":
    unittest.main()
