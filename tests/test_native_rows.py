import ctypes
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from tools.native_row_store import NativeRows, pack_native

ROOT = Path(__file__).resolve().parents[1]


class NativeRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.tmp.name) / "librows.so"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-Wall", "-Wextra",
                        "-Werror", "-fPIC", "-shared", str(ROOT / "native/meft_rows.c"),
                        "-lz", "-o", str(cls.library)], check=True)
        cls.example = Path(cls.tmp.name) / "gather"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-I" + str(ROOT / "native"), str(ROOT / "examples/native_gather.c"),
                        str(ROOT / "native/meft_rows.c"), "-lz", "-o", str(cls.example)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_preserves_arbitrary_bytes_ids_and_batches(self):
        rng = random.Random(3)
        for width in (1, 3, 64, 256):
            for dedup in (False, True):
                with self.subTest(width=width, dedup=dedup), tempfile.TemporaryDirectory() as tmp:
                    # Zero blocks compress; random blocks select raw fallback.
                    rows = [b"\0" * width for _ in range(6)]
                    rows += [rng.getrandbits(width * 8).to_bytes(width, "little") for _ in range(31)]
                    rows += [rows[1], rows[12], rows[12]]
                    source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
                    source.write_bytes(b"".join(rows))
                    pack_native(source, packed, width, 4, dedup)
                    with NativeRows(packed, self.library) as table:
                        ids = [rng.randrange(len(rows)) for _ in range(200)]
                        self.assertEqual(table.rows(ids), b"".join(rows[i] for i in ids))
                        self.assertEqual(table.rows(range(len(rows))), b"".join(rows))
                        self.assertEqual(table.rows([]), b"")
                        with self.assertRaises(IndexError):
                            table.row(-1)
                        bad_id = (ctypes.c_uint64 * 1)(len(rows))
                        out = ctypes.create_string_buffer(width)
                        with self.assertRaisesRegex(ValueError, "row ID"):
                            table.gather_into(ctypes.addressof(bad_id), 1, ctypes.addressof(out), width)
                        valid_id = (ctypes.c_uint64 * 1)(0)
                        with self.assertRaisesRegex(ValueError, "capacity"):
                            table.gather_into(ctypes.addressof(valid_id), 1, ctypes.addressof(out), width - 1)
                    result = subprocess.run([str(self.example), str(packed), "0", "12", "0"],
                                            check=True, capture_output=True, text=True)
                    self.assertIn("id=12 bytes=" + rows[12].hex(), result.stdout)

    def test_native_coalesces_blocks_and_rejects_bad_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            rows = [bytes([i]) * 128 for i in range(16)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 4, False)
            with NativeRows(packed, self.library) as table:
                table.rows([0, 4, 0, 4, 1, 5])
                self.assertEqual(table.last_block_reads, 2)
                self.assertEqual(table.last_decompressions, 2)
            data = packed.read_bytes()
            for mutated in (data[:20], data[:-1], b"BADMAGIC" + data[8:], data + b"extra"):
                packed.write_bytes(mutated)
                with self.assertRaises(ValueError):
                    NativeRows(packed, self.library)
            # Corrupt the first queried forwarding ID: reject on gather.
            changed = bytearray(data)
            struct.pack_into("<Q", changed, 56, 99999)
            packed.write_bytes(changed)
            with NativeRows(packed, self.library) as table, self.assertRaisesRegex(ValueError, "forwarding"):
                table.row(0)
            # Corrupt a compressed payload while keeping structural offsets.
            changed = bytearray(data)
            offset = struct.unpack_from("<Q", data, 56 + len(rows) * 8)[0]
            changed[offset] ^= 255
            packed.write_bytes(changed)
            with NativeRows(packed, self.library) as table, self.assertRaisesRegex(ValueError, "decompression"):
                table.row(0)

    def test_native_torch_adapter_direct_output(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import export_multihead, MappedMultiHeadEmbedding
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(7)
                weight = torch.randn(10, 9, dtype=dtype)
                weight[8] = weight[0]
                weight[1, 0] = -0.0
                export_multihead(weight, [3, 5, 2], tmp, 2, backend="native")
                ids = torch.tensor([[[2, 4, 1], [0, 0, 0]], [[1, 2, 0], [2, 4, 1]]])
                with MappedMultiHeadEmbedding(tmp, native_library=self.library) as mapped:
                    for query in (ids, ids.transpose(0, 1), ids.to(torch.int32), ids[:0]):
                        original = torch.nn.functional.embedding(query.to(torch.int64) + torch.tensor([0, 3, 8]), weight)
                        actual = mapped(query)
                        self.assertEqual(actual.shape, original.shape)
                        self.assertTrue(torch.equal(actual.view(torch.uint8), original.contiguous().view(torch.uint8)))

    def test_ordinary_embedding_and_max_norm_boundary(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import MappedEmbedding, export_embedding
        with tempfile.TemporaryDirectory() as tmp:
            original = torch.nn.Embedding(17, 7, padding_idx=0).eval()
            export_embedding(original, tmp, block_rows=4)
            with MappedEmbedding(tmp, self.library) as mapped:
                for ids in (torch.tensor([[0, 16, 0], [5, 2, 7]]), torch.tensor(2),
                            torch.empty((0, 3), dtype=torch.int64)):
                    self.assertTrue(torch.equal(mapped(ids).view(torch.uint8), original(ids).view(torch.uint8)))
                with self.assertRaises(RuntimeError):
                    mapped.train()
            with self.assertRaises(ValueError):
                export_embedding(torch.nn.Embedding(17, 7, max_norm=1), Path(tmp) / "maxnorm")


if __name__ == "__main__":
    unittest.main()
