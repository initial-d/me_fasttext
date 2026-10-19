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

    def test_xor_codec_adapts_and_preserves_reordered_original_ids(self):
        rng = random.Random(23)
        width = 256
        base = rng.randbytes(width)
        # Individually high-entropy rows share most bits; XOR residuals compress.
        related = [base[:i] + bytes([base[i] ^ 1]) + base[i + 1:] for i in range(8)]
        rows = related + [b"\0" * width] * 8 + [rng.randbytes(width) for _ in range(8)]
        for dedup in (False, True):
            with self.subTest(dedup=dedup), tempfile.TemporaryDirectory() as tmp:
                source, packed, plain = (Path(tmp) / name for name in ("raw", "xor.men", "plain.men"))
                source.write_bytes(b"".join(rows))
                # Reverse within groups to exercise original-ID forwarding.
                permutation = [i for start in range(0, len(rows), 8)
                               for i in reversed(range(start, start + 8))]
                result = pack_native(source, packed, width, 8, dedup, "xor", permutation)
                control = pack_native(source, plain, width, 8, dedup, row_order=permutation)
                self.assertLessEqual(result["artifact_bytes"], control["artifact_bytes"])
                self.assertGreater(result["xor_blocks"], 0)
                self.assertGreater(result["raw_blocks"], 0)
                if not dedup:
                    self.assertGreater(result["zlib_blocks"], 0)
                with NativeRows(packed, self.library) as table:
                    self.assertEqual(table.rows(range(len(rows))), b"".join(rows))
                    ids = [23, 0, 7, 8, 0, 16, 1, 23]
                    self.assertEqual(table.rows(ids), b"".join(rows[i] for i in ids))
                    self.assertEqual(table.last_block_reads, 3)
                data = packed.read_bytes()
                # Version 1 must reject a version 2 codec rather than misdecode it.
                changed = bytearray(data)
                struct.pack_into("<Q", changed, 8, 1)
                packed.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, "descriptor"):
                    NativeRows(packed, self.library)
                changed = bytearray(data)
                struct.pack_into("<Q", changed, 8, 3)
                packed.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, "header"):
                    NativeRows(packed, self.library)
                # Corrupt the XOR zlib stream, keeping descriptor offsets valid.
                changed = bytearray(data)
                offset = struct.unpack_from("<Q", data, 56 + len(rows) * 8)[0]
                changed[offset] ^= 255
                packed.write_bytes(changed)
                with NativeRows(packed, self.library) as table:
                    with self.assertRaisesRegex(ValueError, "decompression"):
                        table.row(0)

    def test_xor_empty_partial_blocks_and_invalid_permutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            for rows in ([], [b"abc"], [b"abc", b"abd", b"abe"]):
                source.write_bytes(b"".join(rows))
                pack_native(source, packed, 3, 2, False, "xor", list(reversed(range(len(rows)))))
                with NativeRows(packed, self.library) as table:
                    self.assertEqual(table.rows(range(len(rows))), b"".join(rows))
            for permutation in ([0, 1], [0, 0, 2], [0, 1, 3], [0, 1, -1], [0, 1, True]):
                with self.assertRaisesRegex(ValueError, "permutation"):
                    pack_native(source, packed, 3, row_order=permutation)
            with self.assertRaisesRegex(ValueError, "predictor"):
                pack_native(source, packed, 3, predictor="invalid")

    def test_xor_torch_reconstructs_nonfinite_bit_patterns(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import export_multihead, MappedMultiHeadEmbedding
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(19)
                # Include NaN payloads, signed zero and infinities as opaque bits.
                bits = torch.randint(0, 256, (1, 512), dtype=torch.uint8).repeat(8, 1)
                special = {torch.float32: "000000800000807f0100c07f",
                           torch.float16: "0080007c017e", torch.bfloat16: "0080807f817f"}[dtype]
                prefix = bytes.fromhex(special)
                bits[:, :len(prefix)] = torch.tensor(list(prefix), dtype=torch.uint8)
                for i in range(8):
                    bits[i, 64 + i] ^= 1
                weight = bits.view(dtype)
                meta = export_multihead(weight, [3, 5], tmp, 8, backend="native", predictor="xor")
                table = Path(tmp) / "table.men"
                descriptor = 56 + 8 * len(weight)
                self.assertEqual(struct.unpack_from("<Q", table.read_bytes(), descriptor + 24)[0], 2)
                with MappedMultiHeadEmbedding(tmp, native_library=self.library) as mapped:
                    ids = torch.tensor([[2, 4], [0, 0], [2, 4]])
                    expected = weight[ids + torch.tensor([0, 3])]
                    self.assertTrue(torch.equal(mapped(ids).view(torch.uint8), expected.view(torch.uint8)))
                    self.assertEqual(mapped.table.last_decompressions, 1)
                self.assertEqual(meta["predictor"], "xor")

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
                export_multihead(weight, [3, 5, 2], tmp, 2, backend="native", predictor="xor")
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

    def test_small_causal_transformer_logits_match_with_untied_embedding(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import MappedEmbedding, export_embedding

        class TinyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(31, 16)
                layer = torch.nn.TransformerEncoderLayer(16, 2, 32, dropout=0, batch_first=True)
                self.blocks = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                self.lm_head = torch.nn.Linear(16, 31, bias=False)  # deliberately untied

            def forward(self, ids):
                mask = torch.triu(torch.ones(ids.shape[-1], ids.shape[-1], dtype=torch.bool), diagonal=1)
                return self.lm_head(self.blocks(self.embedding(ids), mask=mask))

        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            torch.manual_seed(7)
            model = TinyDecoder().eval()
            ids = torch.tensor([[0, 1, 2, 3, 4], [4, 4, 2, 0, 30]])
            before = model(ids)
            export_embedding(model.embedding, tmp, 4)
            with MappedEmbedding(tmp, self.library) as replacement:
                model.embedding = replacement
                after = model(ids)
                self.assertTrue(torch.equal(before.view(torch.uint8), after.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
