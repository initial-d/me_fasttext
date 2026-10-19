import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "optional PyTorch adapter tests; install torch")
class EngramGatherTest(unittest.TestCase):
    def test_gather_preserves_head_offsets_shape_dtype_and_all_bytes(self):
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead

        head_sizes = [3, 5, 2]
        offsets = torch.tensor([0, 3, 8])
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(7)
                weight = torch.randn(10, 4, dtype=dtype)
                # Duplicates across head boundaries must keep original IDs.
                weight[8] = weight[0]
                weight[2, 0] = -0.0
                # Different NaN payloads must preserve their raw stored bits.
                bits = weight.view(torch.int32 if dtype == torch.float32 else torch.int16)
                bits[4, 1] = 0x7FC00001 if dtype == torch.float32 else 0x7FFF
                export_multihead(weight, head_sizes, tmp, block_rows=2)
                with MappedMultiHeadEmbedding(tmp, cache_blocks=1) as mapped:
                    for i in range(10):
                        self.assertEqual(mapped.table.row(i), weight[i].view(torch.uint8).numpy().tobytes())
                    ids = torch.tensor([[[0, 0, 0], [2, 4, 1]], [[1, 1, 1], [0, 0, 0]]])
                    # Include noncontiguous and int32 inputs, boundary IDs,
                    # repeated batch positions, one-position and empty batches.
                    for query in (ids, ids.transpose(0, 1), ids.to(torch.int32), ids[0, 0], ids[:0]):
                        expected = torch.nn.functional.embedding(query.to(torch.int64) + offsets, weight)
                        got = mapped(query)
                        self.assertEqual(got.shape, expected.shape)
                        self.assertEqual(got.dtype, expected.dtype)
                        self.assertTrue(torch.equal(got.view(torch.uint8), expected.contiguous().view(torch.uint8)))
                    output = mapped(ids)
                    output.zero_()
                    self.assertTrue(torch.equal(mapped(ids).view(torch.uint8),
                                                torch.nn.functional.embedding(ids + offsets, weight).view(torch.uint8)))

    def test_rejects_cross_head_ids_metadata_mismatch_and_training(self):
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead

        with tempfile.TemporaryDirectory() as tmp:
            export_multihead(torch.zeros(7, 3), [3, 4], tmp, 2)
            with MappedMultiHeadEmbedding(tmp) as mapped:
                for ids in (torch.tensor([[3, 0]]), torch.tensor([[0, -1]])):
                    with self.assertRaises(IndexError):
                        mapped(ids)
                with self.assertRaises(TypeError):
                    mapped(torch.tensor([[0.0, 0.0]]))
                with self.assertRaises(ValueError):
                    mapped(torch.tensor([[0]]))
                with self.assertRaises(RuntimeError):
                    mapped.train()
                self.assertIs(mapped.eval(), mapped)
            path = Path(tmp) / "embedding.json"
            meta = json.loads(path.read_text())
            meta["offsets"] = [0, 2]
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                MappedMultiHeadEmbedding(tmp)

    def test_export_checks_official_interface_offsets_and_preserves_source(self):
        from tools.engram_row_adapter import export_demo_embedding

        class DemoInterface(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(7, 3)
                self.register_buffer("offsets", torch.tensor([0, 3]))

        with tempfile.TemporaryDirectory() as tmp:
            source = DemoInterface()
            original = source.embedding.weight.detach().clone()
            export_demo_embedding(source, [3, 4], tmp)
            self.assertTrue(torch.equal(source.embedding.weight, original))
            with self.assertRaises(FileExistsError):
                export_demo_embedding(source, [3, 4], tmp)
            with self.assertRaises(ValueError):
                export_demo_embedding(source, [2, 5], Path(tmp) / "wrong")

    def test_modified_artifact_is_rejected_before_gather(self):
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead
        with tempfile.TemporaryDirectory() as tmp:
            export_multihead(torch.randn(7, 3), [3, 4], tmp)
            path = Path(tmp) / "table.mer"
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                MappedMultiHeadEmbedding(tmp)

    def test_factory_layout_checks_and_cleanup_on_failure(self):
        from functools import partial
        from unittest.mock import patch
        from tools.engram_row_adapter import export_multihead, open_demo_embedding, MappedMultiHeadEmbedding
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.arange(21, dtype=torch.float32).reshape(7, 3)
            export_multihead(source, [3, 4], tmp, 2)
            factory = partial(open_demo_embedding, tmp)
            with factory(list_of_N=[3, 4], D=3) as mapped:
                ids = torch.tensor([[2, 3], [0, 0]])
                expected = torch.nn.functional.embedding(ids + torch.tensor([0, 3]), source)
                self.assertTrue(torch.equal(mapped(ids), expected))
                with self.assertRaises(RuntimeError):
                    mapped.train()
            self.assertTrue(mapped.table.file.closed)
            # Same total rows does not mean the head addressing is compatible.
            for sizes, dim in (([4, 3], 3), ([3, 4], 4)):
                opened = MappedMultiHeadEmbedding(tmp)
                with patch("tools.engram_row_adapter.MappedMultiHeadEmbedding", return_value=opened), \
                        self.assertRaisesRegex(ValueError, "requesting Engram layer"):
                    factory(list_of_N=sizes, D=dim)
                self.assertTrue(opened.table.file.closed)


if __name__ == "__main__":
    unittest.main()
