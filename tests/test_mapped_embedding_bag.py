import ctypes
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and shutil.which("cc"), "optional PyTorch and compiler required")
class MappedBagTest(unittest.TestCase):
    def test_weighted_sum_preserves_each_occurrence_with_bounded_chunking(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                source = torch.nn.EmbeddingBag(12, 5, mode="sum", padding_idx=0,
                                               include_last_offset=True, dtype=dtype).eval()
                with torch.no_grad():
                    source.weight.copy_(torch.arange(60).reshape(12, 5).to(dtype) / 13)
                    source.weight[0] = float("nan")
                original_bytes = source.weight.detach().view(torch.uint8).numpy().tobytes()
                export_embedding_bag(source, tmp, block_rows=3, predictor="xor")
                # Repeated IDs have different positive, negative and zero weights.
                ids = torch.tensor([2, 2, 0, 7, 2, 0, 0, 11], dtype=torch.int32)
                weights = torch.tensor([.25, -2, float("nan"), 0, 3, float("inf"), -1, .1], dtype=dtype)
                offsets = torch.tensor([0, 3, 3, 5, 7, 8], dtype=torch.int32)
                oracle = []
                for a, b in zip(offsets[:-1], offsets[1:]):
                    valid = ids[a:b] != 0
                    rows = source.weight[ids[a:b][valid]].double()
                    scale = weights[a:b][valid].double()
                    oracle.append((rows * scale[:, None]).sum(0))
                oracle = torch.stack(oracle).float().to(dtype)
                for strategy, method, chunk in (("unique", "auto", 1024), ("unique", "sort", 1024),
                                                 ("chunked", "auto", 1), ("chunked", "auto", 2),
                                                 ("chunked", "auto", 1024)):
                    with self.subTest(strategy=strategy, method=method, chunk=chunk):
                        with open_embedding_bag(tmp, native_library=self.library, strategy=strategy,
                                                unique_method=method, chunk_rows=chunk) as mapped:
                            actual = mapped(ids, offsets, per_sample_weights=weights)
                            expected = source(ids, offsets, per_sample_weights=weights) if strategy == "unique" else oracle
                            self.assertTrue(torch.equal(actual, expected))
                            self.assertFalse(actual.requires_grad)
                            self.assertEqual(mapped.gather.table.rows(range(12)), original_bytes)
                            empty = mapped(torch.empty(0, dtype=torch.int64), torch.tensor([0, 0, 0]),
                                           per_sample_weights=torch.empty(0, dtype=dtype))
                            self.assertEqual(empty.shape, (2, 5))
                            self.assertTrue(empty.eq(0).all())

    def test_weighted_noncontiguous_fixed_bags_and_nonfinite_scalars(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.nn.EmbeddingBag(8, 3, mode="sum").eval()
            export_embedding_bag(source, tmp)
            ids = torch.tensor([[1, 2, 1], [4, 4, 7]]).t()
            weights = torch.tensor([[.5, -1, 2], [3, 0, -.25]]).t()
            self.assertFalse(weights.is_contiguous())
            for strategy in ("unique", "chunked"):
                with open_embedding_bag(tmp, native_library=self.library, strategy=strategy) as mapped:
                    torch.testing.assert_close(mapped(ids, per_sample_weights=weights),
                                               source(ids, per_sample_weights=weights), atol=5e-7, rtol=5e-7)
                    special = torch.tensor([[float("nan"), 1., 1.], [1., float("inf"), 1.]]).t()
                    torch.testing.assert_close(mapped(ids, per_sample_weights=special),
                                               source(ids, per_sample_weights=special), equal_nan=True)
                    output = mapped(ids, per_sample_weights=weights)
                    output.zero_()
                    self.assertFalse(mapped(ids, per_sample_weights=weights).eq(0).all())

    def test_weighted_inputs_reject_wrong_shape_dtype_training_and_mean(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.nn.EmbeddingBag(8, 3, mode="sum").eval()
            export_embedding_bag(source, tmp)
            ids = torch.tensor([[1, 2, 1]])
            with open_embedding_bag(tmp, native_library=self.library) as mapped:
                for weights in (torch.ones(3), torch.ones_like(ids), torch.ones(ids.shape, dtype=torch.float64),
                                torch.ones(ids.shape, device="meta"), torch.ones(ids.shape, requires_grad=True)):
                    with self.assertRaises(ValueError):
                        mapped(ids, per_sample_weights=weights)
                with self.assertRaises(TypeError):
                    mapped(ids, per_sample_weights=[1, 1, 1])
                mapped.mode = "mean"
                with self.assertRaisesRegex(ValueError, "sum mode"):
                    mapped(ids, per_sample_weights=torch.ones(ids.shape))

    def test_weighted_native_abi_checks_weight_capacity_and_null_pointer(self):
        from tools.native_row_store import NativeRows, pack_native
        with tempfile.TemporaryDirectory() as tmp:
            raw, table = Path(tmp) / "raw", Path(tmp) / "table"
            raw.write_bytes(struct.pack("<6f", 1., 2., 3., 4., 5., 6.))
            pack_native(raw, table, 8, block_rows=2)
            with NativeRows(table, self.library) as native:
                ids = (ctypes.c_uint64 * 3)(0, 2, 0)
                offsets = (ctypes.c_uint64 * 3)(0, 2, 3)
                weights = (ctypes.c_float * 3)(.5, -2., 3.)
                output = (ctypes.c_float * 4)()
                def call(address=ctypes.addressof(weights), size=12, mean=False):
                    native.embedding_bag_into(ctypes.addressof(ids), 3, ctypes.addressof(offsets), 2,
                                              1, mean, 2**64 - 1, 1, ctypes.addressof(output), 4,
                                              weights_address=address, weights_bytes=size)
                call()
                self.assertEqual(list(output), [-9.5, -11., 3., 6.])
                for kwargs in ({"size": 11}, {"size": 0}, {"size": -1},
                               {"address": 0}, {"mean": True}):
                    with self.assertRaises(ValueError):
                        call(**kwargs)

    def test_dense_compaction_and_budget_fallback_preserve_sorted_original_ids(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.nn.EmbeddingBag(32, 7, mode="mean", padding_idx=0).eval()
            export_embedding_bag(source, tmp)
            ids = torch.tensor([[31, 2, 31, 0, 4]]).repeat(40, 1)
            with open_embedding_bag(tmp, native_library=self.library) as dense:
                actual = dense(ids)
                self.assertEqual(dense.last_unique_method, "dense")
                with open_embedding_bag(tmp, native_library=self.library, unique_method="sort") as sorted_bag:
                    self.assertTrue(torch.equal(actual, sorted_bag(ids)))
                self.assertTrue(torch.equal(actual, source(ids)))
                # A tiny query must not scan a large addressable row domain.
                dense(torch.tensor([[31]]))
                self.assertEqual(dense.last_unique_method, "sort")
            for budget in (0, 32 * 16 - 1):
                with open_embedding_bag(tmp, native_library=self.library, unique_index_bytes=budget) as fallback:
                    self.assertTrue(torch.equal(actual, fallback(ids)))
                    self.assertEqual(fallback.last_unique_method, "sort")
            with open_embedding_bag(tmp, native_library=self.library, unique_index_bytes=32 * 16) as exact:
                self.assertTrue(torch.equal(actual, exact(ids)))
                self.assertEqual(exact.last_unique_method, "dense")

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.tmp.name) / "rows.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared",
                        str(Path(__file__).parents[1] / "native/meft_rows.c"), "-lz", "-o", str(cls.library)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_three_dtypes_ragged_padding_and_chunk_sizes_against_double_oracle(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for mode in ("sum", "mean"):
                with self.subTest(dtype=dtype, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    torch.manual_seed(19)
                    source = torch.nn.EmbeddingBag(19, 7, mode=mode, padding_idx=0, dtype=dtype,
                                                   include_last_offset=True).eval()
                    with torch.no_grad():
                        source.weight[0] = 123  # Padding must be excluded, even if its stored row is nonzero.
                        source.weight[2] = source.weight[1]
                    original = source.weight.detach().clone()
                    export_embedding_bag(source, tmp, block_rows=4, predictor="xor")
                    ids, offsets = torch.tensor([1, 2, 0, 4, 4, 7, 0]), torch.tensor([0, 3, 3, 6, 7])
                    expected = []
                    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
                        valid = ids[a:b][ids[a:b] != 0]
                        total = original[valid].double().sum(0)
                        expected.append(total / max(1, len(valid)) if mode == "mean" else total)
                    expected = torch.stack(expected).float().to(dtype)
                    outputs = []
                    for chunk in (1, 2, 1024):
                        with open_embedding_bag(tmp, chunk_rows=chunk, native_library=self.library, strategy="chunked") as mapped:
                            output = mapped(ids.to(torch.int32), offsets.to(torch.int32))
                            self.assertTrue(torch.equal(output, expected))
                            self.assertEqual(output.dtype, dtype)
                            self.assertEqual(mapped.gather.table.rows(range(19)),
                                             original.view(torch.uint8).numpy().tobytes())
                            outputs.append(output)
                    self.assertTrue(torch.equal(outputs[0], outputs[-1]))
                    self.assertTrue(torch.equal(original, source.weight))
                    # PyTorch's lower-precision accumulation can differ by rounding.
                    torch.testing.assert_close(outputs[0], source(ids, offsets),
                                               atol={torch.float32: 3e-7, torch.float16: 2e-3,
                                                     torch.bfloat16: .02}[dtype], rtol=.01)

    def test_half_subnormals_nonfinite_values_and_owned_output(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.nn.EmbeddingBag(4, 4, mode="sum", dtype=torch.float16)
            with torch.no_grad():
                source.weight[0] = torch.tensor([2**-24, -2**-24, 0., -0.])
                source.weight[1] = torch.tensor([float("inf"), float("nan"), 2., -3.])
            export_embedding_bag(source, tmp)
            with open_embedding_bag(tmp, native_library=self.library, strategy="chunked") as mapped:
                ids = torch.tensor([[0], [1]])
                actual = mapped(ids)
                torch.testing.assert_close(actual, source(ids), atol=0, rtol=0, equal_nan=True)
                actual.zero_()
                self.assertTrue(torch.isnan(mapped(ids)[1, 1]))

    def test_fixed_noncontiguous_and_empty_bags(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.nn.EmbeddingBag(10, 5, mode="mean").eval()
            export_embedding_bag(source, tmp)
            with open_embedding_bag(tmp, native_library=self.library) as mapped:
                query = torch.tensor([[1, 2, 3], [4, 5, 6]]).t()
                torch.testing.assert_close(mapped(query), source(query), atol=2e-7, rtol=2e-7)
                self.assertEqual(mapped(torch.empty((0, 3), dtype=torch.int64)).shape, (0, 5))
                self.assertEqual(mapped.gather.table.last_block_reads, 0)
                self.assertTrue(mapped(torch.empty((3, 0), dtype=torch.int64)).eq(0).all())
                self.assertEqual(mapped(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)).shape, (0, 5))
                with self.assertRaises(ValueError):
                    mapped(torch.tensor([1]), torch.tensor([1]))
                with self.assertRaises(IndexError):
                    mapped(torch.tensor([[10]]))
                with self.assertRaises(RuntimeError):
                    mapped.train()
                with self.assertRaises(ValueError):
                    mapped(query, per_sample_weights=torch.ones_like(query))
                self.assertIs(mapped.eval(), mapped)
            with self.assertRaises(ValueError):
                mapped(query)
            with self.assertRaises(ValueError):
                mapped(torch.empty((0, 3), dtype=torch.int64))

    def test_unique_queries_preserve_dense_bag_results_and_original_ids(self):
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        from unittest.mock import patch
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for mode in ("sum", "mean"):
                for padding in (None, 7):
                    with self.subTest(dtype=dtype, mode=mode, padding=padding), tempfile.TemporaryDirectory() as tmp:
                        source = torch.nn.EmbeddingBag(20, 6, mode=mode, padding_idx=padding, dtype=dtype).eval()
                        ids = torch.tensor([[19, 19, 7, 2], [2, 2, 2, 7], [7, 7, 7, 7]])
                        export_embedding_bag(source, tmp, predictor="xor")
                        with open_embedding_bag(tmp, native_library=self.library) as mapped:
                            actual = mapped(ids)
                            self.assertTrue(torch.equal(actual, source(ids)))
                            self.assertEqual(mapped.last_unique_rows, 3)
                            # The padding row need not occur in the current query.
                            other = torch.tensor([[2, 19, 2]])
                            # Keeping a configured but unused padding ID selects
                            # the same reduction kernel, including x86 BF16.
                            with patch("tools.mapped_embedding_bag.F.embedding_bag",
                                       wraps=torch.nn.functional.embedding_bag) as reduce:
                                other_actual = mapped(other)
                                used_padding = reduce.call_args.kwargs["padding_idx"]
                                self.assertEqual(used_padding, 2 if padding is not None else None)
                            self.assertTrue(torch.equal(other_actual, source(other)))
                            self.assertEqual(mapped.last_unique_rows, 2)
                            # Ragged bags include an empty bag and an all-padding bag.
                            ragged = torch.tensor([19, 2, 19, 7, 7])
                            offsets = torch.tensor([0, 3, 3])
                            self.assertTrue(torch.equal(mapped(ragged, offsets), source(ragged, offsets)))
                        source.include_last_offset = True
                        complete_dir = Path(tmp) / "complete-offsets"
                        export_embedding_bag(source, complete_dir, predictor="xor")
                        with open_embedding_bag(complete_dir, native_library=self.library) as mapped:
                            complete = torch.tensor([0, 3, 3, 5])
                            self.assertTrue(torch.equal(mapped(ragged, complete), source(ragged, complete)))
                            empty = mapped(torch.empty(0, dtype=torch.int64), torch.tensor([0, 0, 0]))
                            self.assertTrue(empty.eq(0).all())
                            self.assertEqual(empty.shape, (2, 6))

    def test_native_abi_rejects_bad_capacity_offsets_dtype_and_ids(self):
        from tools.native_row_store import NativeRows, pack_native
        with tempfile.TemporaryDirectory() as tmp:
            raw, table = Path(tmp) / "raw", Path(tmp) / "table"
            raw.write_bytes(struct.pack("<6f", 1., 2., 3., 4., 5., 6.))
            pack_native(raw, table, 8, block_rows=2)
            with NativeRows(table, self.library) as native:
                ids = (ctypes.c_uint64 * 3)(0, 2, 0)
                offsets = (ctypes.c_uint64 * 3)(0, 2, 3)
                output = (ctypes.c_float * 4)()
                def call(off=offsets, dtype=1, capacity=4, chunk=2):
                    native.embedding_bag_into(ctypes.addressof(ids), 3, ctypes.addressof(off), 2,
                                              dtype, True, 2**64 - 1, chunk, ctypes.addressof(output), capacity)
                call()
                self.assertEqual(list(output), [3., 4., 1., 2.])
                for kwargs in ({"capacity": 3}, {"dtype": 9}, {"chunk": 0},
                               {"off": (ctypes.c_uint64 * 3)(0, 3, 2)},
                               {"off": (ctypes.c_uint64 * 3)(0, 4, 3)}):
                    with self.assertRaises(ValueError):
                        call(**kwargs)
                ids[2] = 3
                with self.assertRaisesRegex(ValueError, "row ID"):
                    call()

    def test_trained_causal_model_uses_bag_path_without_new_parameters(self):
        from tools.causal_ngram_lm import CausalNgramLM, encode_segment, fit_registry, train_reference, evaluate, prediction_delta
        from tools.engram_row_adapter import export_embedding
        from tools.mapped_embedding_bag import MappedEmbeddingBag
        text = "abc abcd abce " * 30
        registry = fit_registry(text, 8, 2)
        segment = encode_segment(text, registry)
        torch.manual_seed(7)
        model = CausalNgramLM(len(registry["characters"]) + 1, len(registry["grams"]), 8)
        train_reference(model, segment, steps=10, batch_size=2, sequence_length=8)
        expected = evaluate(model, segment, burn_in=8)
        original = model.memory
        with tempfile.TemporaryDirectory() as tmp:
            export_embedding(original, tmp, block_rows=4, predictor="xor")
            with MappedEmbeddingBag(tmp, mode="mean", padding_idx=0,
                                    native_library=self.library, chunk_rows=3) as mapped:
                model.memory = mapped
                actual = evaluate(model, segment, burn_in=8)
                delta = prediction_delta(actual, expected)
                self.assertLess(delta["mean_prediction_kl_nats"], 1e-12)
                self.assertLess(delta["maximum_absolute_logit_change"], 1e-6)
                self.assertEqual(delta["top1_changed_fraction"], 0)
                self.assertFalse(list(mapped.parameters()))
                model.memory = original


if __name__ == "__main__":
    unittest.main()
