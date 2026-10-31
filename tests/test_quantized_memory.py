"""Codec/forwarding correctness fixtures; not compression/quality evidence."""
from argparse import Namespace
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(np is not None, "optional NumPy required")
class CodecTest(unittest.TestCase):
    def test_int8_code_equality_does_not_discard_different_scales(self):
        from tools.quantized_memory import encode_rows, decode_rows
        rows = np.array([[1., -.5], [2., -1.], [0., 0.]], dtype=np.float32)
        encoded = encode_rows(rows, "int8")
        np.testing.assert_array_equal(encoded[0, :2], encoded[1, :2])
        self.assertNotEqual(encoded[0].tobytes(), encoded[1].tobytes())
        decoded = decode_rows(encoded, 2, "int8")
        np.testing.assert_array_equal(decoded[1], decoded[0] * 2)
        np.testing.assert_array_equal(decoded[2], 0)
        # Independent nearest-even oracle: scale 1, +/-0.5 ties round to zero.
        ties = encode_rows(np.array([[127., .5, -.5, 1.5, -1.5]], dtype=np.float32), "int8")
        np.testing.assert_array_equal(ties[0, :5].view("i1"), [127, 0, 0, 2, -2])

    def test_finite_subnormal_zero_signed_zero_and_overflow(self):
        from tools.quantized_memory import encode_rows, decode_rows
        tiny = np.nextafter(np.float32(0), np.float32(1))
        rows = np.array([[tiny, -tiny], [0., -0.]], dtype=np.float32)
        for encoding in ("fp32", "fp16", "int8"):
            result = decode_rows(encode_rows(rows, encoding), 2, encoding)
            self.assertTrue(np.isfinite(result).all())
        self.assertEqual(decode_rows(encode_rows(rows, "fp32"), 2, "fp32").tobytes(), rows.tobytes())
        extreme = np.array([[np.finfo(np.float32).max, -np.finfo(np.float32).max]], dtype=np.float32)
        decoded = decode_rows(encode_rows(extreme, "int8"), 2, "int8")
        self.assertTrue(np.isfinite(decoded).all())
        np.testing.assert_allclose(decoded, extreme, rtol=2e-7, atol=0)
        for rows, encoding in (([[float("nan")]], "int8"), ([[float("inf")]], "fp32"), ([[1e10]], "fp16")):
            with self.assertRaises(ValueError):
                encode_rows(np.array(rows, dtype=np.float32), encoding)

    def test_failed_export_removes_only_new_owned_directory(self):
        from tools.quantized_memory import pack_quantized
        rows = np.ones((3, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bundle"
            with patch("tools.quantized_memory.pack_native", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    pack_quantized(rows, target, "int8")
            self.assertFalse(target.exists())
            target.mkdir()
            keep = target / "keep"
            keep.write_text("existing")
            with self.assertRaises(FileExistsError):
                pack_quantized(rows, target, "fp16")
            self.assertEqual(keep.read_text(), "existing")

    def test_file_export_streams_chunks_and_checks_original_id_hashes(self):
        from tools.quantized_memory import pack_quantized_file, encode_rows, decode_rows
        rows = np.array([[127., .5, -.5], [1., .3, -.2], [0., -0., 0.],
                         [2., .6, -.4], [1., .3, -.2]], dtype="<f4")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.f32"
            source.write_bytes(rows.tobytes())
            for encoding in ("fp32", "fp16", "int8"):
                with patch("tools.quantized_memory.encode_rows", wraps=encode_rows) as encoder:
                    target = Path(tmp) / encoding
                    meta = pack_quantized_file(source, target, 3, encoding, native=False, chunk_rows=2)
                    self.assertEqual([len(call.args[0]) for call in encoder.call_args_list], [2, 2, 1])
                encoded = encode_rows(rows, encoding)
                self.assertEqual((target / "table.rows").read_bytes(), encoded.tobytes())
                self.assertEqual(meta["source_float32_sha256"], hashlib.sha256(rows.tobytes()).hexdigest())
                expected = decode_rows(encoded, 3, encoding).tobytes()
                self.assertEqual(meta["decoded_float32_sha256"], hashlib.sha256(expected).hexdigest())
            bad = Path(tmp) / "bad"
            source.write_bytes(rows.tobytes() + b"x")
            with self.assertRaisesRegex(ValueError, "complete"):
                pack_quantized_file(source, bad, 3, "int8")
            self.assertFalse(bad.exists())
            rows[-1, 1] = np.inf
            source.write_bytes(rows.tobytes())
            with self.assertRaisesRegex(ValueError, "finite"):
                pack_quantized_file(source, bad, 3, "int8", chunk_rows=2)
            self.assertFalse(bad.exists())

    def test_same_size_source_mutation_is_rejected_by_streaming_audit(self):
        from tools.quantized_memory import pack_quantized_file, encode_rows
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.f32"
            source.write_bytes(np.ones((4, 2), dtype="<f4").tobytes())
            target = Path(tmp) / "bundle"
            calls = 0
            def mutate_after_encoding(chunk, encoding):
                nonlocal calls
                result = encode_rows(chunk, encoding)
                calls += 1
                if calls == 1:
                    changed = np.ones((4, 2), dtype="<f4")
                    changed[0, 0] = 2
                    source.write_bytes(changed.tobytes())
                return result
            with patch("tools.quantized_memory.encode_rows", side_effect=mutate_after_encoding):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    pack_quantized_file(source, target, 2, "int8", native=False, chunk_rows=2)
            self.assertFalse(target.exists())


@unittest.skipUnless(np is not None and shutil.which("cc"), "NumPy and native C compiler required")
class QuantizedNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.tmp.name) / ("rows.dylib" if sys.platform == "darwin" else "rows.so")
        subprocess.run(["cc", "-std=c11", "-O2", "-fPIC",
                        "-dynamiclib" if sys.platform == "darwin" else "-shared",
                        str(ROOT / "native/meft_rows.c"), "-lz", "-o", str(cls.library)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_original_ids_padding_repeated_queries_and_scale_sensitive_dedup(self):
        from tools.quantized_memory import pack_quantized, QuantizedRows, encode_rows, decode_rows
        rows = np.array([[0., 0.], [1., -.5], [2., -1.], [1., -.5]], dtype=np.float32)
        ids = np.array([3, 0, 2, 1, 3], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            for encoding in ("fp32", "fp16", "int8"):
                encoded = encode_rows(rows, encoding)
                rounded = decode_rows(encoded, 2, encoding)
                for native in (False, True):
                    target = Path(tmp) / f"{encoding}-{native}"
                    meta = pack_quantized(rows, target, encoding, native=native,
                                          deduplicate=native, max_block_bytes=16)
                    if native:
                        self.assertEqual(meta["storage"]["unique_rows"], 3)
                    with QuantizedRows(target, self.library, cache_bytes=16) as reader:
                        for _ in range(2):
                            self.assertEqual(reader.encoded_rows(ids).tobytes(), encoded[ids].tobytes())
                            self.assertEqual(reader.gather(ids).tobytes(), rounded[ids].tobytes())
                            out = np.empty((len(ids), 2), dtype="<f4")
                            scratch = np.empty((len(ids), reader.width), dtype="u1")
                            reader.gather_into(ids, out, encoded_scratch=scratch)
                            self.assertEqual(out.tobytes(), rounded[ids].tobytes())
                        self.assertEqual(reader.gather([]).shape, (0, 2))
                        for bad in ([-1], [4], [1.5], [[0]]):
                            with self.assertRaises(ValueError):
                                reader.gather(bad)
                            with self.assertRaises(ValueError):
                                reader.gather_into(bad, out)
                        with self.assertRaises(ValueError):
                            reader.gather_into(ids, out[:, :1])
                        out.flags.writeable = False
                        with self.assertRaises(ValueError):
                            reader.gather_into(ids, out)
                    with self.assertRaises(ValueError):
                        reader.gather(ids)

    def test_mark_compact_forwards_entire_quantized_representative(self):
        from tools.quantized_memory import pack_quantized_file, QuantizedRows, encode_rows, decode_rows
        rows = np.array([[0., 0.], [1., .3], [2., .6]], dtype=np.float32)
        mapping = [0, 1, 1]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shared"
            source = Path(tmp) / "source.f32"
            source.write_bytes(rows.tobytes())
            meta = pack_quantized_file(source, target, 2, "int8", representatives=mapping,
                                       row_order=[0, 1, 2], max_block_bytes=12, chunk_rows=1)
            self.assertEqual(meta["storage"]["marked_rows"], 2)
            self.assertFalse(meta["source_bytes_preserved"])
            expected = decode_rows(encode_rows(rows, "int8"), 2, "int8")[mapping]
            with QuantizedRows(target, self.library) as reader:
                self.assertEqual(reader.gather([2, 0, 1]).tobytes(), expected[[2, 0, 1]].tobytes())
                output = np.empty((3, 2), dtype="<f4")
                reader.gather_into([2, 0, 1], output)
                self.assertEqual(output.tobytes(), expected[[2, 0, 1]].tobytes())

    @unittest.skipUnless(torch is not None, "optional PyTorch required")
    def test_quantized_reader_preserves_model_reduction_logits_and_restores_on_failure(self):
        from tools.quantized_memory import pack_quantized, encode_rows, decode_rows
        from tools.causal_ngram_lm import CausalNgramLM, fit_registry, encode_segment, evaluate
        from tools.benchmark_quantized_structural import score_bundle
        registry = fit_registry("ababc " * 20, 3, 1)
        model = CausalNgramLM(len(registry["characters"]) + 1, len(registry["grams"]), 8).eval()
        segment = encode_segment("ababc Xababc " * 2, registry)
        original = model.memory
        rows = original.weight.detach().numpy().copy()
        cfg = Namespace(native_library=self.library, cache_bytes=512, burn_in=3, eval_chunk=8)
        with tempfile.TemporaryDirectory() as tmp:
            for encoding in ("fp32", "fp16", "int8"):
                rounded = decode_rows(encode_rows(rows, encoding), 8, encoding)
                with torch.no_grad():
                    original.weight.copy_(torch.from_numpy(rounded))
                expected = evaluate(model, segment, cfg.burn_in, cfg.eval_chunk)
                target = Path(tmp) / encoding
                pack_quantized(rows, target, encoding)
                actual = score_bundle(model, original, target, segment, cfg)
                self.assertTrue(torch.equal(actual["logits"], expected["logits"]))
                self.assertIs(model.memory, original)
            with patch("tools.benchmark_quantized_structural.evaluate", side_effect=RuntimeError("injected")):
                with self.assertRaises(RuntimeError):
                    score_bundle(model, original, target, segment, cfg)
            self.assertIs(model.memory, original)


@unittest.skipUnless(np is not None and torch is not None, "optional NumPy/PyTorch required")
class ValidationSelectionTest(unittest.TestCase):
    def test_size_winner_requires_all_budgets_and_cannot_use_test(self):
        from tools.benchmark_quantized_structural import choose_validation
        cfg = Namespace(max_ppl_increase=.005, max_kl=.0001, max_top1_flip=.01)
        def variant(name, size, ppl=0, kl=0, flip=0, encoding="int8"):
            return {"name": name, "encoding": encoding, "backend_with_key_registry_bytes": size,
                    "validation_delta": {"perplexity_relative_change": ppl,
                        "mean_prediction_kl_nats": kl, "top1_changed_fraction": flip}}
        good = variant("good", 100)
        candidates = [good, variant("bad-kl", 1, kl=.01), variant("bad-ppl", 2, ppl=.1),
                      variant("bad-flip", 3, flip=.1), variant("wrong-encoding", 4, encoding="fp16")]
        good["test"] = {"perplexity_relative_change": 1}
        self.assertIs(choose_validation(candidates, "int8", cfg), good)
        self.assertIsNone(choose_validation(candidates[1:], "int8", cfg))

    def test_analysis_uses_smallest_matched_baseline_and_preserves_negative_results(self):
        from tools.analyze_quantized_structural import analyze
        variants, selections = [], {}
        for encoding in ("fp32", "fp16", "int8"):
            for policy, size in (("raw", 100), ("identity", 90), ("layout", 80), ("dedup", 70), ("joint-0.99", 75)):
                variants.append({"name": f"{encoding}-{policy}", "backend_with_key_registry_bytes": size,
                    "full_model_serving_artifact_bytes": size + 100,
                    "fidelity_vs_float32": {"weighted_relative_frobenius": .01, "relative_l2_max": .1},
                    "added_fidelity_vs_quantization_only": {"weighted_relative_frobenius": .009},
                    "queries": {"warm_p50_ms": 2 if policy != "raw" else 1}})
            selections[encoding] = {"feasible": True, "name": f"{encoding}-joint-0.99",
                                   "test_all_budgets_passed": False, "test_delta": {"mean_prediction_kl_nats": .01}}
        report = {"commit": "fixture", "settings": {"planner": "native"}, "failures": [],
                  "runs": [{"category": "gru", "max_order": 4, "seed": 3, "variants": variants, "selection": selections}]}
        summary = analyze(report)
        self.assertEqual(summary["models"][0]["best_encoded_baseline"], "fp32-dedup")
        self.assertLess(summary["models"][0]["joint_099_added_saving_vs_best_encoded_baseline"], 0)
        self.assertEqual(len(summary["test_budget_exceedances"]), 3)
        self.assertEqual(summary["models"][0]["validation_selection"]["name"], "fp32-joint-0.99")


if __name__ == "__main__":
    unittest.main()
