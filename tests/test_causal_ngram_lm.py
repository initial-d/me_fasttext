from argparse import Namespace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


@unittest.skipUnless(torch is not None, "causal LM reference requires optional torch/numpy")
class CausalMemoryTest(unittest.TestCase):
    def test_train_only_registry_and_original_split_boundaries(self):
        from tools.causal_ngram_lm import fit_registry, encode_segment
        from tools.benchmark_causal_lm_compression import text_splits
        split = text_splits("ab" * 40 + "XX" * 5 + "YY" * 5, 0)
        registry = fit_registry(split["train"], 4, 1)
        self.assertEqual(registry["characters"], ["a", "b"])
        self.assertFalse(any("X" in g or "Y" in g for g in registry["grams"]))
        validation = encode_segment(split["validation"], registry)
        self.assertEqual(validation["unknown_characters"], 10)
        self.assertTrue(validation["memory_ids"].eq(0).all())
        known = encode_segment("baba", registry)
        self.assertTrue(known["memory_ids"][0].eq(0).all())
        self.assertEqual(known["memory_ids"][1].ne(0).sum(), 1)

    def test_future_perturbation_cannot_change_previous_keys_or_logits(self):
        from tools.causal_ngram_lm import fit_registry, encode_segment, CausalNgramLM
        torch.manual_seed(7)
        registry = fit_registry("abababababababab", 4, 1)
        model = CausalNgramLM(3, len(registry["grams"]), 8).eval()
        a, b = [encode_segment(t, registry) for t in ("ababababab", "abababXXXX")]
        self.assertTrue(torch.equal(a["memory_ids"][:6], b["memory_ids"][:6]))
        with torch.no_grad():
            la, _ = model(a["characters"][None], a["memory_ids"][None])
            lb, _ = model(b["characters"][None], b["memory_ids"][None])
        torch.testing.assert_close(la[:, :6], lb[:, :6], atol=0, rtol=0)

    def test_streaming_scores_include_unknown_targets_and_keep_state(self):
        from tools.causal_ngram_lm import fit_registry, encode_segment, CausalNgramLM, evaluate
        registry = fit_registry("abababababababab", 3, 1)
        model = CausalNgramLM(3, len(registry["grams"]), 8).eval()
        segment = encode_segment("abababXabaYab", registry)
        a, b = [evaluate(model, segment, burn_in=2, chunk_size=n) for n in (1, 20)]
        self.assertEqual(int(a["targets"].eq(0).sum()), 2)
        self.assertEqual(len(a["targets"]), len(segment["characters"]) - 3)
        torch.testing.assert_close(a["logits"], b["logits"], atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(a["positions"], b["positions"])

    def test_full_softmax_metrics_against_scalar_probability_oracle(self):
        from tools.causal_ngram_lm import prediction_delta
        from torch.nn import functional as F
        def scores(logits):
            targets = torch.tensor([0, 1])
            top = logits.argmax(-1)
            return {"targets": targets, "logits": logits, "top1": top.numpy(),
                    "correct": top.eq(targets).numpy(),
                    "loss": F.cross_entropy(logits, targets, reduction="none").numpy().astype(np.float64)}
        reference = scores(torch.tensor([[0., 1., 2.], [2., 1., 0.]], dtype=torch.float64))
        candidate = scores(torch.tensor([[.5, 1., 2.], [2., 1.1, 0.]], dtype=torch.float64))
        delta = prediction_delta(candidate, reference)
        probabilities = []
        for data in (reference, candidate):
            x = data["logits"].numpy()
            probabilities.append(np.exp(x) / np.exp(x).sum(1, keepdims=True))
        p, q = probabilities
        expected_kl = np.mean(np.sum(p * np.log(p / q), axis=1))
        expected_ratio = np.exp(np.mean([-np.log(q[0, 0]) + np.log(p[0, 0]),
                                        -np.log(q[1, 1]) + np.log(p[1, 1])])) - 1
        self.assertAlmostEqual(delta["mean_prediction_kl_nats"], expected_kl, places=12)
        self.assertAlmostEqual(delta["perplexity_relative_change"], expected_ratio, places=12)
        self.assertEqual(delta["top1_changed_fraction"], 0)

    def test_validation_selection_rejects_distribution_shift_despite_low_ppl(self):
        from tools.benchmark_causal_lm_compression import select_variant
        lossless = {"name": "lossless", "memory_raw_to_artifact_factor": 1.1,
                    "validation_delta": {"perplexity_relative_change": 0,
                                         "mean_prediction_kl_nats": 0, "top1_changed_fraction": 0}}
        bad = {"name": "too-different", "memory_raw_to_artifact_factor": 9,
               "validation_delta": {"perplexity_relative_change": -.001,
                                    "mean_prediction_kl_nats": .01, "top1_changed_fraction": 0},
               "test": {"character_perplexity": 1}}
        self.assertEqual(select_variant([lossless, bad], .005, .0001, .01)["name"], "lossless")

    def test_identical_scores_have_zero_paired_character_block_interval(self):
        from tools.benchmark_causal_lm_compression import paired_ppl_interval
        scores = {"positions": np.array([0, 1, 256, 257]), "loss": np.array([1., 2., 3., 4.])}
        result = paired_ppl_interval(scores, scores, repeats=100)
        self.assertEqual(result["perplexity_relative_change_95_percent_interval"], [0, 0])
        self.assertEqual(result["observed_blocks"], 2)

    def test_test_budget_audit_preserves_validation_choice(self):
        from copy import deepcopy
        from tools.analyze_causal_lm_compression import analyze
        chosen = {"name": "validation-choice", "memory_raw_to_artifact_factor": 1.5,
                  "full_parameter_payload_to_compressed_factor": 1.4,
                  "parameter_and_registry_to_compressed_factor": 1.3}
        report = {"failures": [], "runs": [{"max_order": 4, "seed": 3, "variants": [chosen],
                 "selection": {"name": chosen["name"], "test_delta": {"perplexity_relative_change": 0,
                                 "mean_prediction_kl_nats": .01, "top1_changed_fraction": 0},
                               "validation_ppl_budget": .005, "validation_mean_kl_budget": .0001,
                               "validation_top1_flip_budget": .01, "test": {}, "test_paired_block_interval": {}},
                 "baseline": {"test": {}}, "inference_memory_ablation": {"test": {"metrics": {"character_perplexity": 10}}}}]}
        before = deepcopy(report)
        audit = analyze(report)
        self.assertEqual(report, before)
        self.assertEqual(audit["models"][0]["chosen_variant"], chosen["name"])
        self.assertFalse(audit["models"][0]["test_all_budgets_passed"])
        self.assertEqual(audit["test_budget_exceedances"][0]["metrics"], ["mean_prediction_kl"])


@unittest.skipUnless(torch is not None and shutil.which("cc"), "native LM test needs torch/numpy and C compiler")
class CausalNativeTest(unittest.TestCase):
    def test_tiny_trained_lm_queries_native_storage_and_has_bit_exact_logits(self):
        from tools.benchmark_causal_lm_compression import run, text_splits
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            library = output / ("rows.dylib" if sys.platform == "darwin" else "rows.so")
            source = Path(__file__).parents[1] / "native" / "meft_rows.c"
            subprocess.run(["cc", "-std=c11", "-O2", "-fPIC",
                            "-dynamiclib" if sys.platform == "darwin" else "-shared",
                            str(source), "-lz", "-o", str(library)], check=True, capture_output=True)
            cfg = Namespace(min_count=2, dim=8, steps=4, batch_size=2, sequence_length=4,
                            dense_lr=.002, memory_lr=.01, burn_in=4, eval_chunk=8,
                            output_dir=output, native_library=library, min_norm_ratio=.99,
                            block_rows=4, cache_bytes=4096, max_ppl_increase=.005,
                            max_kl=.0001, max_top1_flip=.01, eval_seed=42)
            split = text_splits("abc abcd abce " * 100, 0)
            result = run(split, cfg, 4, 3, [.99, .90])
            self.assertTrue(result["variants"][0]["both_split_logits_and_cross_entropy_bit_identical"])
            self.assertEqual(result["variants"][0]["validation_delta"]["perplexity_relative_change"], 0)
            self.assertEqual(result["variants"][0]["validation_delta"]["mean_prediction_kl_nats"], 0)
            self.assertTrue(result["inference_memory_ablation"])
            self.assertTrue(result["selection"]["test_paired_block_interval"])
            self.assertEqual(len(result["variants"]), 3)
            self.assertGreater(result["updated_memory_row_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
