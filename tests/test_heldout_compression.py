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


@unittest.skipUnless(torch is not None, "held-out reference needs optional torch/numpy")
class HeldoutProtocolTest(unittest.TestCase):
    def test_vocabulary_and_ngrams_fit_training_only(self):
        from tools.benchmark_heldout_compression import split_tokens, context_pairs
        from tools.train_ngram_similarity import prepare
        split = split_tokens(" ".join(["playing"] * 80 + ["validationword"] * 10 + ["testword"] * 10))
        words, grams, _, _, _, _, counts = prepare(" ".join(split["train"]), 8, 50, 1)
        self.assertEqual(words, ["playing"])
        self.assertEqual(dict(counts), {"playing": 80})
        self.assertNotIn("test", grams)
        self.assertEqual(context_pairs(split["validation"], words).shape, (0, 3))
        self.assertEqual([len(v) for v in split.values()], [80, 10, 10])

    def test_oov_gaps_never_collapse_or_cross_split_boundary(self):
        from tools.benchmark_heldout_compression import context_pairs
        words = ["one", "two"]
        pairs = context_pairs(["one", "unknown", "unknown", "two"], words)
        self.assertEqual(len(pairs), 0)
        self.assertEqual(context_pairs(["one", "unknown", "two"], words).tolist(), [[0, 1, 0], [1, 0, 2]])
        self.assertEqual(len(context_pairs(["one"], words)), 0)
        self.assertEqual(len(context_pairs(["two"], words)), 0)

    def test_fixed_evaluation_ids_negatives_and_original_positions(self):
        from collections import Counter
        from tools.benchmark_heldout_compression import evaluation_set
        tokens, words = ["one", "two"] * 20, ["one", "two"]
        counts = Counter({"one": 80, "two": 20})
        a = evaluation_set(tokens, words, counts, 10, 5, 42)
        b = evaluation_set(tokens, words, counts, 10, 5, 42)
        self.assertEqual(a["sha256"], b["sha256"])
        for key in ("pairs", "positions", "negative"):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertNotEqual(a["sha256"], evaluation_set(tokens, words, counts, 10, 5, 43)["sha256"])
        self.assertEqual(a["token_coverage"], 1)
        self.assertEqual(a["negative"].shape, (10, 5))

    def test_composition_keeps_word_rows_and_ngram_multiplicity(self):
        from tools.benchmark_heldout_compression import compose
        weights = torch.tensor([[0., 0.], [3., 6.], [1., 2.], [7., 4.]])
        # Word row 1 + gram row 2 twice + gram row 3, followed by padding.
        actual = compose(weights, torch.tensor([[1, 2, 2, 3, 0]]), torch.tensor([4]))
        torch.testing.assert_close(actual, torch.tensor([[3., 3.5]]))

    def test_objective_and_ties_against_scalar_oracle(self):
        from tools.benchmark_heldout_compression import score
        hidden = torch.tensor([[1., 2.], [-1., 0.]])
        outputs = torch.tensor([[.5, 1.], [1., -1.]])
        data = {"pairs": np.array([[0, 1], [1, 0]]), "negative": np.array([[0, 1], [1, 0]])}
        result = score(hidden, outputs, data, batch_size=1)
        for i, (center, target) in enumerate(data["pairs"]):
            positive = float(np.dot(hidden[center].numpy(), outputs[target].numpy()))
            negatives = [float(np.dot(hidden[center].numpy(), outputs[j].numpy())) for j in data["negative"][i]]
            expected_loss = np.logaddexp(0, -positive) + sum(np.logaddexp(0, n) for n in negatives)
            expected_win = sum(1 if positive > n else .5 if positive == n else 0 for n in negatives) / 2
            self.assertAlmostEqual(result["loss"][i], expected_loss, places=6)
            self.assertEqual(result["discrimination"][i], expected_win)

    def test_threshold_selection_ignores_test_metrics(self):
        from tools.benchmark_heldout_compression import choose_variant
        feasible = {"name": "lossless", "ngram_raw_to_artifact_factor": 1.1,
                    "validation_delta": {"loss_relative_change": 0, "discrimination_change": 0}}
        invalid = {"name": "overcompressed", "ngram_raw_to_artifact_factor": 10,
                   "validation_delta": {"loss_relative_change": .1, "discrimination_change": 0},
                   "test": {"sgns_loss_per_pair": 0}}
        self.assertEqual(choose_variant([feasible, invalid], .005, .005)["name"], "lossless")

    def test_identical_predictions_have_zero_paired_block_interval(self):
        from tools.benchmark_heldout_compression import block_bootstrap
        scores = {"loss": np.array([1., 2., 3., 4.]), "discrimination": np.array([.2, .5, .7, .9])}
        report = block_bootstrap(scores, scores, np.array([0, 1, 128, 256]), repeats=100)
        self.assertEqual(report["observed_blocks"], 3)
        self.assertEqual(report["loss_relative_change_95_percent_interval"], [0, 0])
        self.assertEqual(report["discrimination_change_95_percent_interval"], [0, 0])


@unittest.skipUnless(torch is not None and shutil.which("cc"), "native reference needs torch/numpy and C compiler")
class HeldoutNativeTest(unittest.TestCase):
    def test_real_training_native_decode_lossless_identity_and_selection(self):
        from tools.benchmark_heldout_compression import run, split_tokens
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            library = directory / ("rows.dylib" if sys.platform == "darwin" else "rows.so")
            source = Path(__file__).parents[1] / "native" / "meft_rows.c"
            subprocess.run(["cc", "-std=c11", "-O2", "-fPIC",
                            "-dynamiclib" if sys.platform == "darwin" else "-shared",
                            str(source), "-lz", "-o", str(library)], check=True, capture_output=True)
            cfg = Namespace(dim=8, vocab_limit=10, min_count=1, steps=5, batch_size=8, negatives=2,
                            lr=.01, eval_pairs=20, eval_seed=42, native_library=library, output_dir=directory,
                            min_norm_ratio=.99, max_loss_increase=.005, max_discrimination_drop=.005)
            tokens = ["playing", "played", "plays", "play"] * 100
            entry = run(split_tokens(" ".join(tokens)), cfg, 8, 3, [.99, .90])
            self.assertTrue(entry["variants"][0]["full_embedding_and_both_split_outputs_bit_identical"])
            self.assertEqual(entry["variants"][0]["validation_delta"]["loss_relative_change"], 0)
            self.assertEqual(len(entry["variants"]), 3)
            self.assertTrue(entry["selection"]["test_paired_token_block_bootstrap"])
            for variant in entry["variants"]:
                self.assertTrue(variant["native_rows_byte_exact_to_reconstruction"])
                expected = entry["original_full_vector_payload_bytes"] / variant["full_compressed_vector_payload_bytes"]
                self.assertEqual(variant["full_vector_payload_to_compressed_factor"], expected)
                if variant["approximate"]:
                    self.assertGreaterEqual(variant["row_fidelity"]["minimum_cosine"], variant["minimum_cosine"] - 1e-12)


if __name__ == "__main__":
    unittest.main()
