import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


@unittest.skipUnless(torch is not None, "optional trained merge experiment needs torch/numpy")
class CosineMergeTest(unittest.TestCase):
    def test_direct_representative_check_prevents_transitive_drift(self):
        from tools.benchmark_cosine_merge import structural_merge, fidelity
        angles = np.deg2rad([0, 7, 14])
        rows = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        mapping = structural_merge(rows, ["ab", "abc", "abcd"], 0.99)
        self.assertEqual(mapping.tolist(), [0, 0, 2])
        self.assertGreaterEqual(fidelity(rows, rows[mapping])["minimum_cosine"], 0.99)

    def test_cosine_only_does_not_bound_norm_error_and_guard_rejects_it(self):
        from tools.benchmark_cosine_merge import structural_merge, fidelity
        rows = np.array([[1, 0], [2, 0], [-1, 0], [0, 0]], dtype=np.float32)
        grams = ["ab", "abc", "abcd", "abcde"]
        mapping = structural_merge(rows, grams, 0.99)
        self.assertEqual(mapping.tolist(), [0, 0, 2, 3])
        self.assertAlmostEqual(fidelity(rows, rows[mapping])["relative_l2_max"], 0.5)
        self.assertEqual(structural_merge(rows, grams, 0.99, 0.99).tolist(), [0, 1, 2, 3])

    def test_suffix_ancestors_allowed_unrelated_equal_vectors_kept(self):
        from tools.benchmark_cosine_merge import structural_merge
        rows = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
        self.assertEqual(structural_merge(rows, ["bc", "abc", "xyz"], 0.99).tolist(), [0, 0, 2])


if __name__ == "__main__":
    unittest.main()
