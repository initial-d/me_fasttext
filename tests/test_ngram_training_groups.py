import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "optional training reference needs PyTorch")
class NgramGroupsTest(unittest.TestCase):
    def test_same_word_training_signature_is_narrower_than_prefix_relation(self):
        from tools.train_ngram_similarity import prepare
        words, grams, signatures, _, _, _, _ = prepare(" " .join(["playing", "playing", "played", "play"] * 10), 8, 10, 1)
        ids = {g: i for i, g in enumerate(grams)}
        self.assertNotEqual(signatures[ids["play"]], signatures[ids["playi"]])
        self.assertEqual(signatures[ids["playi"]], signatures[ids["playin"]])
        self.assertEqual({words[i] for i, _ in signatures[ids["playi"]]}, {"playing"})

    def test_shared_gradient_preserves_initial_offset_but_raises_cosine_with_large_update(self):
        initial = torch.tensor([[0.001, 0.0], [0.0, 0.001]])
        gradient_update = torch.tensor([0.2, 0.1])
        trained = initial + gradient_update
        self.assertTrue(torch.allclose(trained[0] - trained[1], initial[0] - initial[1]))
        cosine = torch.nn.functional.cosine_similarity(trained[0:1], trained[1:2])
        self.assertGreater(float(cosine), 0.99)
        self.assertFalse(torch.equal(trained[0].view(torch.uint8), trained[1].view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
