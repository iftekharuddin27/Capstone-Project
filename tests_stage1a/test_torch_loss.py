import unittest

try:
    import torch
except ImportError:  # The low-resource local environment is allowed to omit PyTorch.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed; synthetic tensor tests are Kaggle-ready")
class TinyTensorLossTests(unittest.TestCase):
    def test_targets_and_missing_masks(self):
        from corrected_pipeline.training import compute_multitask_loss

        outputs = {
            "logits_3class": torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
            "logit_hate": torch.tensor([0.2, -0.4]),
            "logit_sarcasm": torch.tensor([-0.7, 0.8]),
        }
        batch = {
            "labels_main": torch.tensor([0, 1]),
            "labels_hate": torch.tensor([1.0, 0.0]),
            "labels_sarcasm": torch.tensor([0.0, 1.0]),
            "mask_hate": torch.tensor([True, False]),
            "mask_sarcasm": torch.tensor([False, True]),
        }
        loss, components = compute_multitask_loss(
            outputs, batch, {"main": 0.4, "hate": 0.3, "sarcasm": 0.3}, None
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(components["hate"])
        self.assertIsNotNone(components["sarcasm"])

        batch["mask_hate"] = torch.tensor([False, False])
        batch["mask_sarcasm"] = torch.tensor([False, False])
        _, masked = compute_multitask_loss(
            outputs, batch, {"main": 0.4, "hate": 0.3, "sarcasm": 0.3}, None
        )
        self.assertIsNone(masked["hate"])
        self.assertIsNone(masked["sarcasm"])


if __name__ == "__main__":
    unittest.main()

