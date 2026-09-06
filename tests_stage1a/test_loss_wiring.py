import unittest

from corrected_pipeline.loss_wiring import LOSS_BINDINGS, validate_loss_bindings


class LossWiringTests(unittest.TestCase):
    def test_exact_targets_and_masks(self):
        validate_loss_bindings()
        self.assertEqual(LOSS_BINDINGS["main"]["target"], "labels_main")
        self.assertEqual(LOSS_BINDINGS["hate"]["target"], "labels_hate")
        self.assertEqual(LOSS_BINDINGS["hate"]["mask"], "mask_hate")
        self.assertEqual(LOSS_BINDINGS["sarcasm"]["target"], "labels_sarcasm")
        self.assertEqual(LOSS_BINDINGS["sarcasm"]["mask"], "mask_sarcasm")

    def test_cross_wiring_is_rejected(self):
        broken = {task: dict(binding) for task, binding in LOSS_BINDINGS.items()}
        broken["hate"]["target"] = "labels_sarcasm"
        with self.assertRaises(ValueError):
            validate_loss_bindings(broken)


if __name__ == "__main__":
    unittest.main()

