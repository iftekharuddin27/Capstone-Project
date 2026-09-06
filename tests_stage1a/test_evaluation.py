import unittest

from corrected_pipeline.evaluation import classification_metrics


class EvaluationTests(unittest.TestCase):
    def test_required_metrics(self):
        metrics = classification_metrics([0, 1, 2, 1], [0, 1, 0, 2])
        for key in (
            "accuracy", "macro_f1", "weighted_f1", "per_class",
            "confusion_matrix", "hate_f1", "sarcasm_f1", "hateful_class_recall",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 3)


if __name__ == "__main__":
    unittest.main()

