import unittest

from corrected_pipeline.labels import (
    BANGLA_SOURCES,
    CLASS_2_SEMANTIC_STATUS,
    GRL_SOURCE_TO_ID,
    MAIN_LABELS,
    LabelError,
    LabelProvenance,
    gold_auxiliary_targets,
    normalize_main_labels,
    proxy_auxiliary_targets,
)


class LabelTests(unittest.TestCase):
    def test_supervisor_confirmed_class_two_is_sarcastic(self):
        self.assertEqual(MAIN_LABELS[2], "Sarcastic")
        self.assertEqual(CLASS_2_SEMANTIC_STATUS, "confirmed_by_supervisor")

    def test_canonical_main_mapping_accepts_only_zero_one_two(self):
        self.assertEqual(normalize_main_labels([0, "1", 2]), (0, 1, 2))
        with self.assertRaises(LabelError):
            normalize_main_labels([3])

    def test_five_bangla_sources(self):
        self.assertEqual(
            set(BANGLA_SOURCES),
            {"ALERT", "BD_SHS", "BenSarc", "BanglaSarc3", "BIDWESH"},
        )
        self.assertEqual(
            GRL_SOURCE_TO_ID,
            {"ALERT": 0, "BD_SHS": 1, "BenSarc": 2, "BanglaSarc3": 3, "BIDWESH": 4},
        )

    def test_proxy_generation_requires_explicit_flag(self):
        with self.assertRaises(LabelError):
            proxy_auxiliary_targets([0, 1, 2], explicitly_enabled=False)
        targets = proxy_auxiliary_targets([0, 1, 2], explicitly_enabled=True)
        self.assertEqual(targets.hate, (0.0, 1.0, 0.0))
        self.assertEqual(targets.sarcasm, (0.0, 0.0, 1.0))
        self.assertEqual(targets.provenance, LabelProvenance.PROXY_AUXILIARY)

    def test_gold_axes_overlap_and_missing_masks(self):
        targets = gold_auxiliary_targets([1, 1, float("nan"), 0], [1, 0, 1, "NA"])
        self.assertEqual(targets.hate, (1.0, 1.0, 0.0, 0.0))
        self.assertEqual(targets.sarcasm, (1.0, 0.0, 1.0, 0.0))
        self.assertEqual(targets.hate_mask, (True, True, False, True))
        self.assertEqual(targets.sarcasm_mask, (True, True, True, False))
        self.assertTrue(targets.hate[0] == targets.sarcasm[0] == 1.0)


if __name__ == "__main__":
    unittest.main()
