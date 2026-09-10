import unittest

from corrected_pipeline.annotation_join import (
    AnnotationJoinError,
    require_stable_join_identifier,
    validate_one_to_one_join,
)


class AnnotationJoinTests(unittest.TestCase):
    def test_text_only_join_is_rejected(self):
        with self.assertRaisesRegex(AnnotationJoinError, "stable ID"):
            require_stable_join_identifier(
                ["text", "hate_type", "sarcasm_type"], ["text_clean", "class"]
            )

    def test_duplicate_or_unmatched_stable_ids_are_rejected(self):
        self.assertEqual(
            require_stable_join_identifier(["sample_id", "hate_type"], ["sample_id", "class"]),
            "sample_id",
        )
        with self.assertRaisesRegex(AnnotationJoinError, "not one-to-one"):
            validate_one_to_one_join(["a", "a"], ["a", "b"])
        with self.assertRaisesRegex(AnnotationJoinError, "unmatched"):
            validate_one_to_one_join(["a", "c"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
