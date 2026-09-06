import csv
import tempfile
import unittest
from pathlib import Path

from corrected_pipeline.data_validation import (
    DataValidationError,
    SplitSpec,
    sha256_file,
    validate_canonical_data,
)


class DataValidationTests(unittest.TestCase):
    def _write(self, path, columns, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _fixture(self, root):
        definitions = {
            "en_train": ([{"text_clean": "en-a", "class": 0}], None),
            "en_validation": ([{"text_clean": "en-b", "class": 1}], None),
            "en_test": ([{"text_clean": "en-c", "class": 2}], None),
            "bn_train": ([
                {"text_clean": "bn-a", "label": 0, "source": "ALERT"},
                {"text_clean": "bn-b", "label": 1, "source": "BD_SHS"},
            ], "source"),
            "bn_validation": ([
                {"text_clean": "bn-c", "label": 2, "source": "BenSarc"},
                {"text_clean": "bn-d", "label": 2, "source": "BanglaSarc3"},
            ], "source"),
            "bn_test": ([{"text_clean": "bn-e", "label": 1, "source": "BIDWESH"}], "source"),
        }
        paths = {}
        specs = {}
        for key, (rows, source_column) in definitions.items():
            path = root / f"{key}.csv"
            language = "english" if key.startswith("en_") else "bangla"
            label = "class" if language == "english" else "label"
            columns = ["text_clean", label] + (["source"] if source_column else [])
            self._write(path, columns, rows)
            paths[key] = path
            specs[key] = SplitSpec(language, key.split("_", 1)[1], len(rows), "text_clean", label, source_column)
        hashes = {key: sha256_file(path) for key, path in paths.items()}
        return paths, hashes, specs

    def test_valid_fixture_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, hashes, specs = self._fixture(Path(directory))
            report = validate_canonical_data(paths, hashes, specs=specs)
            self.assertEqual(report["status"], "passed")

    def test_cross_split_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _, specs = self._fixture(root)
            self._write(paths["en_test"], ["text_clean", "class"], [{"text_clean": "en-a", "class": 2}])
            hashes = {key: sha256_file(path) for key, path in paths.items()}
            with self.assertRaisesRegex(DataValidationError, "exact text duplicates"):
                validate_canonical_data(paths, hashes, specs=specs)


if __name__ == "__main__":
    unittest.main()

