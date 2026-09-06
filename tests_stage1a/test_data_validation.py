import csv
import tempfile
import unittest
from pathlib import Path

from corrected_pipeline.data_validation import (
    build_hash_manifest,
    canonical_sha256_file,
    DataValidationError,
    SplitSpec,
    raw_sha256_file,
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
                {"text_clean": "bn-f", "label": 2, "source": "BIDWESH"},
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
        hashes = build_hash_manifest(paths)
        return paths, hashes, specs

    def test_lf_and_crlf_have_same_canonical_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.csv"
            crlf = root / "crlf.csv"
            cr = root / "cr.csv"
            lf.write_bytes(b"text_clean,class\na,0\nb,1\n")
            crlf.write_bytes(b"text_clean,class\r\na,0\r\nb,1\r\n")
            cr.write_bytes(b"text_clean,class\ra,0\rb,1\r")
            self.assertEqual(canonical_sha256_file(lf), canonical_sha256_file(crlf))
            self.assertEqual(canonical_sha256_file(lf), canonical_sha256_file(cr))
            self.assertNotEqual(raw_sha256_file(lf), raw_sha256_file(crlf))
            manifest = build_hash_manifest({"lf": lf}, include_raw_sha256=False)
            self.assertEqual(set(manifest["lf"]), {"canonical_sha256"})

    def test_validation_uses_canonical_hash_and_raw_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, hashes, specs = self._fixture(Path(directory))
            target = paths["en_train"]
            original_raw = hashes["en_train"]["raw_sha256"]
            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
            self.assertNotEqual(raw_sha256_file(target), original_raw)
            report = validate_canonical_data(paths, hashes, specs=specs)
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["splits"]["en_train"]["raw_sha256_matches_manifest"])

    def test_changed_value_changes_canonical_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.csv"
            changed = root / "changed.csv"
            original.write_bytes(b"text_clean,class\na,0\n")
            changed.write_bytes(b"text_clean,class\na,1\n")
            self.assertNotEqual(canonical_sha256_file(original), canonical_sha256_file(changed))

    def test_valid_fixture_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, hashes, specs = self._fixture(Path(directory))
            report = validate_canonical_data(paths, hashes, specs=specs)
            self.assertEqual(report["status"], "passed")

    def test_bangla_train_validation_only_scope_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            all_paths, _, all_specs = self._fixture(Path(directory))
            keys = ("bn_train", "bn_validation")
            paths = {key: all_paths[key] for key in keys}
            specs = {key: all_specs[key] for key in keys}
            report = validate_canonical_data(paths, build_hash_manifest(paths), specs=specs)
            self.assertEqual(set(report["splits"]), set(keys))

    def test_cross_split_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _, specs = self._fixture(root)
            self._write(paths["en_test"], ["text_clean", "class"], [{"text_clean": "en-a", "class": 2}])
            hashes = {
                key: {"canonical_sha256": canonical_sha256_file(path)}
                for key, path in paths.items()
            }
            with self.assertRaisesRegex(DataValidationError, "exact text duplicates"):
                validate_canonical_data(paths, hashes, specs=specs)

    def test_schema_count_and_label_checks_remain_active(self):
        cases = (
            ("schema", "en_train", ["wrong_text", "class"], [{"wrong_text": "x", "class": 0}], "missing required columns"),
            ("count", "en_train", ["text_clean", "class"], [], "expected 1 rows"),
            ("label", "en_train", ["text_clean", "class"], [{"text_clean": "x", "class": 9}], "unexpected labels"),
        )
        for name, split, columns, rows, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths, _, specs = self._fixture(root)
                self._write(paths[split], columns, rows)
                hashes = {
                    key: {"canonical_sha256": canonical_sha256_file(path)}
                    for key, path in paths.items()
                }
                with self.assertRaisesRegex(DataValidationError, message):
                    validate_canonical_data(paths, hashes, specs=specs)


if __name__ == "__main__":
    unittest.main()
