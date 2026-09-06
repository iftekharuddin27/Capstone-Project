# Corrected pipeline

This package implements the Stage 1A pipeline without changing historical notebooks or results.

- `data_validation.py` validates canonical schemas, exact counts, labels, five Bangla sources, missing values, cross-split duplicates, and LF-normalized `canonical_sha256` hashes. Optional byte-exact `raw_sha256` values are diagnostic and never replace canonical validation.
- `labels.py` keeps main, proxy, and human-gold provenance explicit and supports missing gold axes.
- `models.py` implements four configurable one-encoder models.
- `training.py` applies the correct loss targets, masks, class weights, gradient clipping, validation-macro-F1 selection, and CPU checkpoint copies.
- `evaluation.py` writes metrics, predictions, configuration, environment, and checkpoint metadata.
- `runner.py` is the Kaggle-only heavy entry point.
- Stage 1B pilots scope validation to the audited repository-relative Bangla train/validation files. Test evaluation is rejected and `dataset_hashes.json` is saved beside the validation report.

Do not invoke `runner.py` locally. Use the guarded notebooks in `kaggle/`.
