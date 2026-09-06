# Stage 1A Report — Corrected Experimental Pipeline

Status: implementation complete; no experiment run

## Outcome

A new, isolated corrected pipeline was added without modifying any historical notebook, checkpoint, prediction, figure, metric, or Stage 0 launcher. Existing Phase 6 artifacts are explicitly marked legacy/preliminary by `Phase 6 (Architectures)/LEGACY_RESULTS_NOTICE.md`. Corrected runs use new `/kaggle/working/corrected_*` directories and refuse to overwrite an existing output directory.

No model/tokenizer was loaded, no checkpoint tensor was deserialized, no inference/training occurred, no dependency was installed, and no network request was made during Stage 1A.

## Implemented controls

### Canonical data validation

`corrected_pipeline/data_validation.py` uses the standard library to enforce:

- `text_clean`; English `class`; Bangla `label`; Bangla `source`;
- exact split counts 83,455/10,432/10,432 and 67,009/8,376/8,377;
- main labels 0/1/2 and the provisional class-2 name `Sarcastic`;
- exactly five released Bangla source values;
- no missing required values or unexpected labels/sources;
- no exact text overlap across train/validation/test;
- LF-normalized canonical SHA-256 hashes for all six files.

The full files are validated before smoke-test subsampling. All failures are collected into one clear `DataValidationError`.

### Cross-platform hash correction (2026-09-06)

Kaggle and the GitHub clone were confirmed to have identical raw CSV bytes, while the earlier expected values came from a Windows working copy whose line endings differed. Validation remains mandatory, but text-file identity is now platform-independent:

1. Read each CSV as bytes.
2. Replace CRLF and standalone CR with LF.
3. Calculate SHA-256 over those normalized bytes.

`canonical_sha256` is the required reproducibility identity used by both `build_hash_manifest()` and `validate_canonical_data()`. `raw_sha256` is optional byte-level provenance: it is recorded and reported when present, but a raw mismatch alone does not fail validation because Git line-ending conversion may legitimately change it. A canonical mismatch still fails before any subset, model, training, or evaluation code can run.

Recorded hashes:

| Split | Required `canonical_sha256` | Optional Windows-working-copy `raw_sha256` |
|---|---|---|
| en_train | `96e1167cbf5bec9265e6a5f688f2e48e153d363cdf072b14f4f0e899e50a3563` | `d1e29e79fb00986ccdbf11a86ee887d9bf6041c917ad5ca9538c3e79c60647e6` |
| en_validation | `cb59ccfbc23adffae38ab2a6c3b58d8f57ab04e3b1f44aa28144a3292d2d32a6` | `665a4aace86224ea7a4ffc581aff5abc6a56203f55361833a637b74e77cad5bd` |
| en_test | `a4647d638471bdee88c3da0d3d0c7110451a2265b89e6a32df9efd4a23bfd3eb` | `c35ca8c097db159af62aa66681538f0b09a4d6de7e837279a3980ed991d2ddd7` |
| bn_train | `5ea874a8d9fe55e7545795770fdac109f3a284bcc2d44be048c7f0df5cb48799` | `79af568f5507437efef1d6b81117a317ccf68d60dbe1fb69e96836ef98930ce8` |
| bn_validation | `1d91c7a27e14c9e1a31efd64cc77e716fdd774fd9e98c7c17de239d064e87297` | `2addabe616c1f184f208e5875aa1aba6cdf0e63f0cbc9ceac03c7d5746bcbb06` |
| bn_test | `c2ac6c268ad94284be0bff8857f9d2f90e68610075bc0c60dac9ff008496df0d` | `b4b3fdeb5999293fd38003aabb6bf4ff63a0e337e91d56aaabc0f84dddabec32` |

All five corrected configurations use this nested hash manifest consistently. Unit tests cover LF/CRLF equivalence, standalone content changes, optional raw hashes, and continued enforcement of schema, row counts, labels, and duplicate detection. The corrected smoke notebook's disabled branch was also repaired so it no longer references `repo` before `find_repo()` assigns it. `RUN_HEAVY=False` remains unchanged.

### Models and training

Four configurable model classes were implemented:

- XLM-R single-task three-class;
- multilingual DistilBERT single-task three-class;
- one-shared-encoder XLM-R multi-task;
- one-shared-encoder multilingual DistilBERT multi-task.

The corrected multi-task model has independent three-class, hate, and sarcasm heads. Main cross entropy uses `labels_main`; hate BCE uses `labels_hate`/`mask_hate`; sarcasm BCE uses `labels_sarcasm`/`mask_sarcasm`. Configured class weights are passed to main cross entropy. The loop uses gradient accumulation, gradient clipping, AdamW, linear warmup, early stopping, and selection strictly by validation macro F1. Each newly selected state is detached, moved to CPU, and cloned before retention.

No corrected class is called “separate encoder.” The historical experiments are described as head ablations.

### Label provenance

`labels.py` distinguishes main-dataset labels, explicitly enabled proxy labels, and independent human-gold axes. Gold hate and sarcasm may overlap and may be missing independently. Missing values receive mask 0 and no auxiliary loss.

Proxy labels cannot be created without `proxy_auxiliary_labels=true`. The only supplied proxy configuration is the guarded smoke test and its status is exactly `SMOKE TEST — NOT A REPORTED RESULT`.

Gold annotations are never merged automatically. The full gold multi-task configurations are valid plans but have `execution.blocked=true`.

### Evaluation artifacts

The reusable evaluator reports accuracy, macro F1, weighted F1, per-class precision/recall/F1/support, confusion matrix, hate F1, sarcasm F1, and hateful-class recall. It writes:

- `metrics.json`;
- `predictions.csv` with class probabilities and optional auxiliary probabilities;
- `run_config.json` including hashes/history;
- `environment.json` with Python/platform/package versions;
- `checkpoint_metadata.json` with selection metric, revisions, provenance, hashes, best epoch/F1;
- a CPU-only checkpoint plus saved tokenizer metadata.

The test split is never passed to training or checkpoint selection. Full configurations evaluate it only after the best validation checkpoint is restored. The smoke test never evaluates it.

### Kaggle guards

`06_corrected_smoke_test.ipynb` defaults to `RUN_HEAVY=False`, checks GPU and all dataset paths, runs compilation/unit tests first, then—only if enabled—uses 96 training rows, 48 validation rows, one epoch, and no test evaluation. It verifies forward shapes, finite combined/main losses, correct head availability, all-missing auxiliary masks, metrics, CPU checkpoint saving, and the checkpoint hash.

`07_corrected_full_reproduction.ipynb` also defaults to false, checks GPU/data/config blocks, and runs local checks before the selected full configuration. It defaults to the multilingual DistilBERT single-task baseline. Gold multi-task plans refuse execution.

## Local validation performed

- Five corrected JSON files parsed successfully.
- Both new notebook JSON files parsed successfully and each has exactly one `RUN_HEAVY = False` assignment.
- Static checks confirmed all four model classes, correct hate/sarcasm target-and-mask wiring, gradient clipping, CPU checkpoint copying, validation-only selection, overwrite protection, seed policy, proxy labeling, and blocked full gold multi-task configurations.
- Lightweight local checks confirmed LF/CRLF canonical equivalence, changed-value sensitivity, and the six LF-normalized canonical dataset hashes. Existing schema, count, label, source, and duplicate controls remain present and are covered by unit tests.
- The smoke notebook was statically checked to ensure `repo` is assigned before use and is not referenced from the disabled branch.
- `run_static_checks.ps1` completed successfully.

Python compilation and unit-test execution were skipped locally because there is no runnable Python interpreter; only inaccessible Windows app aliases are present, and WSL access is denied. No Python/PyTorch installation was attempted. Both new Kaggle notebooks run `compileall` and the full import-independent/tiny-tensor unit-test suite before any experiment command.

## Supervisor confirmation required

1. Final meaning/name of class 2 for each language: `Sarcastic` versus `Implicit Hateful`.
2. Confirmation that hate and sarcasm are independent, potentially overlapping gold axes.
3. Gold annotation rules, missing-value convention, and adjudication policy.
4. A stable ID/hash manifest and one-to-one join protocol for the annotation subsets, which currently lack sample IDs.
5. Whether “implicit hateful” is a separate task, hate subtype, or derived gold-axis intersection.
6. Approval before implementing/running the optional GRL design.
7. An immutable multilingual DistilBERT model/tokenizer revision to replace the explicit but moving `main` revision before any result is treated as final/reportable. XLM-R is pinned to the repository-recorded commit `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.

Until items 1–4 are resolved, full corrected multi-task experiments remain blocked. Corrected single-task baselines and the non-reportable smoke diagnostic are structurally ready for Kaggle, but neither has been run.

## Files created in Stage 1A

### Corrected source

- `corrected_pipeline/__init__.py`
- `corrected_pipeline/README.md`
- `corrected_pipeline/config.py`
- `corrected_pipeline/data_validation.py`
- `corrected_pipeline/evaluation.py`
- `corrected_pipeline/labels.py`
- `corrected_pipeline/loss_wiring.py`
- `corrected_pipeline/models.py`
- `corrected_pipeline/runner.py`
- `corrected_pipeline/training.py`

### Corrected configurations

- `configs/corrected_smoke_mdistilbert_multitask.json`
- `configs/corrected_full_xlmr_single_task.json`
- `configs/corrected_full_mdistilbert_single_task.json`
- `configs/corrected_full_xlmr_multitask_gold.json`
- `configs/corrected_full_mdistilbert_multitask_gold.json`

### Documentation/notices

- `LABEL_RECONCILIATION.md`
- `GRL_DESIGN.md`
- `Phase 6 (Architectures)/LEGACY_RESULTS_NOTICE.md`
- `STAGE1A_REPORT.md`

### Tests

- `tests_stage1a/__init__.py`
- `tests_stage1a/test_config.py`
- `tests_stage1a/test_data_validation.py`
- `tests_stage1a/test_evaluation.py`
- `tests_stage1a/test_labels.py`
- `tests_stage1a/test_loss_wiring.py`
- `tests_stage1a/test_torch_loss.py`
- `tests_stage1a/run_static_checks.ps1`

### New guarded Kaggle notebooks

- `kaggle/06_corrected_smoke_test.ipynb`
- `kaggle/07_corrected_full_reproduction.ipynb`

No historical notebook, historical result, checkpoint, or original dataset was modified.
