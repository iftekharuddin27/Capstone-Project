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
- exact SHA-256 hashes for all six files.

The full files are validated before smoke-test subsampling. All failures are collected into one clear `DataValidationError`.

Recorded SHA-256 values:

| Split | SHA-256 |
|---|---|
| en_train | `d1e29e79fb00986ccdbf11a86ee887d9bf6041c917ad5ca9538c3e79c60647e6` |
| en_validation | `665a4aace86224ea7a4ffc581aff5abc6a56203f55361833a637b74e77cad5bd` |
| en_test | `c35ca8c097db159af62aa66681538f0b09a4d6de7e837279a3980ed991d2ddd7` |
| bn_train | `79af568f5507437efef1d6b81117a317ccf68d60dbe1fb69e96836ef98930ce8` |
| bn_validation | `2addabe616c1f184f208e5875aa1aba6cdf0e63f0cbc9ceac03c7d5746bcbb06` |
| bn_test | `b4b3fdeb5999293fd38003aabb6bf4ff63a0e337e91d56aaabc0f84dddabec32` |

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
- Previously computed local checks confirmed the six canonical SHA-256 hashes and zero exact text overlap across each language's splits.
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

No pre-existing file was modified in Stage 1A.
