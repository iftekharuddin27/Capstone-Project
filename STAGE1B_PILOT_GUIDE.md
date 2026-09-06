# Stage 1B Bangla Seed-42 Single-Task Pilot Guide

Status: prepared for Kaggle; no training or model loading has been run locally

Every artifact from these runs must retain the label **PILOT — NOT FINAL TEST RESULT**. These pilots do not produce a final test result and must not be merged into historical result tables.

## Pilot scope

The launcher runs one clean single-task, three-class Bangla classifier at a time:

| `MODEL_TO_RUN` | Base model | Configuration | Output directory |
|---|---|---|---|
| `mdistilbert` | `distilbert-base-multilingual-cased` | `configs/pilot_bn_mdistilbert_single_seed42.json` | `/kaggle/working/corrected_pilot_bn_mdistilbert_single_seed42` |
| `xlmr` | `xlm-roberta-base` | `configs/pilot_bn_xlmr_single_seed42.json` | `/kaggle/working/corrected_pilot_bn_xlmr_single_seed42` |

Both configurations use seed 42, maximum sequence length 128, five maximum epochs, early-stopping patience 2, AdamW, linear warmup, gradient clipping, and the audited Bangla class weights. Multilingual DistilBERT uses learning rate `3e-5`, batch size 32, and accumulation 2. XLM-R uses learning rate `2e-5`, batch size 16, and accumulation 2.

There is no multi-task, proxy-label, human-gold auxiliary, or GRL path in either configuration. Main-task loss weight is 1.0; hate and sarcasm auxiliary weights are 0.0.

## Dataset boundary

Only these repository-relative audited files are present in the pilot configurations:

- `Phase 2/Bangla data/bn_train.csv` — all 67,009 rows;
- `Phase 2/Bangla data/bn_val.csv` — all 8,376 rows.

The test path and test hash are intentionally absent. `execution.evaluate_test` is `false`, and the pilot configuration validator rejects any Stage 1B configuration that includes another split. The runner resolves paths relative to the repository root, validates only the two declared files, and never opens a test CSV.

Cross-platform validation uses required LF-normalized `canonical_sha256` values. Optional `raw_sha256` values remain diagnostic. Schema, exact counts, labels, the five Bangla source values, missing values, canonical hashes, and exact train/validation duplicates are checked before pandas, PyTorch, Transformers, or model construction is reached.

## Kaggle procedure

1. Upload or attach the complete `Capstone-Project` repository, including the two audited Bangla CSVs.
2. Open `kaggle/08_bn_single_task_seed42_pilot.ipynb` in a Kaggle GPU session. Internet access is needed only if Kaggle does not already have the selected base model cached.
3. Leave `MODEL_TO_RUN` as `mdistilbert`, or change it to `xlmr`. Those are the only accepted values.
4. Confirm `RUN_HEAVY = False` while reviewing paths and settings.
5. Set `RUN_HEAVY = True` only on Kaggle and run the notebook once. It compiles the corrected source and runs the complete lightweight unit-test suite before checking the GPU or invoking the runner.
6. Download the selected model's output directory after completion.
7. To run the other model, use a separate Kaggle run/session and select the other value. The launcher never loops over both models.

The notebook contains no package-install command. Do not install or change package versions during a pilot without recording and reviewing that change.

## Produced validation artifacts

The selected output directory contains `data_validation.json`, a standalone `dataset_hashes.json`, and a `bangla/` directory. The latter contains the CPU checkpoint and tokenizer metadata plus `validation/` with:

- `metrics.json`: accuracy, macro/weighted F1, per-class precision/recall/F1/support, confusion matrix, and `sarcasm_f1` computed as main class 2 versus the other classes;
- `predictions.csv`: validation row IDs, true/predicted labels, three class probabilities, and the pilot status;
- `run_config.json`: the immutable selected configuration plus training history and active language;
- `environment.json`: Python, platform, and relevant package versions;
- `checkpoint_metadata.json`: selection metric, best epoch, best validation macro F1, model/tokenizer revisions, checkpoint SHA-256, auxiliary-label status, and dataset hash manifests;
- `best_checkpoint_cpu.pt` and saved tokenizer files in the parent `bangla/` directory.

Checkpoint selection is exclusively by validation macro F1. The reported `sarcasm_f1` operationally treats main label 2 as sarcastic; its final semantic name remains subject to the existing supervisor-confirmation note.

Output directories are distinct and `allow_overwrite=false`. If an output directory already exists, execution stops instead of replacing it.
