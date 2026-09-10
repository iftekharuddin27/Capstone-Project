# Stage 2 Reportable Validation Plan

## Scope

Stage 2 is a validation-only, clean single-task three-class comparison for English and Bangla using mDistilBERT and XLM-R. It contains 12 predeclared runs: 2 languages × 2 architectures × 3 seeds. Class `2` is Sarcastic. No auxiliary labels, proxy labels, GRL, LOSO, or test data are used.

Locked settings are: seeds `42`, `123`, `2026`; maximum length 128; at most 5 epochs; patience 2; selection by validation macro F1; AdamW; weight decay 0.01; warm-up ratio 0.10; dropout 0.30; gradient clipping 1.0. mDistilBERT uses learning rate 3e-5 and batch 32 × accumulation 2. XLM-R uses learning rate 2e-5 and batch 16 × accumulation 4. Both therefore have effective batch size 64. Bangla runs use the audited class weights `[0.67442656, 1.09080106, 1.66527498]`.

## Run matrix

| Language | Architecture | Seeds | Kaggle launcher |
|---|---|---|---|
| English | mDistilBERT | 42, 123, 2026 | `kaggle/10_stage2_en_mdistilbert.ipynb` |
| English | XLM-R | 42, 123, 2026 | `kaggle/11_stage2_en_xlmr.ipynb` |
| Bangla | mDistilBERT | 42, 123, 2026 | `kaggle/12_stage2_bn_mdistilbert.ipynb` |
| Bangla | XLM-R | 42, 123, 2026 | `kaggle/13_stage2_bn_xlmr.ipynb` |

The earlier seed-42 runs must be repeated. They are explicitly pilot—not reportable—outputs, and the XLM-R pilot used a different effective batch size. Reusing them would break both status and controlled-comparison requirements.

## Kaggle execution procedure

1. Upload or Git-clone this corrected branch in Kaggle; attach the audited dataset only if the repository copy is unavailable.
2. If independently verifying the mDistilBERT pin, run `kaggle/09_resolve_mdistilbert_revision.ipynb`. It queries metadata only and never loads weights.
3. Choose one launcher and one allowed `SEED_TO_RUN`. Leave `RUN_HEAVY=False` while reviewing its printed preflight information.
4. Enable a Kaggle GPU and set `RUN_HEAVY=True` for exactly one run. Compilation, unit tests, config validation, GPU checking, and safety assertions execute before the runner.
5. Download the run's entire unique output directory: validation metrics and predictions, training history, environment, run configuration, dataset validation/hashes, tokenizer files, checkpoint metadata, and CPU checkpoint. Preserve the Kaggle notebook log as provenance.
6. Repeat seeds in the fixed order 42, 123, 2026 within each language/model cell of the matrix; then advance English mDistilBERT → English XLM-R → Bangla mDistilBERT → Bangla XLM-R. Never reuse an existing output directory.

For each language/model combination, summarize validation macro F1, accuracy, weighted F1, per-class precision/recall/F1, and Sarcastic-class F1 across three seeds using arithmetic mean and sample standard deviation. Retain individual-seed results and do not select a seed for test evaluation.

## Locks and later stages

The test split remains locked because test feedback during architecture, hyperparameter, or seed decisions would turn it into validation data and bias the final estimate. These configs contain no test path and `evaluate_test=false`; the runner adds a second validation-only guard.

Before gold multi-task work, provide an immutable one-to-one annotation ID mapping, adjudicated normalization codebook, and human-annotation provenance. Before GRL/LOSO work, freeze the five-source mapping, define folds and leakage controls, and create separately reviewed configs/tests. Supervisor approval permits that future design; it does not authorize execution in Stage 2.

The source-domain mapping is already made explicit in code for that future review: `ALERT → 0`, `BD_SHS → 1`, `BenSarc → 2`, `BanglaSarc3 → 3`, and `BIDWESH → 4`. No GRL or LOSO execution is implemented by the Stage 2 launchers.
