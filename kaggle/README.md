# Kaggle launchers

These Stage 0 notebooks are execution guards around the original experiment notebooks. They do not train or evaluate anything unless `RUN_HEAVY` is changed from `False` to `True` on Kaggle.

| Launcher | Heavy work covered |
|---|---|
| `01_classical_baselines.ipynb` | Classical models, C search, ensemble, oversampling |
| `02_neural_baselines.ipynb` | LSTM, BiLSTM, CNN, and the later test-oriented baseline notebook |
| `03_transformer_baselines.ipynb` | English/Bangla transformer training and transformer test evaluation |
| `04_phase6_architectures.ipynb` | Shared dual-head, lightweight dual-head, English/Bangla head ablations |
| `05_transfer_and_external.ipynb` | Zero/few-shot transfer and external OLID inference |

Upload or attach the complete `Capstone-Project` directory, attach the required Kaggle datasets/checkpoints, enable a T4 GPU for launchers 02–05, and run only the explicitly selected source notebook(s). Each launcher copies source notebooks into `/kaggle/working` before execution, so the originals remain unchanged.

Review `ENVIRONMENT_AUDIT.md` and `EXPERIMENT_AUDIT.md` first. In particular, decide whether a run is intended to reproduce the historical lightweight bug exactly or create a separately named corrected experiment.

## Corrected protocol launchers

Notebooks `06`–`08` cover the corrected smoke/full and Stage 1B pilot paths. Stage 2 adds:

| Launcher | Purpose |
|---|---|
| `09_resolve_mdistilbert_revision.ipynb` | Guarded metadata-only verification of the immutable mDistilBERT SHA; no model load |
| `10_stage2_en_mdistilbert.ipynb` | English mDistilBERT, one selected reportable seed |
| `11_stage2_en_xlmr.ipynb` | English XLM-R, one selected reportable seed |
| `12_stage2_bn_mdistilbert.ipynb` | Bangla mDistilBERT, one selected reportable seed |
| `13_stage2_bn_xlmr.ipynb` | Bangla XLM-R, one selected reportable seed |

All remain `RUN_HEAVY=False` by default. The Stage 2 launchers accept only seeds 42, 123, and 2026, contain no test paths, and run compilation/unit tests before their single training invocation. Follow `STAGE2_REPORTABLE_PLAN.md` and download each unique output directory in full.
