# Environment Audit

Audit date: 2026-09-05  
Scope: Stage 0 static inspection only

## Safety boundary used for this audit

No notebook cell, training loop, inference loop, evaluation, model import, or package installation was run. No network request was made. No Hugging Face model or dataset was downloaded. The `.pt`, `.pth`, `.bin`, `.safetensors`, `.zip`, and `.npy` artifacts were not deserialized. Inspection was limited to notebook JSON/source/output text, small JSON/CSV metadata, file names/sizes, and a single grouped read of the 83,762-row cleaned Bangla CSV to verify source names and counts.

The original notebooks, saved outputs, checkpoints, figures, predictions, and reported-result CSVs were not changed.

## Repository inventory

- 17 Jupyter notebooks were found and inspected, including their code/markdown sources and stored textual/error outputs.
- The 17 originals contain 147 code cells, no markdown cells, and 354 stored output records. Rich image payloads were inventoried through their output records and generating code; they were not recomputed.
- All notebook metadata reports Python `3.12.12`.
- 13 notebooks record a Kaggle Tesla T4 GPU environment; four analysis/preprocessing notebooks record no accelerator.
- Notebook metadata uses Kaggle image IDs `31328` and `31329`, so runs were not all captured under one image.
- The repository includes a separate FastAPI deployment environment at `hate-sarcasm-extension/backend/requirements.txt`. It is not merged into the new experiment requirements because it serves a different runtime.

### Notebook inspection ledger

| Notebook | Cells | Stored outputs | Stored errors | Role/status |
|---|---:|---:|---:|---|
| `Phase 1/capstone-work.ipynb` | 12 | 14 | 0 | EDA and source analysis |
| `Phase 2/Preprocessed Data.ipynb` | 9 | 8 | 0 | cleaning and deterministic splits |
| `Phase 3/classical ML Models/Classical Models.ipynb` | 7 | 26 | 0 | classical validation experiments/search |
| `Phase 3/Deep Learning Models/Deep Learnind Models.ipynb` | 10 | 17 | 0 | LSTM/BiLSTM/CNN validation lineage |
| `Phase 3/ML and DL Test.ipynb` | 10 | 24 | 0 | LR/BiLSTM final-test lineage |
| English transformer training notebook | 10 | 38 | 0 | BERT/DistilBERT training |
| Bangla transformer training notebook | 13 | 40 | 0 | BanglaBERT/mBERT training |
| `Transformer model test results.ipynb` | 9 | 27 | 0 | transformer test evaluation |
| `Phase 5/Data evaluation.ipynb` | 14 | 31 | 0 | annotation/error analysis |
| shared XLM-R notebook | 7 | 21 | 0 | primary dual-head training/test |
| lightweight notebook | 8 | 24 | 0 | DistilBERT dual-head training/test |
| English ablation notebook | 6 | 18 | 0 | three English head ablations |
| Bangla ablation notebook | 6 | 18 | 0 | three Bangla head ablations |
| transfer-learning notebook | 7 | 20 | 0 | zero/few-shot transfer |
| completed external-test lineage | 6 | 13 | 0 | OLID inference completed |
| lowercase external-test duplicate | 6 | 9 | 1 | failed checkpoint load |
| final-comparison notebook | 7 | 6 | 0 | hard-coded/report aggregation |

### Large artifacts deliberately not loaded

The largest checkpoint artifacts are approximately 678.49 MiB (mBERT), 515.76 MiB (lightweight Bangla ZIP), 422.00 MiB (BanglaBERT), 417.67 MiB (English BERT), 255.43 MiB (English DistilBERT), and 254.95 MiB (lightweight English). Tokenizer JSON files reach about 16.96 MiB. These files should remain on disk or in Kaggle input storage until a GPU run is explicitly authorized.

## Minimal experiment environment

The new root `requirements.txt` contains only direct third-party packages imported by the notebooks or required by Hugging Face `Trainer`:

- data/numerics: NumPy, pandas, SciPy;
- classical evaluation/modeling: scikit-learn;
- plots: Matplotlib, seaborn, wordcloud;
- heavy execution: PyTorch, Transformers, Datasets, Accelerate.

Standard-library imports (`os`, `re`, `gc`, `json`, `random`, `warnings`, `unicodedata`, `collections`, `pathlib`, and similar modules) are intentionally omitted. Jupyter is also omitted because Kaggle supplies the notebook runtime. The requirements file is an environment declaration, not authorization to install these packages locally.

The complete notebook-level third-party import set is `numpy`, `pandas`, `scipy`, `sklearn`, `matplotlib`, `seaborn`, `wordcloud`, `torch`, `transformers`, and `datasets`. `accelerate` is not directly imported but is required by the Transformers Trainer runtime. The backend separately imports `fastapi`, `pydantic`, `pytest`, and `httpx`, with Uvicorn and multipart support declared in its existing requirements file.

Version provenance is incomplete. Saved transformer configs say `transformers_version: 5.0.0`, while the deployment backend pins `transformers==4.44.2` and `torch==2.4.1`. The experiment requirements therefore use bounded compatibility ranges instead of claiming a nonexistent exact lock. For exact reruns, export `pip freeze` from the first successful Kaggle environment and preserve it as a lock file.

## Data paths

The preprocessing notebooks expect raw Kaggle files at:

- `/kaggle/input/datasets/iftekharuddin27/capstone-datasets/English_combined_dataset.csv`
- `/kaggle/input/datasets/iftekharuddin27/capstone-datasets/bangla_hate_pool.csv`

Training notebooks expect preprocessed splits under `/kaggle/input/datasets/iftekharuddin27/preprocessed-datasets`. Local checked-in equivalents are under `Phase 2/English data` and `Phase 2/Bangla data`.

Other hard-coded Kaggle inputs are:

- annotations: `/kaggle/input/datasets/iftekharuddin27/annotated-data`;
- Phase 3 transformer checkpoints: `/kaggle/input/datasets/iftekharuddin27/transformer-models`;
- main Phase 6 checkpoints: `/kaggle/input/datasets/iftekharuddin27/transformer-learning`;
- external OLID data: `/kaggle/input/datasets/faisalshanto/external-hate-dataset`;
- one stale duplicate external notebook instead uses `/kaggle/input/datasets/iftekharuddin27/duelhead-models`.

All generated files are intended for `/kaggle/working`.

Machine-readable mappings are in `configs/datasets.json`, `configs/baselines.json`, and `configs/phase6.json`.

## Notebook/output integrity findings

- `External Hate Dataset Testing/external-dataset-test.ipynb` contains a stored `UnpicklingError: invalid load key, 'v'.` in cell 3. The similarly named `External Hate Dataset Testing/External hate Dataset test.ipynb` records a completed run and uses a different model-input directory.
- Several stored outputs show Hugging Face downloads of hundreds of MiB to more than 1 GiB. They are historical outputs only; nothing was downloaded during this audit.
- The classical ensemble output contains scikit-learn convergence warnings for SAGA and Liblinear.
- Hugging Face load reports contain unexpected language-model-head keys. This is plausible when loading a base encoder into a custom head, but the exact library/model revisions were not recorded.
- Most notebooks contain only code cells and few or no explanatory markdown cells. Execution order is therefore part of the undocumented state.
- The transformer training notebooks select the best checkpoint using `eval_loss`, while the rest of the project generally selects/report models using validation macro F1.

## Kaggle preparation

Five guarded launcher notebooks are provided under `kaggle/`. They do not execute anything while `RUN_HEAVY = False`. Each launcher copies the selected original notebook to `/kaggle/working` and executes the copy with `nbconvert`, preserving the checked-in source notebook.

Before a future run:

1. Add this repository as a Kaggle dataset or clone it into `/kaggle/working`.
2. Attach the required datasets/checkpoints at the hard-coded paths above, or deliberately update paths in a new Kaggle working copy.
3. Enable a T4 GPU for neural/transformer/Phase 6 launchers.
4. Open only the launcher for the requested experiment, review `SOURCE_NOTEBOOKS`, set `RUN_HEAVY = True`, and run it.
5. Download the executed notebook and `/kaggle/working` artifacts after completion.

No Stage 1 launcher has been run.
