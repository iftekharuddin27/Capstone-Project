# Experiment Audit

Audit date: 2026-09-05  
Scope: recovery from checked-in notebook source, stored outputs, JSON configs, and reported-result CSVs; no experiment was run

## Canonical data and labels

The project-wide three-class mapping is:

| ID | Canonical meaning |
|---:|---|
| 0 | Non-hateful |
| 1 | Hateful |
| 2 | Sarcastic |

English uses label column `class`; Bangla uses `label`. Both use `text_clean`. Bangla additionally carries `source`.

Phase 5 introduces a semantic conflict: its Bangla annotation analysis names IDs `0/1/2` as `Not Hateful / Explicit Hateful / Implicit Hateful`, whereas training and every Phase 3/6 classifier name ID 2 `Sarcastic`. Those meanings are not interchangeable and must be reconciled before using the annotated subset as training or test gold data.

### Dataset construction and splits

Stored Phase 1 outputs describe raw English as 104,737 rows and raw Bangla as 83,992 rows. Phase 2 removes three empty rows from each language and then removes cleaned-text duplicates, leaving:

| Language | Full clean | Train | Validation | Test | Label column |
|---|---:|---:|---:|---:|---|
| English | 104,319 | 83,455 | 10,432 | 10,432 | `class` |
| Bangla | 83,762 | 67,009 | 8,376 | 8,377 | `label` |

The split is stratified 80/10/10 with `random_state=42`, implemented as 80/20 followed by a 50/50 split of the temporary subset.

### Four-versus-five Bangla source investigation

The checked-in evidence supports **five**, not four, Bangla sources.

| Source | Raw count in Phase 1 output | Count in checked-in cleaned CSV | Labels present after cleaning |
|---|---:|---:|---|
| ALERT | 1,006 | 1,004 | 0 only |
| BD_SHS | 50,281 | 50,199 | 0, 1 |
| BenSarc | 25,636 | 25,616 | 0, 2 |
| BanglaSarc3 | 4,008 | 3,957 | 2 only |
| BIDWESH | 3,061 | 2,986 | 0, 1 |

The five cleaned counts sum to 83,762 exactly. Neither notebook source nor stored output contains a four-source mapping. A four-dataset statement elsewhere is therefore most likely an omission—possibly excluding the small, single-class ALERT source—or it counts only a subset of contributing corpora. That interpretation is an inference, not documented provenance. Until upstream collection records prove otherwise, report “five source values in the released combined dataset” and list them explicitly.

This also confirms severe source-label confounding: sarcastic samples occur only in BenSarc/BanglaSarc3, while hateful samples occur only in BD_SHS/BIDWESH. ALERT is exclusively non-hateful.

## Are auxiliary labels gold?

For all Phase 6 training, auxiliary labels are **derived proxies**, not independent gold labels:

```text
is_hateful   = 1 if main three-class label == 1 else 0
is_sarcastic = 1 if main three-class label == 2 else 0
```

Thus the hate and sarcasm heads do not add new supervision; they deterministically restate the same mutually exclusive target. They also cannot represent a sample that is simultaneously hateful and sarcastic.

Phase 4/5 has a separately annotated 999-row subset per language. Its binary axes are derived from human-entered `hate_type` and `sarcasm_type` fields and are the only candidate gold auxiliary labels in the repository. The checked-in Phase 6 training notebooks never load those annotation files.

## GRL and domain mapping

No gradient reversal layer (GRL), autograd reversal function, domain classifier, domain loss, domain label, or source-to-domain mapping exists in any checked-in notebook or Python source. Therefore the GRL source-domain mapping is **not recoverable because it was never implemented here**. The only available source vocabulary is the five-valued Bangla `source` field above. Inventing a numeric mapping would create a new experiment rather than reproduce an existing one.

## Recovered baseline configurations

### Classical baseline

The final test-oriented notebook uses word TF-IDF (1–3 grams, 50,000 features), character TF-IDF (English 3–5 grams; Bangla 3–6 grams; 30,000 features), and standardized engineered features. Logistic regression uses `C=5`, SAGA, `max_iter=2000`, seed 42, and balanced weights for Bangla only. The primary metric is macro F1; accuracy, per-class precision/recall/F1, reports, and confusion matrices are also generated.

The older classical notebook is materially different: its first model uses only 5,000 word plus 5,000 character features and Liblinear, while a later six-value `C` search uses SAGA. Its hyperparameter search and voting ensemble belong on Kaggle under the current resource rules.

### Word-level neural baselines

LSTM, BiLSTM, and CNN use a 30,000-token whitespace vocabulary, maximum length 128, learned 128-dimensional embeddings, dropout 0.3, and three output classes. LSTM/BiLSTM use hidden size 128 and two recurrent layers. CNN uses 100 filters each for widths 3, 4, and 5. Training uses Adam at `1e-3`, cross entropy, gradient clipping at 1.0, 10 epochs, train batch 128, and validation/test batch 256. Bangla uses computed balanced class weights `[0.67442656, 1.09080106, 1.66527498]`. No scheduler is used.

The final `ML and DL Test.ipynb` explicitly seeds Python, NumPy, and PyTorch with 42 and restores the best validation-macro-F1 BiLSTM state. The older deep-learning notebook does not explicitly seed and evaluates `best_preds` but returns the final model state, so it cannot reliably reproduce its own best prediction result.

### Single-task transformers

| Experiment | Hugging Face model | Epochs | Train batch | Max length | Bangla class weights |
|---|---|---:|---:|---:|---|
| English BERT | `bert-base-uncased` | 3 | 32 | 128 | no |
| English DistilBERT | `distilbert-base-uncased` | 3 | 64 | 128 | no |
| BanglaBERT | `csebuetnlp/banglabert` | 3 | 32 | 128 | yes |
| Bangla mBERT | `bert-base-multilingual-cased` | 3 | 32 | 128 | yes |

Both transformer training notebooks set 500 warmup steps, weight decay 0.01, evaluation/save each epoch, FP16 when CUDA is present, and best-model selection by validation loss. Learning rate and seed are omitted from the notebook and therefore resolve to the `TrainingArguments` defaults (historically `5e-5` and 42). The exact optimizer is also an unrecorded, version-dependent Trainer default; this is a reproducibility gap. The scheduler resolves to the default linear schedule.

## Recovered Phase 6 configurations

| Experiment | Backbone | Epochs | Train batch | LR | Optimizer/schedule | Loss weights |
|---|---|---:|---:|---:|---|---|
| Shared dual-head | `xlm-roberta-base` | 5 | 32 | 2e-5 | AdamW, wd 0.01, linear, 10% warmup | hate 0.3, sarcasm 0.3, 3-class 0.4 |
| Lightweight English | `distilbert-base-uncased` | 5 | 64 | 3e-5 | AdamW, wd 0.01, linear, 10% warmup | intended 0.3/0.3/0.4 |
| Lightweight Bangla | `distilbert-base-multilingual-cased` | 5 | 64 | 3e-5 | AdamW, wd 0.01, linear, 10% warmup | intended 0.3/0.3/0.4 |
| Ablations | `xlm-roberta-base` | 5 | 64 | 2e-5 | AdamW, wd 0.01, linear, 10% warmup | active BCE heads 0.3 each; 3-class 0.4 |
| EN→BN few-shot | `xlm-roberta-base` | 3 | 16 | 1e-5 | AdamW, wd 0.01, linear, no warmup | 0.3/0.3/0.4 |

All use maximum length 128 and seed 42. Shared/lightweight validation batches are twice the train batch; test and zero-shot inference use 64. Few-shot Bangla samples 10% independently within each class using seed 42.

### Architecture and implementation discrepancies

- The directories named `xlm-roberta-base (separate encoder)` do not implement separate hate/sarcasm encoders. Each ablation model creates one `AutoModel` encoder and optional heads. They are single-encoder head ablations.
- In `lightweight-model.ipynb`, the written loss is `ALPHA*BCE(hl, ls) + BETA*BCE(sl, ls) + GAMMA*CE(...)`. The hate logit `hl` is incorrectly trained against the sarcasm label `ls`; `lh` is read but unused. This affects the reported lightweight results.
- The lightweight notebook computes Bangla class weights but never passes them into cross entropy.
- Fixing either lightweight issue creates a new result. Exact historical replication should preserve the bug and label it; a corrected experiment should use a new output directory and must not overwrite reported artifacts.
- Best states are cloned on-device in several Phase 6 loops. This can materially increase GPU memory use.

## Recovered reported test results

These are historical stored outputs/CSVs, not results produced by this audit.

| Model | English macro F1 | Bangla macro F1 |
|---|---:|---:|
| Logistic regression | 0.7920 | 0.7583 |
| BiLSTM | 0.8241 | 0.7616 |
| BERT / BanglaBERT | 0.9029 | 0.8200 |
| DistilBERT / mBERT | 0.8937 | 0.7739 |
| Shared dual-head XLM-R | 0.8983 | 0.8151 |
| Lightweight dual-head | 0.8927 | 0.7919 |

Ablation macro F1 is 0.8976/0.8097 without the hate head, 0.8962/0.8055 without the sarcasm head, and 0.8993/0.8092 for single-task English/Bangla respectively. Transfer results report EN→BN zero-shot 0.2633, BN→EN zero-shot 0.3346, and EN plus 10% Bangla few-shot 0.7332. External OLID evaluation reports 0.5814 over the two classes present.

## Evaluation interpretation risks

- Results mix validation and test metrics. The master CSV explicitly says the SVM value is validation-only, while the dedicated transformer test notebook supplies transformer test scores.
- The master CSV's `EN_Sarc_F1`/`BN_Sarc_F1` values are not consistently traceable to a single generated artifact for every model.
- The external notebook compares a dual-head XLM-R result against a hard-coded “own EN test F1” of 0.9029 that belongs to the BERT baseline, not the dual-head model. The stated drift gap is therefore not model-matched.
- No model revision/commit hash, dataset version hash, exact dependency lock, deterministic worker configuration, or GPU type beyond notebook metadata is recorded.
- Source leakage and the deterministic proxy heads weaken causal claims that the architecture learned sarcasm-aware hate detection.

## Reproduction artifacts created in Stage 0

- `requirements.txt`: minimal direct experiment dependencies;
- `configs/datasets.json`: paths, schemas, split, counts, source values, label provenance;
- `configs/baselines.json`: classical, word-level neural, and single-task transformer settings;
- `configs/phase6.json`: shared/lightweight/ablation/transfer/external settings and GRL status;
- `kaggle/*.ipynb`: guarded launchers for all heavy notebook groups.

No experiment has been run and no historical result has been altered.
