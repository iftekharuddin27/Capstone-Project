# Label Reconciliation

Status: supervisor confirmation required before any corrected full multi-task run

## The conflict

The training pipeline consistently defines the main three-class target as:

| ID | Training meaning |
|---:|---|
| 0 | Non-hateful |
| 1 | Hateful |
| 2 | Sarcastic |

The Bangla annotation analysis instead displays the same IDs as:

| ID | Annotation-analysis meaning |
|---:|---|
| 0 | Not Hateful |
| 1 | Explicit Hateful |
| 2 | Implicit Hateful |

Consequently, class 2 currently has two incompatible meanings: a linguistic form (`Sarcastic`) and a hate-intent category (`Implicit Hateful`). Existing Phase 3/6 numbers remain historical results under the first mapping. They must not be relabeled after the fact.

## Sarcasm and hate are not mutually exclusive

Sarcasm describes how meaning is expressed; hate describes intent or target harm. A text can be sarcastic and hateful, sarcastic and non-hateful, literal and hateful, or neither. Encoding both as mutually exclusive values in one three-class label prevents these combinations and makes the legacy proxy axes deterministic restatements of the main label.

The corrected pipeline therefore treats the two auxiliary axes as independent binary labels with separate missing-label masks. It never derives them unless an experiment explicitly sets `proxy_auxiliary_labels=true`, and proxy runs must be labeled diagnostic rather than evidence of independent multi-task supervision.

## Recommended design

Preserve two layers of labels:

1. **Legacy main task:** keep the released three-class label unchanged (`0=Non-hateful`, `1=Hateful`, `2=Sarcastic`) so historical baselines can be compared honestly. The word “Sarcastic” remains provisional in new prose until confirmed.
2. **Gold two-axis task:** store independently annotated `is_hateful_gold` and `is_sarcastic_gold` columns, each in `{0, 1, missing}`. A missing value has mask 0 and contributes no auxiliary loss. The axes may both equal 1.

Any future “implicit hateful” target should be a separately named field with a written derivation or annotation guideline. It must not silently replace legacy class 2.

## Safe annotation integration

The checked-in annotation CSVs contain no stable row identifier. English has `text,class,sarcasm_type,hate_type`; Bangla has `text,label,label_name,source,sarcasm_type,hate_type`. Text-only joins are unsafe because preprocessing, Unicode normalization, duplicates, and spelling changes can create false or ambiguous matches.

The corrected pipeline does not merge these files. Before integration, create and approve a manifest containing a stable sample ID, canonical split, source dataset/version, original text hash, cleaned text hash, annotator-derived axes, and adjudication status. Validate one-to-one cardinality and ensure no sample crosses train/validation/test boundaries.

## Experiments blocked pending confirmation

- Full XLM-R multi-task training using gold auxiliary axes.
- Full multilingual DistilBERT multi-task training using gold auxiliary axes.
- Any experiment or thesis claim that equates class 2 with implicit hate.
- Any comparison claiming the proxy-head experiments use independent sarcasm/hate supervision.
- Any automatic annotation-to-training merge.
- Any final per-axis gold evaluation until the sample join and missing-label policy are approved.

The corrected single-task baselines are not blocked by this decision because they preserve the released main labels. The guarded proxy-label smoke test is also unblocked because it is solely a wiring diagnostic and is explicitly marked “SMOKE TEST — NOT A REPORTED RESULT.”

## Decisions requested from the supervisor

1. Confirm the intended semantics and name of main class 2 for English and Bangla.
2. Confirm that hate and sarcasm are independent, potentially overlapping axes.
3. Approve the gold-label definitions, missing-value convention, and adjudication rule.
4. Approve a stable sample-ID/join manifest for the 999-row annotation subsets.
5. Decide whether future implicit-hate evaluation is a separate binary task, a subtype of hate, or a derived intersection such as `is_hateful_gold AND is_sarcastic_gold`.

