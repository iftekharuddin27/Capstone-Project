# Annotation Audit

## Scope and method

The repository-wide source search for `hate_type` or `sarcasm_type` identified two data files and three documentation/notebook references. The candidate CSVs were inspected read-only. Counts below preserve raw spelling and whitespace. Matching to canonical splits used exact text equality only; no fuzzy matching or data modification was performed.

Candidate data files:

1. `Phase 4/English Annotation - English_balanced_1k_cleaned.csv`
2. `Phase 4/Bangla Annotation - bangla_hate_speech_1k.csv`

Other files containing the column names are `Phase 5/Data evaluation.ipynb`, `EXPERIMENT_AUDIT.md`, and `LABEL_RECONCILIATION.md`; they are not annotation tables.

## English candidate

- Rows: **999**
- Columns: `text`, `class`, `sarcasm_type`, `hate_type`
- Missing `hate_type`: **0**; missing `sarcasm_type`: **0**
- Exact duplicate full rows: **0**
- Duplicate text: **1 group / 1 excess row**. The duplicated sentence has conflicting main labels (`2` with `Humor`; `1` with `Sarcasm`), while both hate values are `None`.
- Duplicate-ID count: **not applicable**; no ID-like column exists.
- Possible stable join identifier: **none**.

Raw `hate_type` values:

| Value | Count |
|---|---:|
| `non_hateful` | 236 |
| `non hateful ` | 187 |
| `Abusive` | 121 |
| `None` | 112 |
| `Offensive` | 108 |
| `Non-hateful` | 105 |
| `offensive ` | 57 |
| `abusive ` | 28 |
| `Hate` | 20 |
| `hate speech ` | 17 |
| `non hateful` | 7 |
| `Not-hateful` | 1 |

Raw `sarcasm_type` values:

| Value | Count |
|---|---:|
| `None` | 681 |
| `Sarcasm` | 142 |
| `Humor` | 86 |
| `sarcasm ` | 63 |
| `Irony` | 20 |
| `metaphor` | 3 |
| `Metaphore` | 2 |
| `humor ` | 1 |
| `irony ` | 1 |

Exact matches against canonical `text_clean`: train **168**, validation **17**, test **20**, any split **205**, unmatched **794**. Of these, main labels agree/disagree by split as train **166/2**, validation **17/0**, and test **20/0**.

Under the conservative binary normalization used for the overlap table, `hate_type` disagrees with `class == 1` on **43** rows and `sarcasm_type` disagrees with `class == 2` on **35** rows. These discrepancies reinforce that the columns are not simple main-label copies, but they require adjudication before training.

## Bangla candidate

- Rows: **999**
- Columns: `text`, `label`, `label_name`, `source`, `sarcasm_type`, `hate_type`
- Missing `hate_type`: **0**; missing `sarcasm_type`: **0**
- Exact duplicate full rows: **0**; duplicate text rows: **0**
- Duplicate-ID count: **not applicable**; no ID-like column exists.
- Possible stable join identifier: **none**. `source` is a domain category, not a unique row key.

Raw `hate_type` values:

| Value | Count |
|---|---:|
| `non hateful ` | 220 |
| `non_hateful` | 190 |
| `Abusive` | 148 |
| `Non-hateful` | 110 |
| `None` | 107 |
| `Offensive` | 98 |
| `offensive ` | 54 |
| `abusive ` | 38 |
| `Hate` | 15 |
| `hate speech ` | 12 |
| `not_hateful` | 5 |
| `no` | 1 |
| `Hateful` | 1 |

Raw `sarcasm_type` values:

| Value | Count |
|---|---:|
| `None` | 673 |
| `Sarcasm` | 134 |
| `Humor` | 75 |
| `sarcasm ` | 62 |
| `Irony` | 34 |
| `humor ` | 9 |
| `none ` | 4 |
| `irony ` | 3 |
| `Metaphore` | 3 |
| `metaphor` | 1 |
| `metaphor ` | 1 |

Exact matches against canonical `text_clean`: train **389**, validation **44**, test **37**, any split **470**, unmatched **529**. All matched main labels agree.

Under the conservative binary normalization, `hate_type` disagrees with `label == 1` on **37** rows and `sarcasm_type` disagrees with `label == 2` on **19** rows.

## Axis overlap

For this diagnostic only, case/whitespace/punctuation spelling variants were conservatively grouped: hate-positive included abusive/offensive/hate terms, and sarcasm-positive included sarcasm/humor/irony/metaphor terms. This normalization was not written to the source data.

| Language | Hate− / Sarcasm− | Hate− / Sarcasm+ | Hate+ / Sarcasm− | Hate+ / Sarcasm+ |
|---|---:|---:|---:|---:|
| English | 338 | 310 | 343 | 8 |
| Bangla | 328 | 305 | 349 | 17 |

The positive/positive cells confirm that the two axes can overlap. Spelling variants (`Metaphore`, whitespace variants, several forms of non-hateful) require an explicit, reviewed normalization codebook before any gold-label use.

## Provenance and join decision

The supervisor confirms these are human annotation columns. Their subtype vocabulary, positive overlaps, and disagreements with one-vs-rest main labels provide evidence that they are not simple `class == 1` / `class == 2` proxies. The Phase 5 notebook normalizes the subtype strings and derives binary axes from those human-entered columns; that transformation is a label-processing step, not evidence that the annotations were generated from the main label. The repository nevertheless lacks annotator identifiers, instructions, inter-annotator agreement, adjudication records, and immutable sample IDs, so provenance and normalization should be recorded before a reportable release.

**Safe stable join: NO. Gold multi-task execution is blocked.** Exact text is incomplete, text is mutable, and the English duplicate is ambiguous. A safe release requires a unique immutable ID present one-to-one in each annotation row and its originating canonical row, plus an adjudicated label codebook and provenance record. Test annotations must remain inaccessible during model selection even after such a mapping exists.
