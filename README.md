# Unmasking Sarcastic Hate: A Sarcasm-Aware Framework for Implicit Hate Speech Detection in Bangla and English

### Author:<br>
Name: KM Iftekhar Uddin<br>
ID:2022-2-60-034

Name: Najmul Haque Majumder<br>
ID:2022-2-60-059

Name: Nelufa Yeasmin<br>
ID:2020-1-60-236

Name: Omar Faisal Shanto<br>
ID:2022-2-60-125


**Institution:** East West University  

**Course:** CSE400 –  Capstone Project

**Supervisor:** Ahmed Abdal Shafi Rasel (AASR)

---

## 📋 Project Overview

This thesis investigates the intersection of **sarcasm** and **implicit hate speech detection** across two languages: **English** and **Bangla**. The core insight is that sarcasm is frequently used as a delivery mechanism for hate speech—rendering conventional detectors ineffective.

### Research Questions

1. **RQ1:** To what extent does explicit sarcasm modeling (via a dedicated sarcasm branch) improve implicit hate speech detection compared to a single end-to-end hate classifier?

2. **RQ2:** What types of sarcastic hate expressions (semantic reversal, rhetorical questions, cultural irony) are most frequently misclassified, and do failure patterns differ between Bangla and English?

3. **RQ3:** How does Bangla-English code-mixing degrade detection performance, and can domain-adaptive pre-training mitigate this?

---

## 🎯 Key Contribution

A **multi-task learning framework** with:
- **Shared encoder** (XLM-RoBERTa or language-specific BERT variants)
- **Dedicated sarcasm branch** (auxiliary task)
- **Dedicated hate branch** (primary task)
- **Fusion mechanism** → final 3-class output (Non-hateful, Hateful, Sarcastic)

This architecture explicitly models sarcasm as a linguistic phenomenon, not just a data class, enabling better capture of implicit hate expressions.

---

## 📊 Datasets

### English Dataset: `English_combined_dataset.csv`
| Metric | Value |
|--------|-------|
| **Total texts** | 104,737 |
| **Non-hateful (Class 0)** | 34,794 (33.2%) |
| **Hateful (Class 1)** | 35,102 (33.5%) |
| **Sarcastic (Class 2)** | 34,841 (33.3%) |
| **Balance** | ✅ Perfectly balanced |
| **Avg text length** | 94 characters |
| **Text source** | Mixed (tweets + news headlines) |

**Characteristics:**
- Sarcastic texts: news headlines (short, no @mentions, no hashtags)
- Hateful/Non-hateful texts: tweets (contain @mentions, hashtags, URLs)
- This creates potential dataset artifacts for model confusion

### Bangla Dataset: `bangla_hate_pool.csv`
| Metric | Value |
|--------|-------|
| **Total texts** | 83,992 |
| **Non-hateful (Label 0)** | 41,497 (49.4%) |
| **Hateful (Label 1)** | 25,669 (30.6%) |
| **Sarcastic (Label 2)** | 16,826 (20.0%) |
| **Balance** | ⚠️ Imbalanced (ratio 2.5:1.5:1) |
| **Avg text length** | 73 characters |

**Source Breakdown:**
| Source | Count | Label Distribution | Issue |
|--------|-------|-------------------|-------|
| BD_SHS | 50,281 | Non-hateful (26K) + Hateful (24K) | ❌ NO sarcasm |
| BenSarc | 25,636 | Non-hateful (12.8K) + Sarcastic (12.8K) | ❌ NO hateful |
| BanglaSarc3 | 4,008 | ONLY sarcastic | ❌ Single class |
| BIDWESH | 3,061 | Non-hateful (1.5K) + Hateful (1.5K) | ❌ NO sarcasm |
| ALERT | 1,006 | ONLY non-hateful | ❌ Single class |

**Critical Problem:** **Source leakage** — sarcasm and hate rarely co-occur in the same source, creating a structural bias that models can exploit without learning true sarcasm detection.

---

## 🛠️ Project Phases

### Phase 1: Data Understanding & EDA 
- [x] Class distribution analysis
- [x] Text length statistics
- [x] Word cloud visualizations
- [x] Bangla source-to-label dependency mapping
- [x] Manual sample review per class


---

### Phase 2: Preprocessing & Dataset Splitting 
- [x] URL, @mention, hashtag removal
- [x] Extra whitespace cleanup
- [x] English lowercasing
- [x] Bangla Unicode normalization
- [x] Stop word preservation (critical for sarcasm)
- [x] Stratified 80/10/10 train/val/test split
- [x] CSV export

---

### Phase 3: Baseline Models 

**Models:**
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Stochastic Gradient Descent (SGD)
- Voting Ensemble

**Feature engineering:**
- Word-level TF-IDF (unigrams + bigrams + trigrams)
- Character-level TF-IDF (3-grams + 4-grams + 5-grams)
- 15 engineered features:
  - Text length, word count, avg word length
  - Capitalization ratio, punctuation count
  - Sentiment polarity (positive/negative/neutral)
  - Subjectivity score
  - Readability metrics (Flesch-Kincaid, Dale-Chall)
  - Named entity density
  - Stop word ratio
  - Unique word ratio
  - Digit ratio

**Results on 10% data:**
| Language | Macro F1 | Precision | Recall | Weighted F1 |
|----------|----------|-----------|--------|-------------|
| English | 0.757 | 0.759 | 0.757 | 0.762 |
| Bangla | 0.719 | 0.721 | 0.719 | 0.722 |

**Expected results on full data:** English F1 ~0.82+, Bangla ~0.75+


#### Deep Learning 
**Planned models:**
- LSTM with attention
- Bidirectional LSTM (BiLSTM)
- Convolutional Neural Network (CNN)
- Hybrid LSTM-CNN

**Features:**
- Word embeddings (Word2Vec, GloVe, FastText)
- Bangla word embeddings (BnWord2Vec)
- Embedding dimension: 300
- Sequence length: padded to 100 characters

**Imbalance handling:**
- Class weights (inverse frequency)
- Oversampling (SMOTE for Bangla)
- Focal loss (if needed)


---

#### Transformer Models 
**Planned models:**
- **Multilingual:** XLM-RoBERTa, mBERT
- **English-specific:** RoBERTa, DistilBERT, BERT-base
- **Bangla-specific:** BanglaBERT, sahajBERT, BanglaBERTBase

**Approach:**
- Fine-tune pre-trained models on the hate speech detection task
- Freeze early layers, train final layers + classifier head
- Experiment with different learning rates (2e-5, 3e-5, 5e-5)
- Early stopping on validation Macro F1


---

####  Class Imbalance Mitigation
**Techniques implemented:**
- Class weights in loss function
- Oversampling minority classes
- SMOTE (for Bangla)


---

### Phase 4: Two-Dimensional Annotation  
**Selection strategy:**
- ~2,500 texts per dataset
- 500 random texts per class
- 500 most confidently misclassified texts (by Phase 3 models)
- 500 texts with highest label uncertainty

**Annotation axes:**
1. **Hateful intent (binary):** Does this text express hate toward a protected group?
2. **Sarcastic expression (binary):** Does this text use sarcasm to convey meaning?

**Sarcastic hate sub-types:**
- **Semantic reversal:** "Oh sure, [group] are TOTALLY [hateful stereotype]"
- **Rhetorical question:** "Aren't [group] just the worst?"
- **Cultural irony:** References to stereotypes with inverted sentiment

**Evaluation:**
- 3 independent annotators
- Fleiss' Kappa (inter-annotator agreement)
- Majority voting for label assignment


---

### Phase 5: Error Analysis 
**Systematic error categorization:**
- False positives (benign text misclassified as hateful)
- False negatives (hateful text missed)
- Sarcasm-specific errors (sarcasm not detected as sarcasm)
- Language-specific errors (comparison between English and Bangla)

**Label noise analysis:**
- Confident-but-wrong predictions
- Uncertain predictions (entropy > threshold)
- Ambiguous texts (annotators disagree)

**Visualization:**
- BERT embeddings + t-SNE
- Confusion matrices by error type
- Misclassification word clouds


---

### Phase 6: Sarcasm-Aware Framework 

#### Architecture Overview (Primary)
```
Input Text
    ↓
Shared Encoder (XLM-RoBERTa or language-specific BERT)
    ↓
┌─────────────────────────┬─────────────────────────┐
│                         │                         │
│   Sarcasm Head          │   Hate Head             │
│   (Auxiliary task)      │   (Auxiliary task)      │
│                         │                         │
│   2-layer MLP           │   2-layer MLP           │
│   Output: Binary        │   Output: Binary        │
│   (Sarcastic/Not)       │   (Hateful/Not)         │
│                         │                         │
└──────────┬──────────────┴────────────┬────────────┘
           │                          │
           └──────────┬───────────────┘
                      ↓
            Fusion (concatenate penultimate
            hidden states from both heads)
                      ↓
            2-layer MLP (shared)
                      ↓
            Output: 3-class (Non-hateful, Hateful, Sarcastic)
```

#### Variants to Explore
1. **Shared encoder + dedicated heads** (primary)
2. **Separate encoders** (one for hate, one for sarcasm) → fusion
3. **Lightweight alternative** (DistilBERT or BiLSTM + same structure)

#### Multi-task Loss
```
Total Loss = λ₁ * Loss_hate + λ₂ * Loss_sarcasm + λ₃ * Loss_3class
where λ₁ + λ₂ + λ₃ = 1.0 (hyperparameters to tune)
```

#### Domain Adaptation (RQ3)
- Masked language modeling on Bangla-English code-mixed text
- Vocabulary expansion for code-mixed tokens
- Continued pre-training on in-domain unlabeled data


---

### Phase 7: Human Baseline & Fairness Analysis 

**Human annotation:**
- 200–300 texts from test set
- 3 independent annotators
- Compute inter-annotator agreement
- Measure human accuracy on hate/sarcasm detection

**Fairness analysis:**
- Performance breakdown by religious group (Muslim, Christian, Hindu, Jewish, Buddhist, etc.)
- Performance breakdown by racial/ethnic group
- Identify disparate impact across groups
- Attention visualization (which words drive predictions?)


---

### Phase 8: Final Evaluation & Thesis Writing 
- Comprehensive evaluation across all model architectures
- Error comparison: Phase 3 baselines vs Phase 6 sarcasm-aware framework
- Human vs model performance comparison
- Limitations and future work
- Full thesis document (target: ~40–50 pages)

---

##  Quick Start

### Prerequisites
```
Python >= 3.8
pandas, numpy, scikit-learn, matplotlib, seaborn
torch, transformers
nltk, Bangla
jupyter
```

### Installation
```bash
git clone https://github.com/capstone-thesis/hate-sarcasm-detection.git
cd hate-sarcasm-detection

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Phase 1: EDA**
```bash
jupyter notebook notebooks/Phase_1_EDA.ipynb
```

**Phase 2: Preprocessing**
```bash
jupyter notebook notebooks/Phase_2_Preprocessing.ipynb
```

**Phase 3A: Classical ML**
```bash
jupyter notebook notebooks/Phase_3A_Classical_ML.ipynb
```

---

## 📁 Project Structure

```
hate-sarcasm-detection/
├── README.md (this file)
├── requirements.txt
├── LICENSE
├── data/
│   ├── raw/
│   │   ├── English_combined_dataset.csv
│   │   └── bangla_hate_pool.csv
│   └── processed/
│       ├── en_train.csv
│       ├── en_val.csv
│       ├── en_test.csv
│       ├── bn_train.csv
│       ├── bn_val.csv
│       └── bn_test.csv
├── notebooks/
│   ├── Phase_1_EDA.ipynb
│   ├── Phase_2_Preprocessing.ipynb
│   ├── Phase_3A_Classical_ML.ipynb
│   ├── Phase_3B_Deep_Learning.ipynb (TBD)
│   ├── Phase_3C_Transformers.ipynb (TBD)
│   ├── Phase_5_Error_Analysis.ipynb (TBD)
│   └── Phase_6_Sarcasm_Framework.ipynb (TBD)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py (Phase 2 code)
│   ├── feature_engineering.py (Phase 3A features)
│   ├── models.py (Phase 3B–3C model definitions)
│   ├── utils.py (helper functions)
│   └── config.py (hyperparameters)
├── results/
│   ├── Phase_3A_baseline_results.json
│   └── (other model outputs TBD)
├── docs/
│   ├── annotation_guidelines.md (Phase 4 TBD)
│   ├── error_taxonomy.md (Phase 5 TBD)
│   └── thesis_final.pdf (Phase 8 TBD)
└── .gitignore
```

---

##  Key Findings (So Far)

### Phase 1: EDA Insights
- English dataset is perfectly balanced but shows mixed text sources (tweets + headlines)
- Bangla dataset is imbalanced (50% non-hateful, 31% hateful, 20% sarcastic)
- **Critical:** Bangla sarcasm is confined to 2 sources (BenSarc + BanglaSarc3); hate is from other sources → source leakage risk

### Phase 2: Preprocessing Insights
- Removing URLs, @mentions, hashtags from tweets reduces information loss for hate signals
- Stop word preservation is essential for sarcasm detection (e.g., "surely," "obviously," "absolutely")
- Unicode normalization important for Bangla consistency

### Phase 3A: Baseline Performance
- **English:** Classical ML achieves 0.757 Macro F1 (on 10% data)
- **Bangla:** Classical ML achieves 0.719 Macro F1 (on 10% data) — hampered by imbalance
- Sarcastic class slightly easier in English (0.73 F1) than Bangla (0.64 F1)

---

##  Known Issues & Limitations


1. **Bangla Source Leakage:** Sarcasm and hate never co-occur in the same source. Models may exploit this shortcut instead of learning true sarcasm detection.

2. **English Dataset Artifacts:** Sarcastic texts are news headlines (different distribution than hateful/non-hateful tweets). Creates potential for distribution-based classification.

3. **Imbalanced Bangla:** Class imbalance (2.5:1.5:1 ratio) makes minority sarcasm class harder to learn. Mitigation strategies needed.

4. **Sarcasm Definition Ambiguity:** Without human annotation, hard to know if "sarcasm" labels truly represent sarcasm or just non-hateful expressions.

---

##  Related Work & References

(To be populated with literature review)

### Hate Speech Detection
- [Reference 1]
- [Reference 2]

### Sarcasm Detection
- [Reference 1]
- [Reference 2]

### Bangla NLP
- [Reference 1]
- [Reference 2]

### Multi-task Learning
- [Reference 1]
- [Reference 2]

---

