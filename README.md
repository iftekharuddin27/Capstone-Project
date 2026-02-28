# Capstone-Project

# Unmasking Sarcastic Hate: A Sarcasm-Aware Framework for Implicit Hate Speech Detection in Bangla and English

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

### As social media platforms implement stricter moderation policies for explicit abuse, toxic behavior frequently evolves into covert, implicit forms of aggression. This project introduces a hybrid Natural Language Processing (NLP) framework designed to automatically detect and classify Implicit Hate Speech—specifically hateful intent concealed behind sarcasm, irony, semantic reversal, and fake politeness—across Bangla, English, and code-mixed ("Banglish") discourse.

### This repository contains the dataset preprocessing scripts, model architectures, and evaluation notebooks developed for this Capstone research project.


## 🎯 Objective
Standard hate speech detection models excel at identifying explicit slurs and direct abuse but frequently fail to recognize implicit hate. For example, a sarcastically polite comment targeting a specific community may bypass standard toxicity filters. This project bridges the gap between Sarcasm Detection and Hate Speech Classification to create a more robust moderation tool.


## 🗂️ Task Definition
The framework formulates the detection task as a 3-Class Classification Problem:


Non-Hate (Class 0): Neutral, positive, or purely general sarcastic statements without malicious intent.


Explicit Hate (Class 1): Direct abuse, slurs, threats, or aggressive language.


Implicit/Sarcastic Hate (Class 2): Hateful intent masked by positive vocabulary, contextual irony, or sarcasm.


## 🏗️ Methodology & Architecture
Baseline CNN: A surface-level model utilizing 1D Convolutions and FastText embeddings to capture local $n$-gram patterns.

Transformer Encoders: Contextual models (BanglaBERT, mBERT, XLM-RoBERTa) to capture long-range dependencies and semantic reversal.

Proposed Hybrid Feature Fusion: A dual-branch architecture. One transformer branch is optimized for general sarcasm detection, while the other is optimized for explicit hate. The contextual embeddings from both branches are concatenated and passed through a Multilayer Perceptron (MLP) to accurately classify the intersection: Sarcastic Hate.


## 📊 Datasets
To train the models, we aggregate and synthesize data from diverse, established NLP corpora to capture a wide spectrum of online discourse.

Establishing the Human Baseline and Evaluation Set
Because implicit hate relies heavily on pragmatics, world knowledge, and cultural context, evaluating the model strictly against automated labels is insufficient. To ensure rigorous evaluation, we establish a Human Baseline through a structured manual annotation process:

Candidate Sampling: A diverse subset of highly ambiguous, potentially sarcastic, and potentially hateful comments is extracted from the broader dataset.

The 2-Axis Annotation Task: Multiple independent human annotators evaluate each comment blindly based on two distinct axes:

Axis A (Hateful Intent): Does this text intend to insult, demean, or attack an individual or group? (Yes/No)

Axis B (Sarcastic Expression): Does the text use irony or semantic reversal to convey a meaning opposite to its literal interpretation? (Yes/No)

Label Mapping & Agreement: The combination of these two axes determines the final ground-truth class. For instance, a text marked Yes for Intent and Yes for Sarcasm maps to Implicit/Sarcastic Hate. We calculate Fleiss' Kappa to measure inter-annotator agreement, ensuring the inherent subjectivity of sarcasm is statistically accounted for.

The Human Baseline: The annotators' consensus serves as the ultimate benchmark. By testing our machine learning models against this rigorously annotated evaluation set, we can directly compare the AI's performance to human comprehension levels, determining exactly where the model succeeds or fails in understanding pragmatic nuance.

---

```
⚙️ Repository Structure
├── data/
│   ├── raw/                  # Original downloaded datasets 
│   ├── processed/            # Cleaned and merged training data
│   └── human_annotated/      # The manually labeled evaluation subset
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_human_annotation_kappa.ipynb
│   ├── 03_cnn_baseline.ipynb
│   ├── 04_transformer_models.ipynb
│   └── 05_hybrid_feature_fusion.ipynb
├── src/                      # Reusable Python scripts for training/evaluation
├── requirements.txt          # Project dependencies
└── README.md
```
