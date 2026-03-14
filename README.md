# 🗺️ TripAdvisor Place Recommendation via Information Retrieval

> A review-based place recommendation system that finds similar experiences across restaurants, hotels, and attractions — without relying on structured metadata.

**Course:** DIA4 — ESILV 2025/2026  
**Authors:** Sandrine Daniel · Raphael Marques Araujo

---

## 📋 Table of Contents

- [Overview](#overview)
- [Core Hypothesis](#core-hypothesis)
- [Dataset](#dataset)
- [Models](#models)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)

---

## Overview

This project builds a **content-based place recommendation system** using TripAdvisor user reviews. Given a query place (restaurant, hotel, or attraction), the system retrieves the most similar experiences from the corpus — using only the text of user reviews, with no structured metadata involved.

Four retrieval approaches are implemented and compared, ranging from classical term-frequency models to modern transformer-based embeddings:

| Model | Type |
|-------|------|
| BM25 | Lexical baseline |
| TF-IDF + Cosine Similarity | Lexical vector space |
| Word2Vec | Static word embeddings |
| SBERT | Contextual sentence embeddings |

---

## Core Hypothesis

> *Similar experiences tend to be described in similar ways.*

If two restaurants share the same cuisine style and atmosphere, their reviews should exhibit sufficient lexical and semantic overlap for a retrieval system to relate them — without needing any category labels or structured fields. This paradigm is especially valuable when metadata is absent, incomplete, or unreliable, and it captures experiential dimensions (atmosphere, service, value) that formal tags often miss.

---

## Dataset

Two CSV files form the basis of this project:

- `reviews83325.csv` — one user review per line, with language metadata
- `Tripadvisor.csv` — descriptive information per place (joined via `idplace / id`)

**Preprocessing decisions:**
- Only **English-language reviews** were retained (filtered via the `langue` attribute)
- Review counts vary heavily across places; each place's representation was **normalized** by retaining a fixed number of reviews or words
- This aggregation was guided by **TF-IDF scoring**, keeping only the most discriminative terms per place

---

## Models

### BM25 (Baseline)

BM25 improves on raw term counting via two mechanisms: **term frequency saturation** (repeated words yield diminishing returns) and **document length normalization** (longer documents aren't unfairly favored). It consistently outperforms TF-IDF across both evaluation levels and both document strategies.

### TF-IDF + Cosine Similarity

Each place is represented as a TF-IDF vector over its concatenated reviews. Cosine similarity is then used to rank candidates. Two document construction strategies were tested:

- **S1 — Truncation:** concatenated reviews truncated to a max word count; vocabulary of 20,000 features, bigrams `(1,2)`, min document frequency of 2
- **S2 — Top-K Keywords:** each place represented only by its most discriminative TF-IDF terms; unigrams `(1,1)`, min document frequency of 1

### Word2Vec

A shallow neural network is trained to project words into a dense lower-dimensional space. Place representations are built by averaging word vectors, then compared with cosine similarity:

$$\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

**Strengths:** fast to train, simple, no need for labeled data.  
**Limitations:** static embeddings — the word *"bank"* has the same vector regardless of context; embeddings cluster tightly, reducing discriminative power for fine-grained retrieval.

### SBERT (Sentence Transformers)

SBERT produces dense, **context-aware** sentence embeddings via the scaled dot-product attention mechanism:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

Unlike Word2Vec, the same word receives different representations depending on its surrounding context. This gives SBERT superior performance on fine-grained semantic retrieval tasks.

---

## Repository Structure

```
.
├── src/
│   ├── __init__.py              # Package initialisation
│   ├── bm25.py                  # BM25 retrieval model
│   ├── tfidf.py                 # TF-IDF + cosine similarity
│   ├── word2vec.py              # Word2Vec embeddings
│   ├── sbert.py                 # SBERT sentence embeddings
│   ├── strategies.py            # S1 truncation / S2 Top-K document strategies
│   ├── processing.py            # Data loading, filtering, preprocessing
│   ├── score.py                 # Scoring utilities
│   └── evaluation.py            # L1 / L2 eval score, ranking error
├── notebook.ipynb               # Main experimentation notebook (BM25, TF-IDF)
├── notebook_demo.ipynb          # Demo notebook
├── word2vec_tripadvisor.model   # Trained Word2Vec model
├── requirements.txt             # Python dependencies
├── Trip_Advisor_Report.pdf      # Project report
├── .gitignore
└── README.md
```

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/Debian-code/TripAdvisor_Recommendation_System.git
cd TripAdvisor_Recommendation_System

pip install rank-bm25 scikit-learn gensim sentence-transformers pandas numpy
```

---

## Usage

### Run a specific model

```bash
# BM25 baseline
python src/bm25.py

# TF-IDF
python src/tfidf.py

# Word2Vec
python src/word2vec.py

# SBERT
python src/sbert.py
```

### Evaluate results

```bash
python src/evaluation.py
```

### Run the full pipeline interactively

Open `notebook.ipynb` for BM25 and TF-IDF experiments, or `notebook_demo.ipynb` for a walkthrough demo.

---

## Results

All models were evaluated on 917 queries at two granularity levels:

- **Level 1 (L1):** coarse type matching — Hotel (H), Restaurant (R), Attraction (A), Attraction Product (AP)
- **Level 2 (L2):** fine-grained matching — cuisine style, restaurant type, attraction sub-category, hotel price range

### BM25 and TF-IDF

| Model | Strategy | L1 Mean Error ↓ | L2 Mean Error ↓ |
|-------|----------|----------------|----------------|
| BM25 | S1 Truncation | **0.638** | 7.052 |
| BM25 | S2 Top-K | 0.699 | **5.000** |
| TF-IDF | S1 Truncation | 0.974 | 8.860 |
| TF-IDF | S2 Top-K | 0.810 | 5.909 |

### Word2Vec and SBERT

| Model | Level | Eval Score ↑ | Ranking Error ↓ | Mean Cos. Sim. |
|-------|-------|-------------|----------------|----------------|
| Word2Vec | L1 | 0.8637 | 1.024 | 0.9719 |
| Word2Vec | L2 | 0.8092 | 2.261 | 0.9719 |
| **SBERT** | **L1** | **0.8680** | 1.092 | 0.7216 |
| **SBERT** | **L2** | **0.8222** | 3.432 | 0.7216 |

---

## Key Findings

- **BM25 > TF-IDF** across the board. Its term frequency saturation and length normalization make it more robust to document noise than raw TF-IDF.
- **Top-K keyword strategy consistently beats raw truncation** for TF-IDF, confirming that removing noisy terms helps fine-grained category matching.
- **Embedding models (Word2Vec, SBERT) significantly outperform lexical models** at L1 (~86–87% top-1 accuracy), confirming the value of semantic representations.
- **SBERT achieves the best eval score at both L1 and L2**, thanks to its contextual attention-based embeddings. However, its lower mean cosine similarity (0.72 vs. 0.97 for Word2Vec) shows it produces a more spread-out vector space — meaning when it gets it wrong, the correct match is ranked further down.
- **Word2Vec embeddings cluster too tightly** (mean cosine sim = 0.97), reducing discriminative power for fine-grained retrieval at L2.

---

## References

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)
- Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (2019)

---

## License

This project is for academic purposes (ESILV DIA4 course, 2025/2026).
