<div align="center">

# GRIS
### **Grammatical Interpretable Translation Scoring**

*An interpretable dependency-based metric for Machine Translation Evaluation.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-NLP-red.svg)]()
[![Machine Translation](https://img.shields.io/badge/Task-MT%20Evaluation-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

</div>

---

## Overview

GRIS (**Grammatical Interpretable Translation Scoring**) is an interpretable Machine Translation evaluation framework that measures translation quality by comparing the grammatical structures of the reference and candidate translations.

Unlike traditional lexical metrics (e.g., BLEU) or purely neural black-box metrics, GRIS combines **dependency tree matching**, **syntactic similarity**, and **semantic similarity** to produce transparent and explainable evaluation scores.

The framework is designed for researchers interested in:

- Machine Translation Evaluation
- Explainable AI
- Natural Language Processing
- Dependency Parsing
- Semantic Similarity
- Structural Language Analysis

---

# Motivation

Machine Translation evaluation remains a challenging task.

Traditional metrics primarily rely on surface-level lexical overlap, while modern neural metrics often provide excellent performance but limited interpretability.

GRIS bridges this gap by introducing a scoring framework that:

- preserves grammatical structure
- explains why a translation receives a particular score
- combines structural and semantic similarity
- produces interpretable evaluation outputs

---

# Key Features

- Dependency-based translation evaluation
- Universal Dependencies parsing
- Structural Tree Edit Distance
- Semantic similarity using Sentence Transformers
- Explainable scoring pipeline
- Modular evaluation framework
- Easy Python API
- Interactive dashboard support

---

# Methodology

The evaluation pipeline consists of the following stages:

```text
Reference Sentence
        │
        ▼
Dependency Parsing
        │
        ▼
Dependency Tree Construction
        │
        ▼
Tree Matching
        │
        ▼
Structural Similarity
        │
        ▼
Embedding Similarity
        │
        ▼
Weighted Score Aggregation
        │
        ▼
Final GRIS Score
```

---

# Repository Structure

```
GRIS
│
├── src/
│   └── gris/
│       ├── parser.py
│       ├── scorer.py
│       ├── matcher.py
│       ├── embedder.py
│       ├── Evaluate.py
│
├── examples/
│
├── tests/
│
├── docs/
│
├── dashboard_corrected.py
│
├── pyproject.toml
│
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/GRIS.git

cd GRIS
```

Install the package

```bash
pip install .
```

or

```bash
pip install -e .
```

---

# Quick Start

```python
from gris import Evaluate

reference = "The cat is sleeping."

candidate = "A cat sleeps."

score = Evaluate(
    reference,
    candidate
)

print(score)
```

---

# Evaluation Components

GRIS combines multiple sources of information:

| Component | Purpose |
|-----------|---------|
| Dependency Parsing | Extract grammatical structure |
| Tree Matching | Compare syntactic similarity |
| Structural Similarity | Measure grammatical correspondence |
| Sentence Embeddings | Capture semantic similarity |
| Weighted Aggregation | Produce final interpretable score |

---

# Technologies

### NLP

- Stanza
- spaCy
- Sentence Transformers
- NLTK
- Gensim

### Machine Learning

- PyTorch
- Scikit-learn

### Data Science

- NumPy
- Pandas
- Matplotlib

---

# Example Output

```
Reference:
The cat is sleeping.

Candidate:
A cat sleeps.

Dependency Similarity : 0.91

Semantic Similarity : 0.95

Final GRIS Score : 0.93
```

---

# Research Applications

GRIS can be applied to:

- Machine Translation Evaluation
- NLP Benchmarking
- Explainable AI
- Structural Similarity Analysis
- Educational NLP
- Linguistic Research

---

# Future Work

- Multilingual evaluation
- Cross-lingual embeddings
- Constituency parsing
- Large Language Model integration
- WMT benchmark evaluation
- Hugging Face integration
- PyPI release

---

# Citation

If you use this work in your research, please cite:

```bibtex
@software{lumbini2026gris,
  title={GRIS: Grammatical Interpretable Translation Scoring},
  author={Chalitha Lumbini},
  year={2026},
  url={https://github.com/YOUR_USERNAME/GRIS}
}
```

---

# Author

**Chalitha Lumbini**

MSc in Statistical Data Analytics (Distinction)

Tampere University

Research Interests:

- Natural Language Processing
- Machine Translation
- Machine Translation Evaluation
- Explainable AI
- Statistical Machine Learning

---

# License

This project is released under the MIT License.
