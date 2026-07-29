<div align="center">


# GRIS

### Grammatical Interpretable Translation Scoring

*An interpretable dependency-based framework for Machine Translation Evaluation.*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Research](https://img.shields.io/badge/NLP-Research-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

## 📖 Overview

GRIS (Grammatical Interpretable Translation Scoring) is a research framework for evaluating machine translation quality through dependency-based grammatical analysis and semantic similarity. The framework combines syntactic structure and semantic representations to produce transparent and interpretable evaluation scores.

---

## 🎯 Why GRIS?

Machine Translation (MT) evaluation is a fundamental task in Natural Language Processing. Traditional evaluation metrics such as BLEU primarily rely on lexical overlap, while recent neural metrics often provide limited interpretability.

GRIS addresses this challenge by combining **dependency-based grammatical analysis** with **semantic similarity**, producing evaluation scores that are both accurate and explainable.

### Key Contributions

- 🌳 Dependency tree-based structural evaluation
- 🧠 Semantic similarity using sentence embeddings
- 🔍 Explainable scoring instead of black-box evaluation
- 📊 Linguistically informed translation assessment
- 🧩 Modular and extensible evaluation framework

---

## 🏗️ Architecture

GRIS evaluates machine translation quality through a multi-stage pipeline that integrates grammatical structure with semantic similarity.

```text
                 Reference Translation
                          │
                          ▼
                 Dependency Parsing
                          │
                          ▼
                 Dependency Tree
                          │
                          │
Candidate Translation ─► Dependency Parsing
                          │
                          ▼
                 Dependency Tree
                          │
                          ▼
                 Tree Matching
                          │
                          ▼
              Structural Similarity Score
                          │
                          ▼
            Sentence Embedding Similarity
                          │
                          ▼
               Weighted Score Aggregation
                          │
                          ▼
                    Final GRIS Score
```

The framework consists of five major components:

| Component | Description |
|-----------|-------------|
| **Dependency Parser** | Extracts grammatical structures from the reference and candidate translations. |
| **Tree Matcher** | Aligns dependency trees to measure structural similarity. |
| **Structural Scorer** | Computes similarity based on dependency relationships. |
| **Semantic Similarity** | Measures sentence-level meaning using sentence embeddings. |
| **Score Aggregation** | Combines structural and semantic information into the final GRIS score. |


## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/chalithalumbini/GRIS.git
cd GRIS
```

### Install dependencies

```bash
pip install -e .
```

or

```bash
pip install .
```

---

## 🚀 Quick Start

```python
from gris import Evaluate

reference = "The cat is sleeping on the mat."
candidate = "A cat sleeps on the mat."

score = Evaluate(
    reference=reference,
    hypothesis=candidate
)

print(score)
```

GRIS returns an interpretable evaluation score by combining dependency-based structural similarity with semantic similarity.

## 📂 Repository Structure

```text
GRIS/
├── src/
│   └── gris/
│       ├── parser.py
│       ├── matcher.py
│       ├── scorer.py
│       ├── embedder.py
│       └── Evaluate.py
├── examples/
├── docs/
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

## 🔬 Research Applications

GRIS is designed for researchers and practitioners working in Natural Language Processing and Machine Translation. Potential applications include:

- 🌍 Machine Translation Evaluation
- 🧠 Explainable AI for NLP
- 🌳 Dependency-based Linguistic Analysis
- 📊 Evaluation Metric Benchmarking
- 🔍 Semantic Similarity Analysis
- 📚 Academic Research in Computational Linguistics

The framework is modular and can be extended to support additional languages, parsing frameworks, and evaluation strategies.

## 🛠️ Technologies

### Programming Languages

- Python

### Natural Language Processing

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

### Development

- Git
- GitHub

## 🚀 Future Work

Future development of GRIS will focus on:

- 🌐 Multilingual machine translation evaluation
- 🤖 Integration with Large Language Models (LLMs)
- 📈 Benchmarking on WMT datasets
- 🔍 Enhanced explainability and visualization
- ⚡ Performance optimization for large-scale evaluation
- 📦 Public release on PyPI


## 📖 Citation

If you use GRIS in your research, please cite this repository:

```bibtex
@software{lumbini2026gris,
  title={GRIS: Grammatical Interpretable Translation Scoring},
  author={Chalitha Lumbini},
  year={2026},
  url={https://github.com/chalithalumbini/GRIS}
}
```

## 👨‍💻 Author

**Chalitha Lumbini**

MSc in Statistical Data Analytics (Distinction)  
Tampere University, Finland

**Research Interests**

- Natural Language Processing
- Machine Translation
- Explainable AI
- Statistical Machine Learning
- Trustworthy AI

📧 Email: chalitha.lumbini@gmail.com
💼 LinkedIn: https://www.linkedin.com/in/chalitha-lumbini-085269170/
