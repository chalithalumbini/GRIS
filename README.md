# GRIS — Interpretable MT Evaluation

Two complementary metrics:
- **GRIS-DepScore** — dependency-structure matching (Hungarian algorithm).
- **GRIS-SynGram** — subtree path n-gram matching.

## Install

```bash
pip install .                 # core (DepScore + SynGram)
pip install ".[eval]"         # + pandas/sacrebleu/bert-score for Evaluate.py
pip install ".[dashboard]"    # + streamlit for the diagnostic dashboard
pip install ".[comet]"        # + optional COMET comparison metric
```

For development (editable install, code changes picked up immediately):

```bash
pip install -e ".[eval,dashboard]"
```

## Usage

```python
from gris import compute_DepScore_emb, compute_syntactic_ngram_metric

dep = compute_DepScore_emb(hyps=["Der Hund lief schnell."],
                            refs=["Der Hund rannte schnell."], lang="de")
```

### Interpretability, without the dashboard

`compute_DepScore_emb` can print (or return) the exact same step-by-step
breakdown the dashboard shows — which dependency edges matched, their
similarity and bonuses, the precision/recall/Fβ, the language blend
weight, and which sentence-level penalties fired.

```python
from gris import compute_DepScore_emb

score = compute_DepScore_emb(
    hyps=["Der Hund lief schnell."],
    refs=["Der Hund rannte schnell."],
    lang="de",
    explain=True,   # prints the full breakdown as it scores
)
```

To get the breakdown back as data instead of (or as well as) printing it:

```python
score, details = compute_DepScore_emb(
    hyps=["Der Hund lief schnell."],
    refs=["Der Hund rannte schnell."],
    lang="de",
    return_details=True,
)

from gris import explain_dep_score
explain_dep_score(details[0])   # pretty-print pair 1's breakdown yourself
print(details[0]["matched"])    # or use the structured data directly
```

The CLI has the same option: `gris-score --hyp hyps.txt --ref refs.txt --lang de --explain`.

Or from the command line:

```bash
gris-score --hyp hyps.txt --ref refs.txt --lang de
```

Run the dashboard (after `pip install ".[dashboard]"`):

```bash
streamlit run $(python -c "import gris, os; print(os.path.join(os.path.dirname(gris.__file__), 'dashboard_corrected.py'))")
```

## Known packaging note

`gris/parser_constituency.py` imports `spacy` and `benepar` unconditionally,
and `gris/model_cache.py` imports that module unconditionally too — so
`spacy`/`benepar` are currently **required** even though constituency
parsing is only used for English and only affects SVG tree visualization.
If you want a lighter install, this import can be made lazy (moved inside
`ConstituencyParser.__init__`) on request.
