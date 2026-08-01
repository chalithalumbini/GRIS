"""
Compute DepScore (dependency-based + embedding + penalties),
with optional dependency/constituency tree visualizations.
"""

import os
import sys
import csv
import ftfy
import unicodedata
from graphviz import Digraph
from nltk import Tree

# Make `import gris.*` work whether or not this package has been pip-installed
# (see dashboard_corrected.py for the same pattern / explanation).
try:
    import gris  # noqa: F401
except ImportError:
    _GRIS_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _GRIS_PARENT not in sys.path:
        sys.path.insert(0, _GRIS_PARENT)

from gris.parser import DepSentence
from gris.matcher import compute_sentence_similarity, explain_dep_score

# Centralized caching for heavy models
from gris.model_cache import get_stanza_parser, get_embedder, get_constituency_parser


def normalize(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def pretty_dep(dep_sent: DepSentence) -> str:
    header = f"{'ID':<4}{'WORD':<15}{'LEMMA':<15}{'POS':<8}{'HEAD':<6}{'REL'}"
    lines = [header, "-" * 60]
    for tok in dep_sent.tokens:
        lemma = tok.lemma or ""
        upos = tok.upos or ""
        lines.append(f"{tok.id:<4}{tok.text:<15}{lemma:<15}{upos:<8}{tok.head:<6}{tok.deprel}")
    return "\n".join(lines)


def _strip_svg_ext(path: str) -> str:
    return path[:-4] if path.lower().endswith(".svg") else path


def save_dep_svg(dep_sent: DepSentence, filepath: str) -> None:
    filepath = _strip_svg_ext(filepath)
    try:
        dot = Digraph(format="svg")
        for tok in dep_sent.tokens:
            dot.node(str(tok.id), f"{tok.text}\n({tok.upos or ''})")
        for tok in dep_sent.tokens:
            if tok.head and tok.head != 0:
                dot.edge(str(tok.head), str(tok.id), label=tok.deprel or "")
        dot.render(filepath, cleanup=True)
    except Exception as e:
        print(f"[WARN] Failed to render dependency SVG ({filepath}): {e}")


def save_const_svg(tree_str: str, filepath: str) -> None:
    filepath = _strip_svg_ext(filepath)
    try:
        tree = Tree.fromstring(tree_str)
        dot = Digraph(format="svg")

        def add_nodes_edges(t, parent=None):
            node_id = str(id(t))
            label = t.label() if isinstance(t, Tree) else str(t)
            dot.node(node_id, label)
            if parent:
                dot.edge(parent, node_id)
            if isinstance(t, Tree):
                for child in t:
                    add_nodes_edges(child, node_id)

        add_nodes_edges(tree)
        dot.render(filepath, cleanup=True)
    except Exception as e:
        print(f"[WARN] Failed to render constituency SVG ({filepath}): {e}")


def compute_DepScore_emb(
    hyps,
    refs,
    lang: str = "en",
    model_name: str = "glove-wiki-gigaword-300",
    embedding_type: str = "word",
    matching: str = "hungarian",
    debug: bool = False,
    save_path=None,   # kept for backward-compat; not used
    save_svg: bool = False,
    csv_output: str | None = None,
    # NEW: global penalty controls (mismatch-only)
    # NOTE: passive_sent_penalty is accepted for backward compatibility but is
    # NOT applied inside compute_sentence_similarity() — passive mismatches are
    # flagged only, never penalized (WMT22 final design).
    passive_sent_penalty: float = 0.90,
    neg_sent_penalty: float = 0.90,
    # NEW: interpretability
    return_details: bool = False,
    explain: bool = False,
):
    """
    Compute GRIS-DepScore for a list of hypothesis/reference pairs.

    By default returns just the average score (float), unchanged from
    earlier versions.

    Set explain=True to print a step-by-step, human-readable breakdown of
    each pair's score (edge matches, penalties, blend weights) as it runs —
    this is the interpretability the dashboard shows, available here for
    plain script/notebook usage.

    Set return_details=True to additionally get the breakdown back as data:
    returns (average_score, details_list), where details_list[i] is the
    dict for pair i (usable directly with `explain_dep_score()` yourself,
    or for building your own reports/tables).
    """
    # Reuse heavy resources across calls (no repeated loading)
    dep_parser = get_stanza_parser(lang=lang)
    const_parser = get_constituency_parser(lang) if lang == "en" else None
    embedder = get_embedder(model_name=model_name, embedding_type=embedding_type)

    total_score = 0.0
    csv_results = []
    details_list = [] if (return_details or explain) else None

    svg_dir = os.path.join(os.getcwd(), "trees")
    if save_svg:
        os.makedirs(svg_dir, exist_ok=True)

    for idx, (hyp, ref) in enumerate(zip(hyps, refs), start=1):
        hyp = normalize(hyp)
        ref = normalize(ref)

        dep_h = dep_parser.parse([hyp])[0]
        dep_r = dep_parser.parse([ref])[0]

        if debug:
            print("\n[Dependency Tree – HYP]")
            print(pretty_dep(dep_h))
            print("\n[Dependency Tree – REF]")
            print(pretty_dep(dep_r))

        # Constituency parsing (English only, if available)
        if const_parser:
            const_h_list = const_parser.parse([hyp])
            const_r_list = const_parser.parse([ref])

            const_h = const_h_list[0][0] if const_h_list and const_h_list[0] else None
            const_r = const_r_list[0][0] if const_r_list and const_r_list[0] else None
        else:
            const_h = const_r = None

        if save_svg:
            save_dep_svg(dep_h, os.path.join(svg_dir, f"pair{idx}_dep_hyp"))
            save_dep_svg(dep_r, os.path.join(svg_dir, f"pair{idx}_dep_ref"))
            if const_h:
                save_const_svg(const_h, os.path.join(svg_dir, f"pair{idx}_const_hyp"))
            if const_r:
                save_const_svg(const_r, os.path.join(svg_dir, f"pair{idx}_const_ref"))

        if return_details or explain:
            s, details = compute_sentence_similarity(
                dep_h,
                dep_r,
                embedder,
                lang=lang,
                matching=matching,
                debug=debug,
                soften_structure=True,
                passive_sent_penalty=passive_sent_penalty,
                neg_sent_penalty=neg_sent_penalty,
                return_details=True,
            )
            details_list.append(details)
            if explain:
                print(f"\n╔══ Pair {idx}/{len(hyps)} " + "═" * 50)
                explain_dep_score(details, hyp=hyp, ref=ref)
        else:
            s = compute_sentence_similarity(
                dep_h,
                dep_r,
                embedder,
                lang=lang,
                matching=matching,
                debug=debug,
                soften_structure=True,
                passive_sent_penalty=passive_sent_penalty,
                neg_sent_penalty=neg_sent_penalty,
            )

        total_score += s
        csv_results.append(
            {
                "pair_id": idx,
                "reference": ref,
                "hypothesis": hyp,
                "score": float(s),
            }
        )

        if debug:
            print(f"PAIR {idx}: DepScore = {s:.4f}")

    final_score = total_score / len(hyps) if hyps else 0.0
    print(f"\n[FINAL] DepScore Average = {final_score:.4f}")

    if csv_output:
        with open(csv_output, "w", newline="", encoding="utf-8-sig") as csvfile:
            fieldnames = ["pair_id", "reference", "hypothesis", "score"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results)
        print(f"[INFO] Results saved to CSV: {csv_output}")

    if return_details:
        return final_score, details_list
    return final_score


def main():
    """CLI entry point (also installed as the `gris-score` console script)."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute GRIS-DepScore")
    parser.add_argument("--hyp", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--model", type=str, default="glove-wiki-gigaword-300")
    parser.add_argument("--embedding_type", type=str, choices=["word", "transformer"], default="word")
    parser.add_argument("--matching", type=str, choices=["hungarian", "greedy"], default="hungarian")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--explain", action="store_true",
                         help="Print a step-by-step breakdown of how each score was derived.")
    parser.add_argument("--save_svg", action="store_true")
    parser.add_argument("--csv", type=str, default=None)

    # NEW CLI knobs
    parser.add_argument("--passive_sent_penalty", type=float, default=0.90)
    parser.add_argument("--neg_sent_penalty", type=float, default=0.90)

    args = parser.parse_args()

    def load_sentences(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    hyps = load_sentences(args.hyp)
    refs = load_sentences(args.ref)

    if len(hyps) != len(refs):
        raise ValueError("Hypotheses and references must have same length.")

    compute_DepScore_emb(
        hyps,
        refs,
        lang=args.lang,
        model_name=args.model,
        embedding_type=args.embedding_type,
        matching=args.matching,
        debug=args.debug,
        explain=args.explain,
        save_svg=args.save_svg,
        csv_output=args.csv,
        passive_sent_penalty=args.passive_sent_penalty,
        neg_sent_penalty=args.neg_sent_penalty,
    )


if __name__ == "__main__":
    main()