# Evaluate.py  —  GRIS evaluation script  (v8.1)
#
# Computes Spearman / Kendall / Pearson correlations at:
#   - Segment level  (per sentence pair vs MQM)
#   - System level   (per-system average vs MQM)
#
# Baseline metrics  : BLEU, CHRF, TER, METEOR, BERTScore, COMET
# GRIS-DepScore     : FULL_V2 + structural ablations + design ablations A/B/C/E
# GRIS-SynGram     : BASE + FULL (reproducibility) + V8 (beat-BERTScore config)
#
# V8.1 matcher.py change (this run):
#   GRIS-DEPSCORE_FULL now uses F_beta(1.5) instead of symmetric F1.
#   Evidence: WMT22 DE ablation shows ABL_C (f_beta=1.5) at Spearman=0.3137
#   vs old FULL (f_beta=1.0) at 0.2687 — a +0.045 gain on 2000 segments.
#   Column renamed GRIS-DEPSCORE_FULL_V2 so old and new results are
#   distinguishable in the report.
#
# V8 SynGram changes vs FULL:
#   - similarity_threshold 0.63 → 0.55  (recovers DE→EN paraphrase matches)
#   - order_weights [0.50,0.25,0.15,0.10] → [0.25,0.40,0.25,0.10]
#   - depth-gated coverage penalty in ngram_scorer
#
# Usage:
#   python Evaluate.py --data wmt22_de.csv --lang de
#                      --out results_de.csv --report report_de.txt
#
# Required CSV columns: hypothesis, reference, score  (MQM), system
# Optional            : source

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import linear_sum_assignment

from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score as bert_score
from nltk.translate.meteor_score import single_meteor_score

try:
    from comet import download_model, load_from_checkpoint
    HAS_COMET = True
except Exception:
    HAS_COMET = False

from sentence_transformers import SentenceTransformer

from gris.scorer import compute_DepScore_emb
from gris.parser import StanzaDependencyParser
from gris.ngram_scorer import compute_syntactic_ngram_metric
from gris.ngram_extractor import SynGramConfig
from gris.shared_utils import sentence_has_passive, sentence_has_negation


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

class DummyStemmer:
    def stem(self, word): return word

class DummyWordNet:
    def synsets(self, word): return []


def find_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            print(f"[INFO] Using '{c}' for {what}")
            return c
    raise KeyError(
        f"Cannot find {what} column. "
        f"Candidates={candidates}. Available={list(df.columns)}"
    )


def load_df(path):
    low = path.lower()
    if low.endswith(".xlsx") or low.endswith(".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


def tokenize(s):
    return str(s).strip().split()


def corr_triple(x, y):
    """Return (Spearman, Kendall, Pearson) ignoring NaN pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    return (
        float(spearmanr(x, y).correlation),
        float(kendalltau(x, y).correlation),
        float(pearsonr(x, y)[0]),
    )


def system_level(df, metric_col, target_col, sys_col):
    """
    Average metric + MQM per system, correlate.
    Returns (Spearman, Kendall, Pearson, n_systems).
    """
    if sys_col not in df.columns:
        return np.nan, np.nan, np.nan, 0
    grp = df.groupby(sys_col)[[metric_col, target_col]].mean().dropna()
    n = len(grp)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    sp, kt, pr = corr_triple(grp[metric_col].tolist(),
                              grp[target_col].tolist())
    return sp, kt, pr, n


# ══════════════════════════════════════════════════════════════════════════════
# Edge extraction (shared by structural ablations)
# ══════════════════════════════════════════════════════════════════════════════

STOP_UPOS = {"DET", "AUX", "PART", "SCONJ", "CCONJ", "INTJ", "PUNCT"}


def extract_edges(dep_parse):
    id2tok = {t.id: t for t in dep_parse.tokens}
    edges = []
    for tok in dep_parse.tokens:
        if not tok.head or tok.head == 0:
            continue
        if tok.upos in STOP_UPOS:
            continue
        head = id2tok.get(tok.head)
        if head is None or head.upos in STOP_UPOS:
            continue
        edges.append((head, tok))
    return edges


# ══════════════════════════════════════════════════════════════════════════════
# Structural ablation helpers (no_structure / no_embeddings / no_penalties)
# ══════════════════════════════════════════════════════════════════════════════

def cosine(u, v):
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if denom == 0 else float(np.dot(u, v) / denom)


def smooth_floor(sim, floor=0.30):
    if sim <= floor:
        return 0.0
    return (sim - floor) / (1.0 - floor)


def no_structure_cos(st_model, hyp, ref):
    """Sentence cosine only — no syntax."""
    v = st_model.encode([hyp, ref], convert_to_numpy=True,
                        normalize_embeddings=False)
    return max(0.0, min(1.0, (cosine(v[0], v[1]) + 1.0) / 2.0))


def no_embeddings_f1(dep_h, dep_r):
    """Structural F1 only — exact text match, no embeddings."""
    h_edges = extract_edges(dep_h)
    r_edges = extract_edges(dep_r)
    if not h_edges and not r_edges:
        return 1.0
    if not h_edges or not r_edges:
        return 0.0
    def key(e):
        hd, dp = e
        return (str(hd.text).lower(), str(dp.text).lower(), str(dp.deprel))
    Hset, Rset = set(key(e) for e in h_edges), set(key(e) for e in r_edges)
    tp   = len(Hset & Rset)
    prec = tp / max(1, len(Hset))
    rec  = tp / max(1, len(Rset))
    if prec + rec == 0:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec)
    lr = min(len(h_edges), len(r_edges)) / max(len(h_edges), len(r_edges))
    lp = 0.85 + 0.15 * lr if lr < 0.5 else 1.0
    return max(0.0, min(1.0, f1 * lp))


def _surface_embed_cache(dep_h, dep_r, st_model):
    texts = list({str(t.text).lower()
                  for dp in (dep_h, dep_r)
                  for t in dp.tokens})
    vecs  = st_model.encode(texts, convert_to_numpy=True,
                             normalize_embeddings=False)
    return {t: v for t, v in zip(texts, vecs)}


def no_penalties_f1(dep_h, dep_r, embed_cache):
    """Full edge matching with embeddings but zero penalties."""
    h_edges = extract_edges(dep_h)
    r_edges = extract_edges(dep_r)
    if not h_edges and not r_edges:
        return 1.0
    if not h_edges or not r_edges:
        return 0.0
    def tok(t): return str(t.text).lower()
    sim = np.zeros((len(h_edges), len(r_edges)), dtype=np.float32)
    for i, (hh, hd) in enumerate(h_edges):
        vh_h = embed_cache.get(tok(hh))
        vh_d = embed_cache.get(tok(hd))
        if vh_h is None or vh_d is None:
            continue
        for j, (rh, rd) in enumerate(r_edges):
            vr_h = embed_cache.get(tok(rh))
            vr_d = embed_cache.get(tok(rd))
            if vr_h is None or vr_d is None:
                continue
            head_s = (cosine(vh_h, vr_h) + 1.0) / 2.0
            dep_s  = (cosine(vh_d, vr_d) + 1.0) / 2.0
            sim[i, j] = smooth_floor(0.5 * (head_s + dep_s))
    row_ind, col_ind = linear_sum_assignment(-sim)
    total = float(sum(sim[i, j] for i, j in zip(row_ind, col_ind)))
    n_h, n_r = len(h_edges), len(r_edges)
    prec = total / n_h
    rec  = total / n_r
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    lr   = min(n_h, n_r) / max(n_h, n_r)
    lp   = 0.85 + 0.15 * lr if lr < 0.5 else 1.0
    return max(0.0, min(1.0, f1 * lp))


# ══════════════════════════════════════════════════════════════════════════════
# Design-decision ablations  A / B / C / E
# ══════════════════════════════════════════════════════════════════════════════

_W = {"en":0.75,"de":0.80,"ru":0.20,"fr":0.70,"es":0.70,
      "fi":0.65,"ar":0.55,"tr":0.50,"zh":0.25,"ja":0.20}
_W_DEF = 0.60

_CORE = {"nsubj","obj","iobj","csubj","ccomp","xcomp","nsubj:pass"}
_VOICE = {frozenset({"nsubj","obl:agent"}),
          frozenset({"nsubj","obl"}),
          frozenset({"obj","nsubj:pass"})}


def _to01(x): return max(0.0, min(1.0, (x + 1.0) / 2.0))


def _deprel_bonus(hr, rr):
    if hr == rr:            return 0.12
    if frozenset({hr,rr}) in _VOICE: return 0.10
    return 0.0


def _edge_score(hh_v, hd_v, rh_v, rd_v,
                hd_lem, rd_lem, hd_rel, rd_rel,
                hd_upos, rd_upos,
                emb_floor=0.30, head_base=0.70, head_scale=0.30,
                bonus_gate=0.82, additive=False):
    if any(v is None for v in (hh_v, hd_v, rh_v, rd_v)):
        return 0.0
    dep_s  = _to01(cosine(hd_v, rd_v))
    head_s = _to01(cosine(hh_v, rh_v))
    if additive:
        base = smooth_floor(0.5 * dep_s + 0.5 * head_s, floor=emb_floor)
    else:
        base = smooth_floor(dep_s, floor=emb_floor) * (head_base + head_scale * head_s)
    if base == 0.0:
        return 0.0
    bonuses = []
    if base >= bonus_gate:
        dr = _deprel_bonus(hd_rel, rd_rel)
        if dr:                                       bonuses.append(dr)
        if hd_upos == "VERB" and rd_upos == "VERB" and head_s > 0.70:
                                                     bonuses.append(0.05)
        if hd_rel in _CORE and rd_rel in _CORE:      bonuses.append(0.05)
    if hd_lem and rd_lem and hd_lem.lower() == rd_lem.lower():
        bonuses.append(0.10)
    return min(1.0, base + sum(bonuses))


def _embed_tokens(dep_parse, st_model):
    """Return {token_id: numpy_vector} using subword averaging."""
    tokens = dep_parse.tokens
    if not tokens:
        return {}
    words = [t.text for t in tokens]
    enc   = st_model.tokenizer(
        words, is_split_into_words=True,
        return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    )
    import torch
    with torch.no_grad():
        out = st_model[0].auto_model(**{k: v for k, v in enc.items()})
    hidden   = out.last_hidden_state[0].cpu().numpy()
    word_ids = enc.word_ids(batch_index=0)
    result   = {}
    for idx, tok in enumerate(tokens):
        pos = [p for p, w in enumerate(word_ids) if w == idx]
        if pos:
            result[tok.id] = hidden[pos].mean(axis=0)
    return result


def ablation_score(dep_h, dep_r, tv_h, tv_r, lang,
                   emb_floor=0.30, head_base=0.70, head_scale=0.30,
                   bonus_gate=0.82, additive=False, f_beta=1.0):
    """
    Full DepScore pipeline with one parameter changed.
    Ablation A: additive=True
    Ablation B: emb_floor=0.20
    Ablation C: f_beta=1.5
    Ablation E: bonus_gate=0.0
    """
    h_edges = extract_edges(dep_h)
    r_edges = extract_edges(dep_r)
    if not h_edges and not r_edges:
        edge_f1 = 1.0
    elif not h_edges or not r_edges:
        edge_f1 = 0.0
    else:
        sim = np.zeros((len(h_edges), len(r_edges)), dtype=np.float32)
        for i, (hh, hd) in enumerate(h_edges):
            for j, (rh, rd) in enumerate(r_edges):
                sim[i, j] = _edge_score(
                    tv_h.get(hh.id), tv_h.get(hd.id),
                    tv_r.get(rh.id), tv_r.get(rd.id),
                    getattr(hd, "lemma", None), getattr(rd, "lemma", None),
                    (getattr(hd, "deprel", "") or "").lower(),
                    (getattr(rd, "deprel", "") or "").lower(),
                    (getattr(hd, "upos", "") or ""),
                    (getattr(rd, "upos", "") or ""),
                    emb_floor=emb_floor, head_base=head_base,
                    head_scale=head_scale, bonus_gate=bonus_gate,
                    additive=additive,
                )
        row_ind, col_ind = linear_sum_assignment(-sim)
        total = float(sum(sim[i, j] for i, j in zip(row_ind, col_ind)))
        n_h, n_r = len(h_edges), len(r_edges)
        prec, rec = total / n_h, total / n_r
        if f_beta == 1.0:
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        else:
            b2 = f_beta ** 2
            den = b2 * prec + rec
            f1  = (1 + b2) * prec * rec / den if den else 0.0
        lr = min(n_h, n_r) / max(n_h, n_r)
        lp = 0.85 + 0.15 * lr if lr < 0.5 else 1.0
        edge_f1 = max(0.0, min(1.0, f1 * lp))
    return max(0.0, min(1.0, _W.get(lang, _W_DEF) * edge_f1))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GRIS evaluation — segment + system level correlations"
    )
    ap.add_argument("--data",    required=True,  help="Input CSV/TSV/XLSX path")
    ap.add_argument("--lang",    default="de",   help="Target language code")
    ap.add_argument("--task",    default=None,
                    help="Translation task direction for STRUCT_WEIGHT lookup "
                         "(default: same as --lang). Set to 'zh' when running "
                         "zh→en so that STRUCT_WEIGHT['zh']=0.25 is used even "
                         "though parse_lang='en'. Similarly 'ja' for ja→en etc.")
    ap.add_argument("--limit",   type=int, default=None,
                    help="Limit rows for quick testing")
    ap.add_argument("--out",     default="results.csv",
                    help="Output CSV with per-sentence scores")
    ap.add_argument("--report",  default="report.txt",
                    help="Output correlation report (text)")
    ap.add_argument("--target",  choices=["mqm", "human"], default="mqm",
                    help="Human judgement column to correlate against")
    ap.add_argument("--gpus",    type=int, default=0,
                    help="GPUs for COMET (0 = CPU)")
    ap.add_argument("--model",   default="minilm",
                    choices=["minilm", "mpnet", "labse"],
                    help="Sentence transformer model: "
                         "minilm=paraphrase-multilingual-MiniLM-L12-v2 (33M, fast), "
                         "mpnet=paraphrase-multilingual-mpnet-base-v2 (110M, stronger), "
                         "labse=LaBSE (470M, best cross-lingual)")
    args = ap.parse_args()

    # ── Derive parse_lang from translation direction ───────────────────────────
    # args.lang is the WMT language-pair identifier, e.g. "zh", "de", "ru".
    # For DepScore we need the language of the HYPOTHESIS and REFERENCE text,
    # because that is what the dependency parser and scorer operate on.
    #
    # For en→de and en→ru: hyp and ref are in the target language → parse_lang = args.lang
    # For zh→en: hyp and ref are BOTH English → parse_lang = "en"
    #
    # Running Stanza with lang="zh" on English text produces badly wrong parses
    # (wrong UPOS tags, wrong deprel labels), which corrupts all DepScore variants.
    # The STRUCT_WEIGHT and F_beta selection in matcher.py must also reflect the
    # text language, not the source language of the translation task.
    #
    # If you add a new language direction where the target is not args.lang
    # (e.g. ja→en), add it to this mapping.
    _TARGET_IS_ENGLISH = {"zh", "ja", "ko", "zh-en", "ja-en", "ko-en"}
    parse_lang = "en" if args.lang.lower() in _TARGET_IS_ENGLISH else args.lang
    if parse_lang != args.lang:
        print(f"[INFO] Translation direction: {args.lang}→en  "
              f"→ dependency parsing and scoring use lang='{parse_lang}' (target text is English)")

    # task_lang controls STRUCT_WEIGHT lookup and SynGram thresholds.
    # For zh→en: parse_lang="en" (text language) but task_lang="zh" (task direction).
    # STRUCT_WEIGHT["zh"]=0.25 correctly gives 75% cosine weight because
    # zh→en surface lexical overlap is uncorrelated with quality (Spearman=0.021, p=0.94).
    # STRUCT_WEIGHT["en"]=0.75 would give 75% edge F1 weight, swamping the cosine signal.
    # Use --task zh when running zh→en to get the correct blend without changing parse_lang.
    task_lang = (args.task or args.lang).lower()
    if task_lang != parse_lang:
        print(f"[INFO] Task direction: {task_lang}  "
              f"→ STRUCT_WEIGHT['{task_lang}'] = {task_lang} used for score blending")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_df(args.data)
    if args.limit:
        df = df.head(args.limit)
    print(f"[INFO] Loaded {len(df)} segments from {args.data}")

    hyp_col = find_col(df, ["hypothesis", "mt", "hyp"], "hypothesis")
    ref_col = find_col(df, ["reference", "ref"],         "reference")
    sys_col = next((c for c in ["system","sys","system_id","mt_system",
                                "systemID","System"] if c in df.columns), None)
    if sys_col:
        print(f"[INFO] System column: '{sys_col}'  "
              f"({df[sys_col].nunique()} systems)")
    else:
        print("[WARN] No system column found — system-level correlations skipped")

    if args.target == "mqm":
        tgt_col = find_col(df, ["mqm_score","score","core","mqm"], "MQM target")
    else:
        tgt_col = find_col(df, ["human_score","human","da","z_score"],
                           "human target")

    hyps   = df[hyp_col].astype(str).tolist()
    refs   = df[ref_col].astype(str).tolist()
    target = df[tgt_col].tolist()

    # ── Baseline metrics ──────────────────────────────────────────────────────
    print("[INFO] Computing BLEU / CHRF / TER ...")
    _bleu = BLEU(effective_order=True)
    _chrf = CHRF()
    _ter  = TER()
    bleu_scores = [_bleu.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    chrf_scores = [_chrf.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    ter_scores  = [_ter.sentence_score(h,  [r]).score for h, r in zip(hyps, refs)]

    print("[INFO] Computing METEOR ...")
    _stemmer, _wn = DummyStemmer(), DummyWordNet()
    meteor_scores = [
        single_meteor_score(tokenize(r), tokenize(h),
                            stemmer=_stemmer, wordnet=_wn)
        for h, r in zip(hyps, refs)
    ]

    print("[INFO] Computing BERTScore ...")
    _, _, F1 = bert_score(hyps, refs, lang=parse_lang,
                          rescale_with_baseline=False)
    bert_scores = F1.tolist()

    if HAS_COMET:
        print("[INFO] Loading COMET ...")
        model_path  = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_data  = [{"src": "", "mt": h, "ref": r}
                       for h, r in zip(hyps, refs)]
        comet_scores = comet_model.predict(
            comet_data, batch_size=8, gpus=args.gpus)["scores"]
    else:
        print("[WARN] COMET not installed — filling with NaN")
        comet_scores = [np.nan] * len(hyps)

    # ── Dependency parsing ────────────────────────────────────────────────────
    print(f"[INFO] Loading Stanza parser (lang='{parse_lang}') and parsing all sentences ...")
    dep_parser = StanzaDependencyParser(lang=parse_lang)
    all_parsed = dep_parser.parse(hyps + refs)
    dep_hyps   = all_parsed[:len(hyps)]
    dep_refs   = all_parsed[len(hyps):]

    bad = sum(1 for d in dep_hyps + dep_refs
              if not d or not d.tokens or len(d.tokens) < 2)
    if bad:
        print(f"[WARN] {bad}/{2*len(hyps)} parses look empty. "
              f"Run: import stanza; stanza.download('{parse_lang}')")

    # ── Sentence transformer ──────────────────────────────────────────────────
    _MODEL_MAP = {
        "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "mpnet":  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "labse":  "sentence-transformers/LaBSE",
    }
    st_name  = _MODEL_MAP[args.model]
    print(f"[INFO] Loading sentence transformer: {st_name} ...")
    st_model = SentenceTransformer(st_name)

    # ── SynGram batch scoring ────────────────────────────────────────────────
    # v8: cfg_base and cfg_full kept identical to previous run so old results
    # are reproducible.  cfg_v8 is the new beat-BERTScore configuration with:
    #   - similarity_threshold lowered 0.63 → 0.55  (recovers paraphrase matches)
    #   - order_weights bigram-dominant [0.25, 0.40, 0.25, 0.10]
    #   - all penalties active

    cfg_base = SynGramConfig(
        matching="greedy",
        max_n=3,
        lemma_content_only=True,
        voice_normalize=True,
        collapse_roles=True,
        verb_only_lemmas=False,
        # BASE: no penalties, no sentence-level negation penalty
        arg_count_mismatch_penalty=0.0,
        neg_edge_penalty=0.0,
        neg_sent_penalty=1.0,
        similarity_threshold=0.63,   # kept for reproducibility
        order_weights=[0.50, 0.25, 0.15, 0.10],  # kept for reproducibility
    )

    cfg_full = SynGramConfig(
        matching="greedy",
        max_n=4,
        lemma_content_only=True,
        voice_normalize=True,
        collapse_roles=True,
        verb_only_lemmas=False,
        # FULL: all penalties active (old thresholds — kept for reproducibility)
        arg_count_mismatch_penalty=0.20,
        neg_edge_penalty=0.25,
        neg_sent_penalty=0.65,
        similarity_threshold=0.63,   # kept for reproducibility
        order_weights=[0.50, 0.25, 0.15, 0.10],  # kept for reproducibility
    )

    cfg_v8 = SynGramConfig(
        matching="greedy",
        max_n=4,
        lemma_content_only=True,
        voice_normalize=True,
        collapse_roles=True,
        verb_only_lemmas=False,
        # v8: beat-BERTScore configuration
        # Threshold lowered: MiniLM valid DE→EN synonym pairs cluster at 0.58–0.62;
        # old 0.63 was silently dropping legitimate paraphrase matches.
        # Order weights bigram-dominant: n=2 head+dep pairs encode predicate-argument
        # structure — the structural advantage that distinguishes SynGram from BERTScore.
        # n=1 unigrams are essentially token-level cosine, inferior to BERTScore's
        # sentence-level BERT; reducing their weight forces syntactic context to dominate.
        arg_count_mismatch_penalty=0.20,
        neg_edge_penalty=0.25,
        neg_sent_penalty=0.65,
        similarity_threshold=0.55,
        order_weights=[0.25, 0.40, 0.25, 0.10],
        adaptive_threshold=True,
    )

    print("[INFO] Computing GRIS-SynGram BASE ...")
    _, ng_base_dbg = compute_syntactic_ngram_metric(
        hyps, refs,
        lang=task_lang,
        parse_lang=parse_lang,
        embedding_type="transformer",
        model_name=st_name,
        cfg=cfg_base,
        return_pair_scores=True,
        penalty_mismatch_only=True,
    )

    print("[INFO] Computing GRIS-SynGram FULL ...")
    _, ng_full_dbg = compute_syntactic_ngram_metric(
        hyps, refs,
        lang=task_lang,
        parse_lang=parse_lang,
        embedding_type="transformer",
        model_name=st_name,
        cfg=cfg_full,
        return_pair_scores=True,
        penalty_mismatch_only=True,
    )

    print("[INFO] Computing GRIS-SynGram V8 (beat-BERTScore config) ...")
    _, ng_v8_dbg = compute_syntactic_ngram_metric(
        hyps, refs,
        lang=task_lang,
        parse_lang=parse_lang,
        embedding_type="transformer",
        model_name=st_name,
        cfg=cfg_v8,
        return_pair_scores=True,
        penalty_mismatch_only=True,
    )

    ng_base_pairs = (ng_base_dbg or {}).get("pairs", [])
    ng_full_pairs = (ng_full_dbg or {}).get("pairs", [])
    ng_v8_pairs   = (ng_v8_dbg   or {}).get("pairs", [])

    # ── Per-sentence GRIS-DepScore loop ───────────────────────────────────────
    PASSIVE_PEN = 0.90
    NEG_PEN     = 0.90   # v2 value

    (dep_full, dep_no_struct, dep_no_embed, dep_no_pen,
     dep_abl_A, dep_abl_B, dep_abl_C, dep_abl_E,
     ng_base_scores, ng_full_scores, ng_v8_scores,
     flag_passive, flag_neg) = (
        [], [], [], [], [], [], [], [], [], [], [], [], []
    )

    print("[INFO] Computing per-sentence DepScore + ablations ...")
    for i, (h, r) in enumerate(zip(hyps, refs), 1):
        dep_h = dep_hyps[i - 1]
        dep_r = dep_refs[i - 1]

        # Full DepScore (uses matcher.py internally — task_lang for STRUCT_WEIGHT)
        s = compute_DepScore_emb(
            [h], [r], lang=task_lang,
            embedding_type="transformer", model_name=st_name,
            matching="hungarian", save_svg=False, csv_output=None,
            passive_sent_penalty=PASSIVE_PEN, neg_sent_penalty=NEG_PEN,
        )
        dep_full.append(float(s))

        # Structural ablations
        dep_no_struct.append(float(no_structure_cos(st_model, h, r)))
        dep_no_embed.append(float(no_embeddings_f1(dep_h, dep_r)))
        cache = _surface_embed_cache(dep_h, dep_r, st_model)
        dep_no_pen.append(float(no_penalties_f1(dep_h, dep_r, cache)))

        # Design-decision ablations A / B / C / E
        try:
            tv_h = _embed_tokens(dep_h, st_model)
            tv_r = _embed_tokens(dep_r, st_model)
            dep_abl_A.append(ablation_score(dep_h, dep_r, tv_h, tv_r,
                lang=task_lang, additive=True))
            dep_abl_B.append(ablation_score(dep_h, dep_r, tv_h, tv_r,
                lang=task_lang, emb_floor=0.20))
            dep_abl_C.append(ablation_score(dep_h, dep_r, tv_h, tv_r,
                lang=task_lang, f_beta=1.5))
            dep_abl_E.append(ablation_score(dep_h, dep_r, tv_h, tv_r,
                lang=task_lang, bonus_gate=0.0))
        except Exception as e:
            print(f"  [WARN] Ablation failed at i={i}: {e}")
            for lst in (dep_abl_A, dep_abl_B, dep_abl_C, dep_abl_E):
                lst.append(dep_full[-1])

        # SynGram scores
        ng_base_scores.append(
            float((ng_base_pairs[i-1] if i-1 < len(ng_base_pairs)
                   else {}).get("score", 0.0)))
        ng_full_scores.append(
            float((ng_full_pairs[i-1] if i-1 < len(ng_full_pairs)
                   else {}).get("score", 0.0)))
        ng_v8_scores.append(
            float((ng_v8_pairs[i-1] if i-1 < len(ng_v8_pairs)
                   else {}).get("score", 0.0)))

        # Penalty flags
        h_pass = sentence_has_passive(dep_h.tokens)
        r_pass = sentence_has_passive(dep_r.tokens)
        h_neg  = sentence_has_negation(dep_h.tokens)
        r_neg  = sentence_has_negation(dep_r.tokens)
        flag_passive.append(int(h_pass != r_pass))
        flag_neg.append(int(h_neg != r_neg))

        if i % 200 == 0:
            print(f"  {i}/{len(hyps)} pairs done")

    # ── Build output DataFrame ────────────────────────────────────────────────
    df_out = df.copy()

    # Baselines
    df_out["BLEU"]      = bleu_scores
    df_out["CHRF"]      = chrf_scores
    df_out["TER"]       = ter_scores
    df_out["METEOR"]    = meteor_scores
    df_out["BERTSCORE"] = bert_scores
    df_out["COMET"]     = comet_scores

    # GRIS-DepScore
    # FULL_V2: matcher.py now uses F_beta(1.5) instead of symmetric F1.
    # ABL_C is kept as-is for reproducibility — it will now be identical
    # to FULL_V2 in the F-score formula, but uses the ablation runner's
    # standalone _edge_score path rather than matcher.py's full pipeline.
    df_out["GRIS-DEPSCORE_FULL_V2"]       = dep_full    # F_beta(1.5), v2.1 matcher
    df_out["GRIS-DEPSCORE_NO_STRUCTURE"]  = dep_no_struct
    df_out["GRIS-DEPSCORE_NO_EMBEDDINGS"] = dep_no_embed
    df_out["GRIS-DEPSCORE_NO_PENALTIES"]  = dep_no_pen
    df_out["GRIS-DEPSCORE_ABL_A"]         = dep_abl_A   # additive formula
    df_out["GRIS-DEPSCORE_ABL_B"]         = dep_abl_B   # floor=0.20
    df_out["GRIS-DEPSCORE_ABL_C"]         = dep_abl_C   # F_beta(1.5) — now same as FULL_V2
    df_out["GRIS-DEPSCORE_ABL_E"]         = dep_abl_E   # ungated bonuses

    # GRIS-SynGram
    df_out["GRIS_NGRAM_BASE"] = ng_base_scores
    df_out["GRIS_NGRAM_FULL"] = ng_full_scores
    df_out["GRIS_NGRAM_V8"]   = ng_v8_scores   # beat-BERTScore config

    # Penalty flags (useful for analysis)
    df_out["FLAG_PASSIVE_MISMATCH"] = flag_passive
    df_out["FLAG_NEG_MISMATCH"]     = flag_neg

    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved scores to {args.out}")

    # ── Correlation reporting ─────────────────────────────────────────────────
    METRICS = [
        # Baselines
        "BLEU", "CHRF", "TER", "METEOR", "BERTSCORE", "COMET",
        # GRIS-DepScore
        "GRIS-DEPSCORE_FULL_V2",
        "GRIS-DEPSCORE_NO_STRUCTURE",
        "GRIS-DEPSCORE_NO_EMBEDDINGS",
        "GRIS-DEPSCORE_NO_PENALTIES",
        "GRIS-DEPSCORE_ABL_A",
        "GRIS-DEPSCORE_ABL_B",
        "GRIS-DEPSCORE_ABL_C",
        "GRIS-DEPSCORE_ABL_E",
        # GRIS-SynGram
        "GRIS_NGRAM_BASE",
        "GRIS_NGRAM_FULL",
        "GRIS_NGRAM_V8",
    ]

    lines = []

    # ── Segment-level ─────────────────────────────────────────────────────────
    lines.append(
        f"Segment-level correlations  "
        f"(target={args.target.upper()} using '{tgt_col}'):"
    )
    lines.append(f"{'Metric':<32}  {'Spearman':>9}  {'Kendall':>9}  {'Pearson':>9}")
    lines.append("-" * 65)
    for m in METRICS:
        if m not in df_out.columns or df_out[m].isna().all():
            continue
        sp, kt, pr = corr_triple(df_out[m].tolist(), target)
        lines.append(f"{m:<32}  {sp:>9.4f}  {kt:>9.4f}  {pr:>9.4f}")

    # ── System-level ──────────────────────────────────────────────────────────
    lines.append("")
    if sys_col:
        lines.append(
            f"System-level correlations  "
            f"(averaged per system, target={args.target.upper()}):"
        )
        lines.append(
            f"  {'Metric':<32}  {'Spearman':>9}  {'Pearson':>9}  "
            f"{'Kendall':>9}  {'N_sys':>6}"
        )
        lines.append("  " + "-" * 65)
        sys_rows = []
        for m in METRICS:
            if m not in df_out.columns or df_out[m].isna().all():
                continue
            sp, kt, pr, n = system_level(df_out, m, tgt_col, sys_col)
            sys_rows.append((sp, m, kt, pr, n))
        for sp, m, kt, pr, n in sorted(sys_rows, key=lambda x: -x[0]
                                        if not np.isnan(x[0]) else -999):
            lines.append(
                f"  {m:<32}  {sp:>9.4f}  {pr:>9.4f}  {kt:>9.4f}  {n:>6}"
            )
    else:
        lines.append(
            "[WARN] System-level correlations skipped — no system column."
        )

    report_text = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[INFO] Saved report to {args.report}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()