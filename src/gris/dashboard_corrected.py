"""
dashboard.py  —  GRIS Interpretable MT Evaluation Dashboard  (v3.1)
Run:  streamlit run dashboard.py

CORRECTED v3.1 (July 2026) — WMT22 Final Code Alignment, COMET/diagnostics removed
-----------------------------------------------------------------------------------
This version reflects the ACTUAL penalties used in the WMT22 evaluation
as implemented in matcher.py:

WMT22 FINAL CODE (matcher.py):
- Negation mismatch: ×0.90 penalty APPLIED (line 561-562)
- Passive mismatch: NO penalty (line 460: passive_sent_penalty=1.0, never applied)

v3.1 changes:
- COMET scoring and the Source input field removed entirely.
- Tense-mismatch and role-swap diagnostics removed. They were labeled
  "diagnostic only" but were, in fact, still being multiplied into the
  final DepScore/SynGram scores (v3.0 bug) — removing them both fixes
  that and aligns the dashboard strictly with the two confirmed WMT22
  final penalties (negation, passive).

See thesis Chapter 3 "System Design and Implementation" for documentation.

Part of the `gris` package — imports the other gris.* modules via relative import.
"""

import sys, os, ftfy, unicodedata
import numpy as np
import streamlit as st
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))

from sacrebleu.metrics import BLEU, CHRF
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer

from gris.parser import StanzaDependencyParser
from gris.ngram_scorer import compute_syntactic_ngram_metric
from gris.ngram_extractor import SynGramConfig
from gris.shared_utils import sentence_has_passive, sentence_has_negation

# ── Constants (v3.1 — WMT22 final evaluation code alignment) ─────────────────
ST_MODEL    = "paraphrase-multilingual-mpnet-base-v2"

# WMT22 FINAL CODE (matcher.py):
# - Negation penalty: ×0.90 (line 88: NEG_SENT_PENALTY, applied line 561-562)
# - Passive penalty: NONE (line 460: passive_sent_penalty=1.0 default, never applied)

NEG_PEN     = 0.90   # GRIS-DepScore negation mismatch (WMT22 final)

STOP_UPOS   = {"DET", "AUX", "PART", "SCONJ", "CCONJ", "INTJ", "PUNCT"}

LANGS = {
    "en — English":  "en",
    "de — German":   "de",
    "ru — Russian":  "ru",
    "fr — French":   "fr",
    "es — Spanish":  "es",
    "zh — Chinese":  "zh",
    "ja — Japanese": "ja",
    "ar — Arabic":   "ar",
    "fi — Finnish":  "fi",
    "tr — Turkish":  "tr",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", ftfy.fix_text(str(text).strip()))

def cosine(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / d) if d > 0 else 0.0

def cos01(u, v):
    return (cosine(u, v) + 1.0) / 2.0

def extract_edges(dep_parse):
    """Content dependency edges with AUX modal inclusion (v2.4)."""
    id2tok = {t.id: t for t in dep_parse.tokens}
    edges  = []
    for tok in dep_parse.tokens:
        if not tok.head or tok.head == 0:
            continue
        if tok.upos in STOP_UPOS:
            # AUX modal exception: include if head is VERB
            if tok.upos == "AUX":
                head   = id2tok.get(tok.head)
                deprel = (tok.deprel or "").lower()
                if head and head.upos == "VERB" and deprel in {"aux", "aux:pass"}:
                    edges.append((head, tok))
            continue
        head = id2tok.get(tok.head)
        if head is None or head.upos in STOP_UPOS:
            continue
        edges.append((head, tok))
    return edges

def build_embed_cache(dep_h, dep_r, st_model) -> dict:
    texts = list({str(t.text).lower() for p in (dep_h, dep_r) for t in p.tokens})
    vecs  = st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return {t: v for t, v in zip(texts, vecs)}

def score_color(v: float) -> str:
    if v >= 0.70: return "green"
    if v >= 0.45: return "orange"
    return "red"

# ── Cached resource loading ───────────────────────────────────────────────────
@st.cache_resource
def load_st_model():
    return SentenceTransformer(ST_MODEL)

@st.cache_resource
def load_dep_parser(lang: str):
    return StanzaDependencyParser(lang=lang)

# ── Core computation ──────────────────────────────────────────────────────────
def compute_all(hyp: str, ref: str, lang: str):
    hyp = normalize(hyp)
    ref = normalize(ref)

    st_model   = load_st_model()
    dep_parser = load_dep_parser(lang)

    # ── Surface metrics ───────────────────────────────────────────────────────
    _bleu = BLEU(effective_order=True)
    _chrf = CHRF()
    bleu  = _bleu.sentence_score(hyp, [ref]).score
    chrf  = _chrf.sentence_score(hyp, [ref]).score
    _, _, F1 = bert_score_fn([hyp], [ref], lang=lang,
                              rescale_with_baseline=False, verbose=False)
    bertscore = float(F1[0])

    # ── Dependency parse ──────────────────────────────────────────────────────
    parsed  = dep_parser.parse([hyp, ref])
    dep_hyp = parsed[0]
    dep_ref = parsed[1]

    # Parse quality check
    def _parse_ok(dp):
        if not dp or not dp.tokens or len(dp.tokens) < 2:
            return False, "empty parse"
        n    = len(dp.tokens)
        flat = sum(1 for t in dp.tokens
                   if (getattr(t, "deprel", "") or "").lower() == "flat")
        if n > 2 and flat / n > 0.60:
            return False, f"{flat}/{n} edges are 'flat' — wrong Stanza model"
        return True, "ok"

    _hyp_ok, _hyp_w = _parse_ok(dep_hyp)
    _ref_ok, _ref_w = _parse_ok(dep_ref)
    _parse_warnings = []
    if not _hyp_ok: _parse_warnings.append(f"Hypothesis: {_hyp_w}")
    if not _ref_ok: _parse_warnings.append(f"Reference: {_ref_w}")

    # ── DepScore edge matching — v2.4 multiplicative formula ──────────────────
    h_edges = extract_edges(dep_hyp)
    r_edges = extract_edges(dep_ref)
    cache   = build_embed_cache(dep_hyp, dep_ref, st_model)

    def tok(t): return str(t.text).lower()

    sim_matrix = np.zeros((len(h_edges), len(r_edges)), dtype=np.float32)
    for i, (hh, hd) in enumerate(h_edges):
        vh_h = cache.get(tok(hh)); vh_d = cache.get(tok(hd))
        if vh_h is None or vh_d is None: continue
        for j, (rh, rd) in enumerate(r_edges):
            vr_h = cache.get(tok(rh)); vr_d = cache.get(tok(rd))
            if vr_h is None or vr_d is None: continue

            dep_s  = cos01(vh_d, vr_d)
            head_s = cos01(vh_h, vr_h)
            sim_matrix[i, j] = dep_s * (1.0 + head_s) / 2.0

    matched_edges, sim_total = [], 0.0
    unmatched_hyp, unmatched_ref = [], []

    if h_edges and r_edges:
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        matched_h = set(row_ind)
        matched_r = set(col_ind)

        for i, j in zip(row_ind, col_ind):
            hh, hd = h_edges[i]; rh, rd = r_edges[j]
            s = float(sim_matrix[i, j]); sim_total += s
            matched_edges.append({
                "hyp":  f"{hh.text} → {hd.text}",
                "ref":  f"{rh.text} → {rd.text}",
                "rel":  str(hd.deprel or "dep"),
                "sim":  round(s, 3),
                "note": (f"rel mismatch ({hd.deprel}→{rd.deprel})"
                         if str(hd.deprel) != str(rd.deprel) else ""),
            })

        unmatched_hyp = [f"{h_edges[i][0].text}→{h_edges[i][1].text}"
                         for i in range(len(h_edges)) if i not in matched_h]
        unmatched_ref = [f"{r_edges[j][0].text}→{r_edges[j][1].text}"
                         for j in range(len(r_edges)) if j not in matched_r]

    n_h  = max(len(h_edges), 1)
    n_r  = max(len(r_edges), 1)
    prec = sim_total / n_h
    rec  = sim_total / n_r

    # F_beta(1.5) — recall-weighted (v2.1+)
    beta   = 1.5
    b2     = beta ** 2
    f_beta = ((1 + b2) * prec * rec / (b2 * prec + rec)
              if (prec + rec) > 0 else 0.0)

    # Verbosity penalty (v2.1+)
    vp = 0.92 if (n_h > 1.4 * n_r) else 1.0
    f_beta *= vp

    # Global mismatch flags
    passive_mm = (sentence_has_passive(dep_hyp.tokens) !=
                  sentence_has_passive(dep_ref.tokens))
    neg_mm     = (sentence_has_negation(dep_hyp.tokens) !=
                  sentence_has_negation(dep_ref.tokens))

    # WMT22 FINAL: Only negation penalty applied (passive NOT penalized)
    d_passive  = 1.0   # always 1.0 — passive mismatch flagged but never penalizes the score
    d_neg      = NEG_PEN     if neg_mm     else 1.0

    # Sentence cosine for W blend
    sent_cos = cos01(
        st_model.encode(hyp, convert_to_numpy=True),
        st_model.encode(ref, convert_to_numpy=True),
    )

    # Language-adaptive W blend (v2.2+)
    W_MAP = {"de": 0.92, "ru": 0.20, "zh": 0.25, "en": 0.75}
    W        = W_MAP.get(lang, 0.75)
    dep_base = W * f_beta + (1.0 - W) * sent_cos
    
    # WMT22 FINAL: passive NOT included (always 1.0)
    dep_final = float(np.clip(
        dep_base * d_neg, 0, 1))

    # ── GRIS-SynGram (v8 — subtree path n-grams) ──────────────────────────────
    _cfg = SynGramConfig(
        matching="greedy",
        max_n=4,
        lemma_content_only=True,
        voice_normalize=True,
        collapse_roles=True,
        verb_only_lemmas=False,
        similarity_threshold=0.55,
        arg_count_mismatch_penalty=0.20,     # v8
        neg_edge_penalty=0.25,               # v8
        neg_sent_penalty=NEG_PEN,
        order_weights=[0.25, 0.40, 0.25, 0.10],
    )
    _, ng_debug = compute_syntactic_ngram_metric(
        [hyp], [ref], lang=lang,
        embedding_type="transformer", model_name=ST_MODEL,
        cfg=_cfg,
        return_pair_scores=True,
        penalty_mismatch_only=True,
    )
    pairs_list = (ng_debug or {}).get("pairs") or []
    pair_info  = pairs_list[0] if pairs_list else {}
    ng_base    = float(pair_info.get("score", 0.0))
    ng_final   = ng_base * d_passive * d_neg

    per_order  = pair_info.get("per_order",  {})
    head_pairs = pair_info.get("head_pairs", [])

    return {
        # top-level scores
        "bleu":       round(bleu,       2),
        "chrf":       round(chrf,       2),
        "bertscore":  round(bertscore,  4),
        "dep_score":  round(dep_final,  4),
        "ng_score":   round(ng_final,   4),
        # parse tokens / edges
        "ref_tokens": [(str(t.text), str(t.upos or "X"),
                        getattr(t, "head", 0) == 0) for t in dep_ref.tokens],
        "hyp_tokens": [(str(t.text), str(t.upos or "X"),
                        getattr(t, "head", 0) == 0) for t in dep_hyp.tokens],
        "ref_deps":   [(str(dep_ref.tokens[t.head-1].text)
                        if t.head and t.head <= len(dep_ref.tokens) else "ROOT",
                        str(t.text), str(t.deprel or "dep"))
                       for t in dep_ref.tokens if t.head and t.head != 0],
        "hyp_deps":   [(str(dep_hyp.tokens[t.head-1].text)
                        if t.head and t.head <= len(dep_hyp.tokens) else "ROOT",
                        str(t.text), str(t.deprel or "dep"))
                       for t in dep_hyp.tokens if t.head and t.head != 0],
        # DepScore breakdown
        "matched_edges":  matched_edges,
        "unmatched_hyp":  unmatched_hyp,
        "unmatched_ref":  unmatched_ref,
        "sim_total":  round(sim_total, 4),
        "n_hyp_edges": len(h_edges),
        "n_ref_edges": len(r_edges),
        "precision":  round(prec,     4),
        "recall":     round(rec,      4),
        "f_beta":     round(f_beta,   4),
        "sent_cos":   round(sent_cos,  4),
        "W":          W,
        "dep_base":   round(dep_base,  4),
        "vp":         vp,
        # flags
        "passive_mm": passive_mm,
        "neg_mm":     neg_mm,
        "d_passive":  d_passive,
        "d_neg":      d_neg,
        # SynGram breakdown
        "ng_base":    round(ng_base,  4),
        "per_order":  per_order,
        "head_pairs": head_pairs,
        "n_hyp_heads": int(pair_info.get("n_hyp_heads", 0)),
        "n_ref_heads": int(pair_info.get("n_ref_heads", 0)),
        "parse_warnings": _parse_warnings,
    }


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRIS — MT Evaluation",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
<style>
  .metric-box {
    background:#f8f7f2; border:1px solid #d8d5cc;
    border-radius:8px; padding:16px 20px; text-align:center;
  }
  .metric-label {
    font-size:11px; font-weight:600; letter-spacing:.1em;
    text-transform:uppercase; color:#888; margin-bottom:6px;
  }
  .metric-val { font-size:30px; font-weight:700; line-height:1; }
  .metric-sub { font-size:11px; color:#999; margin-top:4px; }
  .parse-token {
    display:inline-block; font-family:monospace; font-size:12px;
    padding:3px 8px; margin:3px; border-radius:4px;
    border:1px solid #d8d5cc; background:#fff; color:#1a1b22 !important;
  }
  .parse-token span { color:#1a1b22 !important; }
  .root-token { border-color:#c8a84b; background:#fffbee; color:#8a5a00; }
  .dep-row { font-family:monospace; font-size:12px; padding:4px 0; }
  .pen-ok   { color:#1e6b3a; background:#eaf3de; padding:3px 10px;
               border-radius:4px; font-size:12px; display:inline-block; }
  .pen-warn { color:#b83232; background:#fcebeb; padding:3px 10px;
               border-radius:4px; font-size:12px; display:inline-block; }
  .step-num {
    display:inline-flex; align-items:center; justify-content:center;
    width:22px; height:22px; border-radius:50%;
    background:#1a1b22; color:#fff; font-size:11px; font-weight:600;
    margin-right:8px;
  }
  .order-box {
    background:#f4f3fd; border:1px solid #c8c5f0;
    border-radius:8px; padding:12px 16px; text-align:center;
  }
  .head-row {
    background:#eef9f5; border:1px solid #9fe1cb;
    border-radius:6px; padding:8px 14px; margin-bottom:6px; font-size:12px;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    lang_label = st.selectbox("Language", list(LANGS.keys()), index=0)
    lang_code  = LANGS[lang_label]
    st.markdown("---")
    st.markdown("**GRIS v3.1** · Tampere University")
    st.markdown("Interpretable MT evaluation via dependency parsing.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:36px;font-weight:800;letter-spacing:-1px;margin-bottom:4px'>
  GRIS <span style='font-size:14px;font-weight:400;color:#888;letter-spacing:.06em'>
  GRAMMATICAL INTERPRETABLE SCORING</span>
</h1>
<p style='color:#888;font-size:14px;margin-bottom:24px'>
  Enter reference and hypothesis translations. GRIS parses both sentences,
  matches dependency edges with the Hungarian algorithm, and shows exactly
  how every score is derived.
</p>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
col_ref, col_hyp = st.columns(2)
with col_ref:
    st.markdown("🟢 **Reference**")
    ref_text = st.text_area("Reference", label_visibility="collapsed",
                             placeholder="The doctor said the patient had fully recovered.",
                             height=100, key="ref")
with col_hyp:
    st.markdown("🔵 **Hypothesis (MT output)**")
    hyp_text = st.text_area("Hypothesis", label_visibility="collapsed",
                             placeholder="The physician confirmed the patient had completely recovered.",
                             height=100, key="hyp")

run = st.button("▶  Evaluate Translation", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
if run:
    if not ref_text.strip() or not hyp_text.strip():
        st.warning("Please enter both reference and hypothesis.")
        st.stop()

    with st.spinner("Parsing and computing metrics…"):
        try:
            R = compute_all(hyp_text, ref_text, lang_code)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    _SC = {"green": "#1e6b3a", "orange": "#8a5a00", "red": "#b83232"}
    _BG = {"green": "#eaf3de", "orange": "#faeeda", "red": "#fcebeb"}

    # ── Score tiles ───────────────────────────────────────────────────────────
    st.markdown("---")
    all_tiles = [
        ("BLEU",          R["bleu"] / 100,  "n-gram precision",      f"{R['bleu']:.1f}"),
        ("chrF",          R["chrf"] / 100,  "char n-gram F",         f"{R['chrf']:.1f}"),
        ("BERTScore F1",  R["bertscore"],   "contextual token sim",  f"{R['bertscore']:.3f}"),
        ("GRIS-DepScore", R["dep_score"],   "edge Fβ(1.5) + blend",  f"{R['dep_score']:.3f}"),
        ("GRIS-SynGram",  R["ng_score"],    "subtree path n-grams",  f"{R['ng_score']:.3f}"),
    ]
    cols = st.columns(len(all_tiles))
    for col, (label, val_norm, sub, display) in zip(cols, all_tiles):
        c = score_color(val_norm)
        col.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>{label}</div>
          <div class='metric-val' style='color:{_SC[c]}'>{display}</div>
          <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Overall quality panel ─────────────────────────────────────────────────
    st.markdown("---")
    dep = R["dep_score"]; ng = R["ng_score"]; bert = R["bertscore"]
    avg         = dep * 0.45 + ng * 0.30 + bert * 0.25
    blend_label = "DepScore×0.45 + SynGram×0.30 + BERTScore×0.25"

    if avg >= 0.88:   qlabel, qcolor = "Excellent", "#1e6b3a";  qverdict = "Very close to reference in meaning and structure."
    elif avg >= 0.78: qlabel, qcolor = "Good",      "#2a5a8a";  qverdict = "Captures main meaning with minor structural differences."
    elif avg >= 0.65: qlabel, qcolor = "Acceptable","#8a5a00";  qverdict = "Conveys general meaning but has notable differences."
    elif avg >= 0.50: qlabel, qcolor = "Poor",      "#b05000";  qverdict = "Significant errors in meaning or structure."
    else:             qlabel, qcolor = "Bad",        "#b83232"; qverdict = "Fails to convey the reference meaning."

    _good, _bad = [], []
    if bert >= 0.90:   _good.append("Semantic meaning well preserved (BERTScore)")
    elif bert >= 0.80: _good.append("Semantic meaning mostly preserved")
    else:              _bad.append("Semantic meaning differs from reference")
    if dep >= 0.85:    _good.append("Syntactic structure closely matches (DepScore)")
    elif dep >= 0.70:  _bad.append("Some syntactic structural differences")
    else:              _bad.append("Significant syntactic restructuring")
    if ng >= 0.85:     _good.append("Predicate-argument structure preserved (SynGram)")
    elif ng >= 0.70:   _bad.append("Minor predicate structure differences")
    else:              _bad.append("Predicate-argument structure differs")
    if R["passive_mm"]: _bad.append("Voice mismatch (active vs passive)")
    if R["neg_mm"]:     _bad.append("Negation mismatch — truth condition changed")

    st.markdown(f"""
    <div style='border:2px solid {qcolor};border-radius:10px;padding:18px 24px;margin-bottom:8px'>
      <div style='display:flex;align-items:center;gap:16px;margin-bottom:12px'>
        <div style='font-size:28px;font-weight:800;color:{qcolor}'>{qlabel}</div>
        <div style='font-size:13px;color:#888'>{qverdict}</div>
      </div>
      <div style='background:#f0f0f0;border-radius:6px;height:10px;width:100%;margin-bottom:14px'>
        <div style='background:{qcolor};border-radius:6px;height:10px;width:{int(avg*100)}%'></div>
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px'>
        <div>
          {"".join(f"<div style='margin-bottom:5px'>✅ {g}</div>" for g in _good)
           or "<div style='color:#888'>—</div>"}
        </div>
        <div>
          {"".join(f"<div style='margin-bottom:5px;color:#b83232'>⚠ {b}</div>" for b in _bad)
           or "<div style='color:#888'>No issues detected</div>"}
        </div>
      </div>
      <div style='margin-top:12px;font-size:11px;color:#aaa'>
        Score = {blend_label} = {avg:.3f}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Dependency parse trees ────────────────────────────────────────
    st.markdown("---")
    if R.get("parse_warnings"):
        st.error("⚠️ **Parse quality warning — " +
                 " | ".join(R["parse_warnings"]) + "**\n\n"
                 f"Run `import stanza; stanza.download('{lang_code}')` then restart.")

    st.markdown("<span class='step-num'>1</span> **Dependency Parse Trees**",
                unsafe_allow_html=True)
    pc_r, pc_h = st.columns(2)

    def render_parse(col, label, tokens, deps, color):
        with col:
            st.markdown(f"**{label}**")
            tok_html = ""
            for text, upos, is_root in tokens:
                cls = "root-token" if is_root else "parse-token"
                tok_html += (f"<span class='{cls}'>{text} "
                             f"<span style='font-size:9px;opacity:.6'>{upos}</span></span>")
            st.markdown(tok_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for head, dep, rel in deps:
                st.markdown(
                    f"<div class='dep-row'>"
                    f"<span style='color:{color};min-width:80px;display:inline-block'>{head}</span>"
                    f" ──▶ "
                    f"<span style='min-width:80px;display:inline-block'>{dep}</span>"
                    f"<span style='background:#f0f0f0;padding:1px 7px;border-radius:3px;"
                    f"font-size:10px;color:#555;margin-left:8px'>{rel}</span>"
                    f"</div>", unsafe_allow_html=True)

    render_parse(pc_r, "🟢 Reference",  R["ref_tokens"], R["ref_deps"], "#1a6b4a")
    render_parse(pc_h, "🔵 Hypothesis", R["hyp_tokens"], R["hyp_deps"], "#1a3a6b")

    # ── Step 2: GRIS-DepScore walkthrough ─────────────────────────────────────
    st.markdown("---")
    st.markdown("<span class='step-num'>2</span> **GRIS-DepScore — Score Derivation**",
                unsafe_allow_html=True)

    with st.expander("Formula (v2.4)", expanded=False):
        st.code(f"""
# Edge similarity — multiplicative (v2.4)
dep_sim  = cos01(embed(dep_hyp),  embed(dep_ref))   # primary signal
head_sim = cos01(embed(head_hyp), embed(head_ref))  # confidence multiplier
edge_sim = dep_sim × (1 + head_sim) / 2

# Optimal 1-to-1 matching
matching = Hungarian(−sim_matrix)

# Recall-weighted Fβ(1.5)
P      = Σ matched_sim / |hyp_edges|
R      = Σ matched_sim / |ref_edges|
F_beta = (1 + 1.5²) × P × R / (1.5² × P + R)
F_beta *= 0.92  if |hyp| > 1.4×|ref|   # verbosity penalty

# Language-adaptive W blend  (W = {R['W']:.2f} for lang={lang_code})
score  = {R['W']:.2f} × F_beta + {1-R['W']:.2f} × sent_cos

# Global mismatch penalties (WMT22 final)
dep_score = score × neg_pen  # passive NOT penalized (×1.0)
        """, language="python")

    # Matched edges table
    st.markdown("**Matched edge pairs** (Hungarian optimal assignment)")
    if R["matched_edges"]:
        import pandas as pd
        df_edges = pd.DataFrame([{
            "Hyp edge":   e["hyp"],
            "Ref edge":   e["ref"],
            "Relation":   e["rel"],
            "Similarity": e["sim"],
            "Note":       e["note"],
        } for e in R["matched_edges"]])
        st.dataframe(
            df_edges.style
            .background_gradient(subset=["Similarity"], cmap="RdYlGn", vmin=0, vmax=1)
            .format({"Similarity": "{:.3f}"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("No content edges matched.")

    if R["unmatched_hyp"]:
        st.markdown(f"**Extra hyp edges** *(precision penalty)*: "
                    f"`{'`, `'.join(R['unmatched_hyp'])}`")
    if R["unmatched_ref"]:
        st.markdown(f"**Missing ref edges** *(recall penalty)*: "
                    f"`{'`, `'.join(R['unmatched_ref'])}`")

    # Edge interpretation
    st.markdown("**Edge-level interpretation**")
    CORE_RELS = {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "nsubj:pass"}
    REL_LABELS = {
        "nsubj": "Subject", "nsubj:pass": "Passive subj", "obj": "Object",
        "iobj": "Indirect obj", "advmod": "Adv. modifier", "amod": "Adj. modifier",
        "obl": "Oblique arg", "aux": "Auxiliary", "aux:pass": "Passive aux",
        "dep": "Dependency",
    }

    ihtml = "<div style='display:flex;flex-direction:column;gap:6px;margin-bottom:8px'>"
    for e in sorted(R["matched_edges"], key=lambda x: -x["sim"]):
        rel = e["rel"]; sim = e["sim"]
        is_core = rel.lower() in CORE_RELS
        if sim >= 0.80:   bg, fg, icon, q = "#eaf3de", "#1e6b3a", "✓", "well matched"
        elif sim >= 0.55: bg, fg, icon, q = "#faeeda", "#8a5a00", "~", "partial match"
        else:             bg, fg, icon, q = "#fcebeb", "#b83232", "✗", "weak match"
        tier = "Core argument" if is_core else "Modifier"
        note = f" — {e['note']}" if e["note"] else ""
        ihtml += (
            f"<div style='background:{bg};border-radius:6px;padding:10px 14px;"
            f"display:flex;align-items:flex-start;gap:12px'>"
            f"<span style='font-size:16px;color:{fg};font-weight:700;min-width:20px'>{icon}</span>"
            f"<div><div style='font-size:13px;font-weight:600;color:{fg}'>"
            f"{REL_LABELS.get(rel.lower(), rel)} ({rel}) — {q} "
            f"<span style='font-size:11px;font-weight:400;opacity:.7'>{tier}{note}</span></div>"
            f"<div style='font-size:12px;color:{fg};opacity:.85;margin-top:3px;"
            f"font-family:monospace'>{e['hyp']} ↔ {e['ref']} &nbsp; sim={sim:.3f}"
            f"</div></div></div>")
    for ue in R["unmatched_ref"]:
        ihtml += (
            f"<div style='background:#fcebeb;border-radius:6px;padding:10px 14px;"
            f"display:flex;gap:12px'><span style='font-size:16px;color:#b83232;"
            f"font-weight:700'>!</span><div>"
            f"<div style='font-size:13px;font-weight:600;color:#b83232'>"
            f"Missing from hypothesis — recall penalty</div>"
            f"<div style='font-family:monospace;font-size:12px;color:#b83232'>{ue}</div>"
            f"</div></div>")
    for ue in R["unmatched_hyp"]:
        ihtml += (
            f"<div style='background:#faeeda;border-radius:6px;padding:10px 14px;"
            f"display:flex;gap:12px'><span style='font-size:16px;color:#8a5a00;"
            f"font-weight:700'>+</span><div>"
            f"<div style='font-size:13px;font-weight:600;color:#8a5a00'>"
            f"Extra in hypothesis — precision penalty</div>"
            f"<div style='font-family:monospace;font-size:12px;color:#8a5a00'>{ue}</div>"
            f"</div></div>")
    ihtml += "</div>"
    st.markdown(ihtml, unsafe_allow_html=True)


    # ── Severity classification (MQM-aligned) ────────────────────────────
    st.markdown("**Error severity classification** (MQM-aligned)")

    CORE_RELS_SET = {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "nsubj:pass"}

    def classify_edge_severity(edge, is_core):
        """Classify a matched edge by MQM severity."""
        sim = edge["sim"]
        has_rel_mismatch = bool(edge.get("note"))
        if is_core:
            if sim < 0.30:                          return "CRITICAL", 25, "Core argument completely wrong"
            elif sim < 0.60 or has_rel_mismatch:   return "MAJOR",    5,  "Core argument error or role mismatch"
            elif sim < 0.80:                        return "MINOR",    1,  "Core argument partial match"
            else:                                   return "OK",       0,  "Correctly translated"
        else:
            if sim < 0.45:                          return "MAJOR",    5,  "Modifier significantly wrong"
            elif sim < 0.65 or has_rel_mismatch:   return "MINOR",    1,  "Modifier partial match or label mismatch"
            else:                                   return "OK",       0,  "Correctly translated"

    SEV_STYLE = {
        "CRITICAL": ("#b83232", "#fcebeb", "●"),
        "MAJOR":    ("#8a5a00", "#faeeda", "◆"),
        "MINOR":    ("#185FA5", "#E6F1FB", "▲"),
        "OK":       ("#1e6b3a", "#eaf3de", "✓"),
    }

    # Classify matched edges
    edge_severities = []
    for e in R["matched_edges"]:
        is_core = e["rel"].lower() in CORE_RELS_SET
        sev, weight, reason = classify_edge_severity(e, is_core)
        edge_severities.append((e, sev, weight, reason, is_core))

    # Classify unmatched reference edges (missing content)
    unmatched_ref_sev = []
    for ue in R["unmatched_ref"]:
        rel_part = ue.split("→")[-1] if "→" in ue else ""
        is_core_u = any(cr in rel_part.lower() for cr in ["nsubj","obj","iobj","csubj"])
        is_aux    = "aux" in rel_part.lower()
        if is_core_u:      sev, weight, reason = "CRITICAL", 25, "Core argument missing from hypothesis"
        elif is_aux:       sev, weight, reason = "MAJOR",    5,  "Modal auxiliary missing"
        else:              sev, weight, reason = "MINOR",    1,  "Modifier missing from hypothesis"
        unmatched_ref_sev.append((ue, sev, weight, reason))

    # Sentence-level flag severities
    # WMT22 FINAL CODE: Only negation penalty applied (×0.90)
    # Passive flagged but NOT penalized (considered valid paraphrase)
    flag_sevs = []
    if R["neg_mm"]:     
        flag_sevs.append(("Negation mismatch", "CRITICAL", 25, "Truth condition reversed (×0.90 penalty applied)"))
    if R["passive_mm"]: 
        flag_sevs.append(("Passive voice mismatch", "INFO", 0, "Flagged only (no penalty in WMT22 final)"))

    # Compute estimated MQM score
    total_penalty = 0
    counts = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "OK": 0}
    for _, sev, weight, _, _ in edge_severities:
        total_penalty += weight; counts[sev] += 1
    for _, sev, weight, _ in unmatched_ref_sev:
        total_penalty += weight; counts[sev] += 1
    for _, sev, weight, _ in flag_sevs:
        total_penalty += weight; counts[sev] += 1
    estimated_mqm = -total_penalty

    # Summary bar
    sev_html = "<div style='display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap'>"
    for sev_label, cnt_key, bg, fg in [
        ("CRITICAL", "CRITICAL", "#fcebeb", "#b83232"),
        ("MAJOR",    "MAJOR",    "#faeeda", "#8a5a00"),
        ("MINOR",    "MINOR",    "#E6F1FB", "#185FA5"),
        ("OK",       "OK",       "#eaf3de", "#1e6b3a"),
    ]:
        n = counts[cnt_key]
        opacity = "1.0" if n > 0 else "0.35"
        sev_html += (
            f"<div style='background:{bg};border:1px solid {fg};border-radius:8px;"
            f"padding:8px 18px;text-align:center;opacity:{opacity}'>"
            f"<div style='font-size:20px;font-weight:800;color:{fg}'>{n}</div>"
            f"<div style='font-size:10px;font-weight:600;color:{fg};letter-spacing:.06em'>{sev_label}</div>"
            f"</div>"
        )
    mqm_col = "#1e6b3a" if estimated_mqm >= -2 else ("#8a5a00" if estimated_mqm >= -10 else "#b83232")
    sev_html += (
        f"<div style='margin-left:auto;background:#f8f7f2;border:2px solid {mqm_col};"
        f"border-radius:8px;padding:8px 20px;text-align:center'>"
        f"<div style='font-size:22px;font-weight:800;color:{mqm_col}'>{estimated_mqm}</div>"
        f"<div style='font-size:10px;font-weight:600;color:#888'>Est. MQM score</div>"
        f"<div style='font-size:9px;color:#aaa'>−(25×crit + 5×maj + 1×min)</div>"
        f"</div>"
    )
    sev_html += "</div>"
    st.markdown(sev_html, unsafe_allow_html=True)

    # Detailed severity cards
    sev_detail_html = "<div style='display:flex;flex-direction:column;gap:6px;margin-bottom:10px'>"

    # Sentence-level flags first (highest severity)
    for flag_name, sev, weight, reason in flag_sevs:
        fg, bg, icon = SEV_STYLE[sev]
        sev_detail_html += (
            f"<div style='background:{bg};border-radius:6px;padding:8px 14px;"
            f"display:flex;align-items:center;gap:12px'>"
            f"<span style='background:{fg};color:white;font-size:9px;font-weight:700;"
            f"padding:2px 7px;border-radius:4px;white-space:nowrap;min-width:58px;text-align:center'>"
            f"{sev} ×{weight}</span>"
            f"<span style='font-size:13px;font-weight:600;color:{fg}'>{flag_name}</span>"
            f"<span style='font-size:11px;color:{fg};opacity:.8;margin-left:4px'>— {reason}</span>"
            f"</div>"
        )

    # Matched edge cards by severity (critical first)
    for sev_order in ["CRITICAL", "MAJOR", "MINOR"]:
        for e, sev, weight, reason, is_core in edge_severities:
            if sev != sev_order: continue
            fg, bg, icon = SEV_STYLE[sev]
            tier = "Core argument" if is_core else "Modifier"
            sev_detail_html += (
                f"<div style='background:{bg};border-radius:6px;padding:8px 14px;"
                f"display:flex;align-items:center;gap:12px'>"
                f"<span style='background:{fg};color:white;font-size:9px;font-weight:700;"
                f"padding:2px 7px;border-radius:4px;white-space:nowrap;min-width:58px;text-align:center'>"
                f"{sev} ×{weight}</span>"
                f"<div style='flex:1'>"
                f"<span style='font-size:12px;font-weight:600;color:{fg}'>{reason}</span>"
                f"<span style='font-size:10px;color:{fg};opacity:.7;margin-left:8px'>({tier} · rel: {e['rel']} · sim={e['sim']:.3f})</span><br>"
                f"<span style='font-family:monospace;font-size:11px;color:{fg};opacity:.85'>"
                f"{e['hyp']} ↔ {e['ref']}"
                f"{'  — ' + e.get('note', '') if e.get('note') else ''}</span>"
                f"</div></div>"
            )

    # Unmatched reference edges
    for ue, sev, weight, reason in unmatched_ref_sev:
        fg, bg, icon = SEV_STYLE[sev]
        sev_detail_html += (
            f"<div style='background:{bg};border-radius:6px;padding:8px 14px;"
            f"display:flex;align-items:center;gap:12px'>"
            f"<span style='background:{fg};color:white;font-size:9px;font-weight:700;"
            f"padding:2px 7px;border-radius:4px;white-space:nowrap;min-width:58px;text-align:center'>"
            f"{sev} ×{weight}</span>"
            f"<div style='flex:1'>"
            f"<span style='font-size:12px;font-weight:600;color:{fg}'>{reason}</span><br>"
            f"<span style='font-family:monospace;font-size:11px;color:{fg};opacity:.85'>"
            f"Missing: {ue}</span>"
            f"</div></div>"
        )

    sev_detail_html += "</div>"

    # Only show details if there are any errors
    if any(s != "OK" for _, s, _, _, _ in edge_severities) or unmatched_ref_sev or flag_sevs:
        st.markdown(sev_detail_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#eaf3de;border-radius:6px;padding:10px 14px;"
            "font-size:13px;color:#1e6b3a;font-weight:600'>"
            "✓ No errors detected — translation closely matches reference.</div>",
            unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:10px;color:#aaa;margin-top:4px'>"
        "Severity aligned with MQM framework: Critical=25pts, Major=5pts, Minor=1pt. "
        "Estimated MQM score is indicative only — professional annotation may differ.</div>",
        unsafe_allow_html=True)


    # Penalty flags
    st.markdown("**Mismatch flags**")
    fp1, fp2 = st.columns(2)
    for col, flag, label, val in [
        (fp1, R["passive_mm"], "passive",  R["d_passive"]),
        (fp2, R["neg_mm"],     "negation", R["d_neg"]),
    ]:
        cls = "pen-warn" if flag else "pen-ok"
        icon = "⚠" if flag else "✓"
        col.markdown(
            f"<span class='{cls}'>{icon} {label} → ×{val}</span>",
            unsafe_allow_html=True)

    # Step-by-step
    st.markdown("**Step-by-step derivation**")
    for n, title, calc in [
        ("1", "Sum of matched similarities",
         f"{' + '.join(str(e['sim']) for e in R['matched_edges'])} = **{R['sim_total']}**"),
        ("2", "Precision = sim_total ÷ |hyp edges|",
         f"{R['sim_total']} ÷ {R['n_hyp_edges']} = **{R['precision']}**"),
        ("3", "Recall = sim_total ÷ |ref edges|",
         f"{R['sim_total']} ÷ {R['n_ref_edges']} = **{R['recall']}**"),
        ("4", "F_β(1.5) = (1+1.5²)×P×R / (1.5²×P+R)",
         f"= **{R['f_beta']}**" +
         (f" × 0.92 (verbosity)" if R["vp"] < 1 else "")),
        ("5", f"W blend  ({R['W']}×F_β + {1-R['W']:.2f}×sent_cos)",
         f"{R['W']}×{R['f_beta']} + {1-R['W']:.2f}×{R['sent_cos']} = **{R['dep_base']}**"),
        ("6", "Global penalties",
         f"{R['dep_base']} × {R['d_passive']} × {R['d_neg']} = **{R['dep_score']}**"),
    ]:
        c1, c2, c3 = st.columns([0.4, 2.5, 3])
        c1.markdown(f"<span class='step-num'>{n}</span>", unsafe_allow_html=True)
        c2.markdown(f"**{title}**")
        c3.markdown(calc)

    dep_color = _SC[score_color(R["dep_score"])]
    st.markdown(f"""
    <div style='border:2px solid #1a1b22;border-radius:8px;padding:16px 24px;
                display:flex;align-items:center;justify-content:space-between;margin-top:12px'>
      <div>
        <div style='font-size:11px;font-weight:600;letter-spacing:.1em;
                    text-transform:uppercase;color:#888;margin-bottom:4px'>GRIS-DepScore</div>
        <div style='font-size:44px;font-weight:800;color:{dep_color}'>{R['dep_score']:.4f}</div>
      </div>
      <div style='font-family:monospace;font-size:12px;color:#888;text-align:right;line-height:2'>
        P={R['precision']} &nbsp; R={R['recall']}<br>
        F_β(1.5) = {R['f_beta']}<br>
        W={R['W']} blend → {R['dep_base']}<br>
        × passive({R['d_passive']}) × neg({R['d_neg']})<br>
        <strong style='color:#1a1b22'>= {R['dep_score']:.4f}</strong>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Step 3: GRIS-SynGram walkthrough ─────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<span class='step-num'>3</span> **GRIS-SynGram — Score Derivation**",
        unsafe_allow_html=True)

    with st.expander("Formula (v8 — subtree path n-grams)", expanded=False):
        st.code("""
# For each content head v — depth-decayed importance w(v) = 1/(1+depth):
#   Traverse dependency subtree up to depth δ=3
#   Each upward path [leaf→…→v] of length k = k-gram pattern
#   Pattern = UPOS[sorted(collapse(deprel):UPOS)]
#   e.g.  VERB[ARG:NOUN, advmod:ADV]   (nsubj/obj collapsed to ARG)

# N-gram similarity (anchor head embedding as primary signal):
#   base_sim      = cos01(embed(head_hyp), embed(head_ref))
#   penalised_sim = base_sim × (1 − min(0.85, arg_pen + neg_pen))
#   sim_ngram     = min(1.0, penalised_sim + exact_lemma_bonus)

# Per-order F1 with adaptive threshold: τ_n = max(0.35, 0.55 − 0.02×n)
# Bigram-dominant order weights: [0.25, 0.40, 0.25, 0.10]

# Head matching: Hungarian on importance-weighted cosine matrix
# Unmatched ref heads at depth≥2 → 20% weight only (depth gate)

# Final: base_score × neg_pen  # passive NOT penalized in GRIS-SynGram
# (voice normalization at extraction makes passive penalty redundant)
        """, language="python")

    # Per-order F1 breakdown
    st.markdown("**Per-order F1** (n=1 unigram → n=4 four-gram)")
    order_weights = [0.25, 0.40, 0.25, 0.10]
    per_order = R.get("per_order", {})
    oc = st.columns(4)
    for i, (col, w) in enumerate(zip(oc, order_weights)):
        n = i + 1
        f1v = per_order.get(n, per_order.get(str(n), None))
        if f1v is not None:
            c = score_color(float(f1v))
            col.markdown(f"""
            <div class='order-box'>
              <div style='font-size:10px;color:#888;font-weight:600'>n={n} · w={w}</div>
              <div style='font-size:24px;font-weight:700;color:{_SC[c]}'>{float(f1v):.3f}</div>
              <div style='font-size:9px;color:#aaa'>
                {"unigram (head cos)" if n==1 else "bigram ↑ wt" if n==2 else f"{n}-gram"}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            col.markdown(f"""
            <div class='order-box'>
              <div style='font-size:10px;color:#888;font-weight:600'>n={n} · w={w}</div>
              <div style='font-size:20px;font-weight:700;color:#aaa'>—</div>
              <div style='font-size:9px;color:#aaa'>no patterns</div>
            </div>""", unsafe_allow_html=True)

    # Per-head matched pairs
    head_pairs = R.get("head_pairs", [])
    if head_pairs:
        st.markdown("**Matched head pairs** (Hungarian head assignment)")
        for hp in head_pairs:
            hh = hp.get("hyp_head", "?"); rh = hp.get("ref_head", "?")
            sc = hp.get("score", 0.0);    w  = hp.get("w", 0.0)
            dh = hp.get("depth_hyp", "?"); dr = hp.get("depth_ref", "?")
            c  = score_color(float(sc))
            st.markdown(
                f"<div class='head-row'>"
                f"<strong style='color:{_SC[c]}'>{hh}</strong> (depth {dh}) "
                f"↔ <strong style='color:{_SC[c]}'>{rh}</strong> (depth {dr}) "
                f"&nbsp;·&nbsp; head_score={float(sc):.3f} "
                f"&nbsp;·&nbsp; pair_w={float(w):.3f}</div>",
                unsafe_allow_html=True)

    # SynGram interpretation
    st.markdown("**What this score means — SynGram interpretation**")
    ng_b = R["ng_base"]
    ng_interp = []

    if ng_b >= 0.85:
        ng_interp.append(("✓", "#1e6b3a", "#eaf3de", "Strong pattern match",
            "Hypothesis subtree paths closely match reference — predicate-argument structure preserved."))
    elif ng_b >= 0.65:
        ng_interp.append(("~", "#8a5a00", "#faeeda", "Partial pattern match",
            "Some predicate-argument structures match; others differ in argument count or modifier depth."))
    else:
        ng_interp.append(("✗", "#b83232", "#fcebeb", "Weak pattern match",
            "Hypothesis predicate-argument structure differs substantially from reference."))

    if per_order:
        n2 = float(per_order.get(2, per_order.get("2", 0)))
        n1 = float(per_order.get(1, per_order.get("1", 0)))
        if n2 > 0.75:
            ng_interp.append(("✓", "#1e6b3a", "#eaf3de", "Dependency edges preserved",
                f"n=2 F1={n2:.3f} — head+dependent pairs match well. "
                f"This is the most informative order for structural accuracy."))
        elif n2 < n1 - 0.15:
            ng_interp.append(("⚠", "#8a5a00", "#faeeda", "Bigram drop detected",
                f"n=2 (F1={n2:.3f}) significantly lower than n=1 (F1={n1:.3f}) — "
                f"hypothesis uses similar words but wrong argument structure."))

    if R["neg_mm"]:
        ng_interp.append(("!", "#b83232", "#fcebeb", "Negation pattern mismatch",
            f"neg:PART detected in one sentence only — edge penalty applied inside "
            f"pattern, plus sentence-level ×{NEG_PEN}."))
    if R["passive_mm"]:
        ng_interp.append(("~", "#8a5a00", "#faeeda", "Voice normalisation applied",
            "nsubj:pass→obj and obl:agent→nsubj applied before pattern matching. "
            f"No sentence penalty (voice normalization makes passive penalty redundant)."))

    for icon, fg, bg, title, msg in ng_interp:
        st.markdown(
            f"<div style='background:{bg};border-radius:6px;padding:10px 14px;"
            f"display:flex;gap:12px;margin-bottom:6px'>"
            f"<span style='font-size:16px;color:{fg};font-weight:700;min-width:20px'>{icon}</span>"
            f"<div><div style='font-size:13px;font-weight:600;color:{fg}'>{title}</div>"
            f"<div style='font-size:12px;color:{fg};opacity:.85;margin-top:3px'>{msg}</div>"
            f"</div></div>", unsafe_allow_html=True)

    # Step-by-step SynGram
    st.markdown("**Step-by-step derivation**")
    for n, title, calc in [
        ("1", "Extract subtree path n-grams per content head",
         f"HYP: {R['n_hyp_heads']} heads · REF: {R['n_ref_heads']} heads · "
         f"depth δ=3 · AUX modals included"),
        ("2", "Hungarian head matching (depth-decayed importance weights)",
         "w(v) = 1/(1+depth) → root=1.0, depth-1=0.500, depth-2=0.333 …"),
        ("3", "Greedy n-gram matching per order (τ₀=0.55, adaptive)",
         "Order weights [0.25, 0.40, 0.25, 0.10] — bigrams (n=2) dominant"),
        ("4", "Base score (importance-weighted head scores / denominator)",
         f"= **{R['ng_base']:.4f}**"),
        ("5", "Global mismatch penalties",
         f"{R['ng_base']:.4f} × {R['d_passive']} × {R['d_neg']} = **{R['ng_score']:.4f}**"),
    ]:
        c1, c2, c3 = st.columns([0.4, 2.5, 3])
        c1.markdown(f"<span class='step-num'>{n}</span>", unsafe_allow_html=True)
        c2.markdown(f"**{title}**")
        c3.markdown(calc)

    ng_color = _SC[score_color(R["ng_score"])]
    order_detail = (" · ".join(
        f"n={n}:{float(per_order.get(n, per_order.get(str(n), 0))):.3f}"
        for n in range(1, 5)) if per_order else "per-order detail unavailable")
    st.markdown(f"""
    <div style='border:2px solid #1a6b4a;border-radius:8px;padding:16px 24px;
                display:flex;align-items:center;justify-content:space-between;margin-top:12px'>
      <div>
        <div style='font-size:11px;font-weight:600;letter-spacing:.1em;
                    text-transform:uppercase;color:#888;margin-bottom:4px'>GRIS-SynGram</div>
        <div style='font-size:44px;font-weight:800;color:{ng_color}'>{R['ng_score']:.4f}</div>
      </div>
      <div style='font-family:monospace;font-size:12px;color:#888;text-align:right;line-height:2'>
        base_score = {R['ng_base']:.4f}<br>
        {order_detail}<br>
        × passive({R['d_passive']}) × neg({R['d_neg']})<br>
        <strong style='color:#1a1b22'>= {R['ng_score']:.4f}</strong>
      </div>
    </div>""", unsafe_allow_html=True)

else:
    st.info("Enter reference and hypothesis above, then click **Evaluate Translation**.")