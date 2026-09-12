"""
IndiaWelfare-606 — Reproduce the paper's main tables from released files.

Requires:
    data/gold_eval_pairs.csv          (from the HuggingFace dataset)
    results/*.csv                     (shipped in this repository)

    pip install pandas numpy scikit-learn scipy statsmodels

Run:
    python3 analysis.py

Reproduces (paper reference in brackets):
    [T-B]  Protocol B main results + 95% bootstrap CIs
    [AUC]  Threshold-free ROC-AUC / PR-AUC
    [McN]  McNemar significance tests
    [CAT]  Per-category macro F1 (Protocol-B systems + lenient RAG)
    [SUP]  Supervised DeBERTa (pair-level + citizen-grouped, consensus)
    [J2 ]  Task 2 judged-subset re-ranking + Wilcoxon tests
    [ERR]  Error composition (FN/FP per system)
    [SCL]  Scale summary (Llama-3.1 8B vs 70B)
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             cohen_kappa_score, f1_score, roc_auc_score,
                             average_precision_score)

TFIDF_THRESHOLD, BGE_THRESHOLD = 0.03, 0.62   # tuned once on the dev split
SPLIT_SEED = 42

# ── Load ────────────────────────────────────────────────────
gold_df = pd.read_csv("data/gold_eval_pairs.csv")
tf      = pd.read_csv("results/tfidf_raw_scores.csv")
bg      = pd.read_csv("results/bge_raw_scores.csv")
rag     = pd.read_csv("results/rag_lenient_predictions.csv")
l70     = pd.read_csv("results/llama70b_predictions.csv")
oof     = pd.read_csv("results/deberta_oof_predictions.csv")
goof    = pd.read_csv("results/deberta_grouped_oof.csv")

gold = (gold_df['gold_label'] == 'eligible').astype(int).values
assert len(gold) == 521

# Recreate the fixed dev/test split (Protocol B)
np.random.seed(SPLIT_SEED)
idx = np.random.permutation(len(gold))
test_idx = idx[int(0.2 * len(gold)):]
g_test = gold[test_idx]

tf_pred = (tf['tfidf_score'].values >= TFIDF_THRESHOLD).astype(int)
bg_pred = (bg['bge_score'].values >= BGE_THRESHOLD).astype(int)
rag_map = rag['rag_lenient_pred'].map(
    {'eligible': 1, 'not eligible': 0, 'unknown': -1}).values


def metrics(g, p):
    pr, re_, f1b, _ = precision_recall_fscore_support(
        g, p, average='binary', zero_division=0)
    return dict(acc=accuracy_score(g, p), prec=pr, rec=re_, f1b=f1b,
                f1m=f1_score(g, p, average='macro', zero_division=0),
                kappa=cohen_kappa_score(g, p))


def bootstrap_ci(g, p, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    g, p = np.asarray(g), np.asarray(p)
    f1s = [f1_score(g[i], p[i], average='macro', zero_division=0)
           for i in (rng.choice(len(g), len(g), True) for _ in range(n))]
    return np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)


def mcn(g, a, b):
    ca, cb = np.asarray(a) == np.asarray(g), np.asarray(b) == np.asarray(g)
    tbl = [[int((ca & cb).sum()), int((ca & ~cb).sum())],
           [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
    r = mcnemar(tbl, exact=False, correction=True)
    return r.statistic, r.pvalue


# ── [T-B] Protocol B main results ───────────────────────────
print("=" * 72)
print("[T-B] Protocol B (417-pair test split, frozen thresholds)")
maj = np.zeros(len(g_test), int)
for name, p in [("Majority class", maj),
                ("TF-IDF similarity", tf_pred[test_idx]),
                ("BGE-M3 dense", bg_pred[test_idx])]:
    m = metrics(g_test, p)
    lo, hi = bootstrap_ci(g_test, p)
    print(f"  {name:<20} Acc={m['acc']:.4f} F1mac={m['f1m']:.4f} "
          f"[{lo:.3f},{hi:.3f}]  kappa={m['kappa']:.4f}")

# ── [AUC] ───────────────────────────────────────────────────
print("\n[AUC] Threshold-free discrimination (test split)")
for name, s in [("TF-IDF", tf['tfidf_score'].values[test_idx]),
                ("BGE-M3", bg['bge_score'].values[test_idx])]:
    print(f"  {name:<8} ROC-AUC={roc_auc_score(g_test, s):.4f}  "
          f"PR-AUC={average_precision_score(g_test, s):.4f}  "
          f"(prevalence floor {g_test.mean():.4f})")

# ── [McN] ───────────────────────────────────────────────────
print("\n[McN] McNemar tests")
for label, a, b, g in [
        ("TF-IDF vs Majority", tf_pred[test_idx], maj, g_test),
        ("BGE-M3 vs Majority", bg_pred[test_idx], maj, g_test),
        ("TF-IDF vs BGE-M3", tf_pred[test_idx], bg_pred[test_idx], g_test)]:
    chi2, p = mcn(g, a, b)
    print(f"  {label:<22} chi2={chi2:7.3f}  p={p:.4g}")

# ── [CAT] Per-category macro F1 ─────────────────────────────
print("\n[CAT] Per-category macro F1 (full set)")
cats = gold_df['category'].values
for cat in sorted(gold_df['category'].unique()):
    m = cats == cat
    lm = rag_map[m] != -1
    lf1 = (f1_score(gold[m][lm], rag_map[m][lm], average='macro',
                    zero_division=0) if lm.sum() else float('nan'))
    print(f"  {cat:<38} n={m.sum():>3} "
          f"TFIDF={f1_score(gold[m], tf_pred[m], average='macro', zero_division=0):.3f} "
          f"BGE={f1_score(gold[m], bg_pred[m], average='macro', zero_division=0):.3f} "
          f"RAG-L={lf1:.3f}")

# ── [SUP] Supervised DeBERTa ────────────────────────────────
print("\n[SUP] DeBERTa-v3 (out-of-fold consensus over 3 seeds)")
for label, frame in [("pair-level", oof), ("citizen-grouped", goof)]:
    cons = ((frame['pred_seed42'] + frame['pred_seed7'] +
             frame['pred_seed2025']) >= 2).astype(int).values
    m = metrics(frame['gold'].values, cons)
    print(f"  {label:<16} Acc={m['acc']:.4f} F1mac={m['f1m']:.4f} "
          f"kappa={m['kappa']:.4f}  FN={int(((cons==0)&(frame['gold']==1)).sum())} "
          f"FP={int(((cons==1)&(frame['gold']==0)).sum())}")

db = ((oof['pred_seed42'] + oof['pred_seed7'] + oof['pred_seed2025']) >= 2
      ).astype(int).values
l70fs = l70['llama70b_fewshot'].map(
    {'eligible': 1, 'not eligible': 0, 'unknown': 0}).values
chi2, p = mcn(gold, db, l70fs)
print(f"  McNemar DeBERTa vs 70B few-shot: chi2={chi2:.3f} p={p:.2g}")
ragb = np.where(rag_map == -1, 0, rag_map)
chi2, p = mcn(gold, db, ragb)
print(f"  McNemar DeBERTa vs RAG-lenient:  chi2={chi2:.3f} p={p:.2g}")

# ── [J2] Task 2 judged-subset re-ranking ────────────────────
print("\n[J2] Judged-subset re-ranking (citizens with >=1 eligible)")
merged = gold_df.copy()
merged['tf'] = tf['tfidf_score'].values
merged['bg'] = bg['bge_score'].values
merged['rg'] = np.where(rag_map == -1, 0, rag_map)
merged['gold_int'] = gold


def rank_eval(col, seed=None):
    per = {}
    for cid, grp in merged.groupby('citizen_id'):
        if len(grp) < 2:
            continue
        if seed is not None:
            rng = np.random.RandomState(seed + hash(cid) % 10000)
            grp = grp.assign(_s=rng.rand(len(grp))).sort_values('_s', ascending=False)
        else:
            grp = grp.sort_values(col, ascending=False, kind='stable')
        gl = grp['gold_int'].tolist()
        if sum(gl) == 0:
            continue
        rr = next((1/(i+1) for i, x in enumerate(gl) if x), 0.0)
        per[cid] = rr
    return per

rand = {c: np.mean([rank_eval(None, s)[c] for s in (42, 7, 2025)])
        for c in rank_eval(None, 42)}
for name, col in [("TF-IDF", 'tf'), ("BGE-M3", 'bg'), ("RAG-lenient", 'rg')]:
    per = rank_eval(col)
    cits = sorted(set(per) & set(rand))
    xa = np.array([per[c] for c in cits])
    xb = np.array([rand[c] for c in cits])
    _, p = wilcoxon(xa, xb)
    print(f"  {name:<12} MRR={xa.mean():.4f}  vs random {xb.mean():.4f}  "
          f"Wilcoxon p={p:.4f}")

# ── [ERR] Error composition ─────────────────────────────────
print("\n[ERR] FN / FP on full set")
for name, p in [("TF-IDF", tf_pred), ("BGE-M3", bg_pred), ("RAG-lenient", ragb),
                ("DeBERTa consensus", db)]:
    fn = int(((p == 0) & (gold == 1)).sum())
    fp = int(((p == 1) & (gold == 0)).sum())
    print(f"  {name:<20} FN={fn:>3}  FP={fp:>3}")

# ── [SCL] Scale summary ─────────────────────────────────────
print("\n[SCL] Llama-3.1 scale summary (see results/llama70b_results.csv)")
print(pd.read_csv("results/llama70b_results.csv").to_string(index=False))
print("\nDone. Numbers correspond to the tables in the paper.")
