"""
TF-IDF Rule-Overlap Baseline for IndiaWelfare-606
Outputs raw similarity scores (not binary predictions)
Threshold is tuned once on dev set, fixed for all seeds
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, f1_score
)
import random

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

with open('data/schemes.json') as f:
    schemes_list = json.load(f)

with open('data/citizen_profiles.json') as f:
    citizens_list = json.load(f)

eval_df = pd.read_csv('data/gold_eval_pairs.csv')

# Build lookup dicts
schemes  = {s['scheme_id']: s for s in schemes_list}
citizens = {c['citizen_id']: c for c in citizens_list}

print(f"Schemes loaded:  {len(schemes)}")
print(f"Citizens loaded: {len(citizens)}")
print(f"Eval pairs:      {len(eval_df)}")
print(f"Gold label dist:\n{eval_df['gold_label'].value_counts()}\n")

# ─────────────────────────────────────────
# 2. TEXT REPRESENTATION
# ─────────────────────────────────────────

def citizen_to_text(c):
    """Convert citizen profile dict to flat text string."""
    parts = []
    for k, v in c.items():
        if k == 'citizen_id':
            continue
        if isinstance(v, bool):
            if v:
                parts.append(k.replace('_', ' '))
        elif v is not None:
            parts.append(f"{k.replace('_',' ')} {v}")
    return ' '.join(parts)

def scheme_to_text(s):
    """Convert scheme to text using eligibility_text + rules."""
    parts = []
    if s.get('eligibility_text'):
        parts.append(s['eligibility_text'])
    rules = s.get('eligibility_rules_json') or []
    for r in rules:
        field = r.get('field', '')
        op    = r.get('op', '')
        value = r.get('value', '')
        parts.append(f"{field} {op} {value}")
    return ' '.join(parts)

# Build text representations for all schemes and citizens
scheme_texts  = {sid: scheme_to_text(s)  for sid, s in schemes.items()}
citizen_texts = {cid: citizen_to_text(c) for cid, c in citizens.items()}

# ─────────────────────────────────────────
# 3. COMPUTE RAW TF-IDF SIMILARITY SCORES
# ─────────────────────────────────────────

print("Computing TF-IDF similarity scores...")

# Collect all texts for fitting the vectorizer
all_texts = list(scheme_texts.values()) + list(citizen_texts.values())
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)
vectorizer.fit(all_texts)

scores_list = []
for _, row in eval_df.iterrows():
    cid = row['citizen_id']
    sid = row['scheme_id']
    gold = 1 if row['gold_label'] == 'eligible' else 0

    c_text = citizen_texts.get(cid, '')
    s_text = scheme_texts.get(sid, '')

    if not c_text or not s_text:
        score = 0.0
    else:
        c_vec = vectorizer.transform([c_text])
        s_vec = vectorizer.transform([s_text])
        score = float(cosine_similarity(c_vec, s_vec)[0][0])

    scores_list.append({
        'citizen_id': cid,
        'scheme_id':  sid,
        'gold_label': gold,
        'tfidf_score': score
    })

scores_df = pd.DataFrame(scores_list)
print(f"Score range: {scores_df['tfidf_score'].min():.4f} – {scores_df['tfidf_score'].max():.4f}")
print(f"Score mean:  {scores_df['tfidf_score'].mean():.4f}")

# ─────────────────────────────────────────
# 4. TUNE THRESHOLD ON DEV SET (20%)
# ─────────────────────────────────────────

np.random.seed(42)
idx = np.random.permutation(len(scores_df))
dev_idx  = idx[:int(0.2 * len(idx))]   # 20% dev
test_idx = idx[int(0.2 * len(idx)):]   # 80% test

dev_df  = scores_df.iloc[dev_idx]
test_df = scores_df.iloc[test_idx]

print(f"\nDev set:  {len(dev_df)} pairs ({dev_df['gold_label'].sum()} eligible)")
print(f"Test set: {len(test_df)} pairs ({test_df['gold_label'].sum()} eligible)")

# Sweep thresholds on dev set
best_threshold = 0.0
best_f1 = 0.0
threshold_results = []

for t in np.arange(0.01, 0.50, 0.01):
    preds = (dev_df['tfidf_score'] >= t).astype(int).tolist()
    gold  = dev_df['gold_label'].tolist()
    f1_mac = f1_score(gold, preds, average='macro', zero_division=0)
    threshold_results.append((round(t, 2), f1_mac))
    if f1_mac > best_f1:
        best_f1 = f1_mac
        best_threshold = round(t, 2)

print(f"\nOptimal threshold (dev Macro F1): {best_threshold} → F1={best_f1:.4f}")

# ─────────────────────────────────────────
# 5. EVALUATE ON TEST SET — 3 SEEDS
# ─────────────────────────────────────────

SEEDS = [42, 7, 2025]
seed_results = []

print(f"\n{'='*60}")
print(f"TFIDF BASELINE — Fixed Threshold={best_threshold} — Test Set")
print(f"{'='*60}")

for seed in SEEDS:
    # Seed controls any stochasticity (here: none, but kept for consistency)
    random.seed(seed)
    np.random.seed(seed)

    gold  = test_df['gold_label'].tolist()
    preds = (test_df['tfidf_score'] >= best_threshold).astype(int).tolist()

    acc = accuracy_score(gold, preds)
    p, r, f1_bin, _ = precision_recall_fscore_support(
        gold, preds, average='binary', zero_division=0
    )
    f1_mac = f1_score(gold, preds, average='macro', zero_division=0)
    kappa  = cohen_kappa_score(gold, preds)

    seed_results.append({
        'seed': seed,
        'threshold': best_threshold,
        'accuracy': round(acc, 4),
        'precision': round(p, 4),
        'recall': round(r, 4),
        'f1_binary': round(f1_bin, 4),
        'f1_macro': round(f1_mac, 4),
        'kappa': round(kappa, 4)
    })

    print(f"Seed {seed}: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  "
          f"F1_bin={f1_bin:.4f}  F1_mac={f1_mac:.4f}  κ={kappa:.4f}")

# ─────────────────────────────────────────
# 6. COMPUTE MEAN ± SD
# ─────────────────────────────────────────

results_df = pd.DataFrame(seed_results)
summary = {}
for col in ['accuracy', 'precision', 'recall', 'f1_binary', 'f1_macro', 'kappa']:
    summary[col] = {
        'mean': round(results_df[col].mean(), 4),
        'std':  round(results_df[col].std(), 4)
    }

print(f"\n{'─'*60}")
print("SUMMARY (Mean ± SD across 3 seeds):")
for metric, vals in summary.items():
    print(f"  {metric:<12}: {vals['mean']:.4f} ± {vals['std']:.4f}")

# ─────────────────────────────────────────
# 7. SAVE OUTPUTS
# ─────────────────────────────────────────

# Save raw scores
scores_df.to_csv('results/tfidf_raw_scores.csv', index=False)

# Save per-seed results
results_df.to_csv('results/tfidf_seed_results.csv', index=False)

# Save summary
summary_rows = []
for metric, vals in summary.items():
    summary_rows.append({
        'model': 'TF-IDF Rule-Overlap',
        'metric': metric,
        'mean': vals['mean'],
        'std': vals['std']
    })
pd.DataFrame(summary_rows).to_csv('results/tfidf_summary.csv', index=False)

print(f"\nFiles saved:")
print("  tfidf_raw_scores.csv")
print("  tfidf_seed_results.csv")
print("  tfidf_summary.csv")
print(f"\nOptimal threshold: {best_threshold}")
