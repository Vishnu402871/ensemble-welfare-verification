"""
BGE-M3 Dense Retrieval Baseline for IndiaWelfare-606
Outputs raw similarity scores (not binary predictions)
Threshold is tuned once on dev set, fixed for all seeds
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, f1_score
)
from sentence_transformers import SentenceTransformer
import random

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

with open('data/schemes.json') as f:
    schemes_list = json.load(f)

with open('data/citizen_profiles.json') as f:
    citizens_list = json.load(f)

eval_df = pd.read_csv('data/gold_eval_pairs.csv')

schemes  = {s['scheme_id']: s for s in schemes_list}
citizens = {c['citizen_id']: c for c in citizens_list}

print(f"Schemes loaded:  {len(schemes)}")
print(f"Citizens loaded: {len(citizens)}")
print(f"Eval pairs:      {len(eval_df)}")

# ─────────────────────────────────────────
# 2. TEXT REPRESENTATION
# ─────────────────────────────────────────

def citizen_to_text(c):
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

scheme_texts  = {sid: scheme_to_text(s)  for sid, s in schemes.items()}
citizen_texts = {cid: citizen_to_text(c) for cid, c in citizens.items()}

# ─────────────────────────────────────────
# 3. LOAD BGE-M3 MODEL AND ENCODE
# ─────────────────────────────────────────

print("\nLoading BGE-M3 model (this may take a minute on first run)...")
model = SentenceTransformer('BAAI/bge-m3')

# Encode all unique citizens and schemes in eval set
unique_cids = eval_df['citizen_id'].unique().tolist()
unique_sids = eval_df['scheme_id'].unique().tolist()

print(f"Encoding {len(unique_cids)} unique citizens...")
citizen_embs = {}
for cid in unique_cids:
    text = citizen_texts.get(cid, '')
    if text:
        citizen_embs[cid] = model.encode(text, normalize_embeddings=True)

print(f"Encoding {len(unique_sids)} unique schemes...")
scheme_embs = {}
for sid in unique_sids:
    text = scheme_texts.get(sid, '')
    if text:
        scheme_embs[sid] = model.encode(text, normalize_embeddings=True)

# ─────────────────────────────────────────
# 4. COMPUTE RAW COSINE SIMILARITY SCORES
# ─────────────────────────────────────────

print("\nComputing cosine similarity scores...")
scores_list = []
for _, row in eval_df.iterrows():
    cid  = row['citizen_id']
    sid  = row['scheme_id']
    gold = 1 if row['gold_label'] == 'eligible' else 0

    c_emb = citizen_embs.get(cid)
    s_emb = scheme_embs.get(sid)

    if c_emb is None or s_emb is None:
        score = 0.0
    else:
        score = float(np.dot(c_emb, s_emb))  # cosine sim (normalized)

    scores_list.append({
        'citizen_id': cid,
        'scheme_id':  sid,
        'gold_label': gold,
        'bge_score':  score
    })

scores_df = pd.DataFrame(scores_list)
print(f"Score range: {scores_df['bge_score'].min():.4f} – {scores_df['bge_score'].max():.4f}")
print(f"Score mean:  {scores_df['bge_score'].mean():.4f}")

# ─────────────────────────────────────────
# 5. TUNE THRESHOLD ON DEV SET (20%)
# ─────────────────────────────────────────

np.random.seed(42)
idx = np.random.permutation(len(scores_df))
dev_idx  = idx[:int(0.2 * len(idx))]
test_idx = idx[int(0.2 * len(idx)):]

dev_df  = scores_df.iloc[dev_idx]
test_df = scores_df.iloc[test_idx]

print(f"\nDev set:  {len(dev_df)} pairs ({dev_df['gold_label'].sum()} eligible)")
print(f"Test set: {len(test_df)} pairs ({test_df['gold_label'].sum()} eligible)")

best_threshold = 0.0
best_f1 = 0.0

for t in np.arange(0.50, 0.95, 0.01):
    preds  = (dev_df['bge_score'] >= t).astype(int).tolist()
    gold   = dev_df['gold_label'].tolist()
    f1_mac = f1_score(gold, preds, average='macro', zero_division=0)
    if f1_mac > best_f1:
        best_f1 = f1_mac
        best_threshold = round(t, 2)

print(f"\nOptimal threshold (dev Macro F1): {best_threshold} → F1={best_f1:.4f}")

# ─────────────────────────────────────────
# 6. EVALUATE ON TEST SET — 3 SEEDS
# ─────────────────────────────────────────

SEEDS = [42, 7, 2025]
seed_results = []

print(f"\n{'='*60}")
print(f"BGE-M3 BASELINE — Fixed Threshold={best_threshold} — Test Set")
print(f"{'='*60}")

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)

    gold  = test_df['gold_label'].tolist()
    preds = (test_df['bge_score'] >= best_threshold).astype(int).tolist()

    acc    = accuracy_score(gold, preds)
    p, r, f1_bin, _ = precision_recall_fscore_support(
        gold, preds, average='binary', zero_division=0
    )
    f1_mac = f1_score(gold, preds, average='macro', zero_division=0)
    kappa  = cohen_kappa_score(gold, preds)

    seed_results.append({
        'seed': seed,
        'threshold': best_threshold,
        'accuracy':   round(acc, 4),
        'precision':  round(p, 4),
        'recall':     round(r, 4),
        'f1_binary':  round(f1_bin, 4),
        'f1_macro':   round(f1_mac, 4),
        'kappa':      round(kappa, 4)
    })

    print(f"Seed {seed}: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  "
          f"F1_bin={f1_bin:.4f}  F1_mac={f1_mac:.4f}  κ={kappa:.4f}")

# ─────────────────────────────────────────
# 7. COMPUTE MEAN ± SD
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
# 8. SAVE OUTPUTS
# ─────────────────────────────────────────

scores_df.to_csv('results/bge_raw_scores.csv', index=False)
results_df.to_csv('results/bge_seed_results.csv', index=False)

summary_rows = []
for metric, vals in summary.items():
    summary_rows.append({
        'model': 'BGE-M3 Dense Retrieval',
        'metric': metric,
        'mean': vals['mean'],
        'std':  vals['std']
    })
pd.DataFrame(summary_rows).to_csv('results/bge_summary.csv', index=False)

print(f"\nFiles saved:")
print("  bge_raw_scores.csv")
print("  bge_seed_results.csv")
print("  bge_summary.csv")
print(f"\nOptimal threshold: {best_threshold}")
