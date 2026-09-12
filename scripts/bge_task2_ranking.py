"""
BGE-M3 Task 2 Ranking Baseline for IndiaWelfare-606
====================================================
Purpose: add a BGE-M3 row to the Task 2 retrieval table so the same
dense retriever is evaluated on both tasks (reviewer symmetry fix).

PREFERRED PATH (Option 1): if you still have the ORIGINAL Task 2 script
that produced the paper's Table 8 (Random / TF-IDF / MiniLM / Rule-overlap),
change ONE line in it:

    model = SentenceTransformer('all-MiniLM-L6-v2')
        -->
    model = SentenceTransformer('BAAI/bge-m3')

and rerun. That guarantees identical pools, identical random baseline,
identical Wilcoxon code. Use this script only if the original is lost.

FALLBACK PATH (Option 2): this standalone script ranks each citizen's
judged candidate set from the gold eval CSV. After running, compare its
Random and TF-IDF rows against the paper's Table 8:
  - If they match (MRR ~0.656 / ~0.674), insert the BGE-M3 row directly.
  - If they differ, the original pools included extra candidates; report
    this script's results as a self-contained comparison instead (all
    four methods from the same protocol -- still fully valid).

Setup:
    pip install sentence-transformers scipy pandas numpy scikit-learn
Files needed in the same folder:
    data/schemes.json
    data/citizen_profiles.json
    data/gold_eval_pairs.csv
Run:
    python3 bge_task2_ranking.py
Output:
    task2_bge_results.csv        (aggregate table, all methods)
    task2_per_citizen_mrr.csv    (per-citizen MRR for Wilcoxon reuse)
Runtime: ~10-15 min (BGE-M3 already cached from the Task 1 run).
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
SCHEMES_FILE  = "data/schemes.json"
CITIZENS_FILE = "data/citizen_profiles.json"
EVAL_FILE     = "data/gold_eval_pairs.csv"

with open(SCHEMES_FILE) as f:
    schemes_list = json.load(f)
with open(CITIZENS_FILE) as f:
    citizens_list = json.load(f)
eval_df = pd.read_csv(EVAL_FILE)

schemes  = {s['scheme_id']: s for s in schemes_list}
citizens = {c['citizen_id']: c for c in citizens_list}
eval_df['gold_int'] = (eval_df['gold_label'] == 'eligible').astype(int)

print(f"Schemes {len(schemes)} | Citizens {len(citizens)} | Pairs {len(eval_df)}")

# ─────────────────────────────────────────
# 2. TEXT REPRESENTATIONS (same as Task 1 scripts)
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
            parts.append(f"{k.replace('_', ' ')} {v}")
    return ' '.join(parts)

def scheme_to_text(s):
    parts = []
    if s.get('eligibility_text'):
        parts.append(s['eligibility_text'])
    for r in (s.get('eligibility_rules_json') or []):
        parts.append(f"{r.get('field','')} {r.get('op','')} {r.get('value','')}")
    return ' '.join(parts)

scheme_texts  = {sid: scheme_to_text(s)  for sid, s in schemes.items()}
citizen_texts = {cid: citizen_to_text(c) for cid, c in citizens.items()}

# ─────────────────────────────────────────
# 3. SCORES
# ─────────────────────────────────────────
print("Encoding with BGE-M3 (cached from Task 1 run)...")
model = SentenceTransformer('BAAI/bge-m3')

uniq_c = eval_df['citizen_id'].unique().tolist()
uniq_s = eval_df['scheme_id'].unique().tolist()
c_emb = {c: model.encode(citizen_texts.get(c, ''), normalize_embeddings=True) for c in uniq_c}
s_emb = {s: model.encode(scheme_texts.get(s, ''),  normalize_embeddings=True) for s in uniq_s}

eval_df['bge_score'] = [
    float(np.dot(c_emb[r.citizen_id], s_emb[r.scheme_id]))
    for r in eval_df.itertuples()
]

print("Computing TF-IDF scores (protocol sanity check)...")
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
vec.fit(list(scheme_texts.values()) + list(citizen_texts.values()))
eval_df['tfidf_score'] = [
    float(cosine_similarity(
        vec.transform([citizen_texts.get(r.citizen_id, '')]),
        vec.transform([scheme_texts.get(r.scheme_id, '')]))[0][0])
    for r in eval_df.itertuples()
]

# ─────────────────────────────────────────
# 4. RANKING METRICS
# ─────────────────────────────────────────
def rr(g):
    for i, x in enumerate(g):
        if x == 1:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_k(g, k):
    dcg  = sum(g[i] / np.log2(i + 2) for i in range(min(k, len(g))))
    idl  = sorted(g, reverse=True)
    idcg = sum(idl[i] / np.log2(i + 2) for i in range(min(k, len(g))))
    return dcg / idcg if idcg > 0 else 0.0

def p_at(g, k):
    return sum(g[:k]) / k

def r_at(g, k):
    tot = sum(g)
    return sum(g[:k]) / tot if tot > 0 else 0.0

def eval_rank(df, col, random_seed=None):
    """Rank each citizen's judged candidate set; return per-citizen metrics."""
    per_cit = {}
    for cid, grp in df.groupby('citizen_id'):
        if len(grp) < 2:
            continue
        if random_seed is not None:
            rng = np.random.RandomState(random_seed + hash(cid) % 10000)
            grp = grp.assign(_s=rng.rand(len(grp))).sort_values('_s', ascending=False)
        else:
            grp = grp.sort_values(col, ascending=False)
        g = grp['gold_int'].tolist()
        if sum(g) == 0:
            continue
        per_cit[cid] = dict(P3=p_at(g, 3), P5=p_at(g, 5), R5=r_at(g, 5),
                            MRR=rr(g), NDCG5=ndcg_k(g, 5))
    return per_cit

# Random averaged over 3 seeds; deterministic methods once
rand_runs = [eval_rank(eval_df, None, random_seed=s) for s in (42, 7, 2025)]
cits = sorted(rand_runs[0].keys())
rand = {c: {m: np.mean([r[c][m] for r in rand_runs])
            for m in ('P3', 'P5', 'R5', 'MRR', 'NDCG5')} for c in cits}
tfid = eval_rank(eval_df, 'tfidf_score')
bge  = eval_rank(eval_df, 'bge_score')

def agg(d):
    return {m: round(float(np.mean([v[m] for v in d.values()])), 3)
            for m in ('P3', 'P5', 'R5', 'MRR', 'NDCG5')}

# ─────────────────────────────────────────
# 5. RESULTS + SANITY CHECK
# ─────────────────────────────────────────
rows = [('Random (3-seed mean)', agg(rand)),
        ('TF-IDF',               agg(tfid)),
        ('BGE-M3',               agg(bge))]

print("\n" + "=" * 70)
print(f"TASK 2 RANKING — {len(cits)} citizens with >=1 eligible scheme")
print("=" * 70)
print(f"{'Method':<24} {'P@3':>6} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'N@5':>6}")
for name, a in rows:
    print(f"{name:<24} {a['P3']:>6} {a['P5']:>6} {a['R5']:>6} "
          f"{a['MRR']:>6} {a['NDCG5']:>6}")

print("\nSANITY CHECK vs paper Table 8:")
print("  Paper Random MRR = 0.656 | Paper TF-IDF MRR = 0.674")
print("  If your Random/TF-IDF above are close -> insert BGE-M3 row directly.")
print("  If not -> original pools differed; report this table as a")
print("  self-contained four-method comparison under one protocol.")

# ─────────────────────────────────────────
# 6. WILCOXON SIGNED-RANK TESTS (per-citizen MRR)
# ─────────────────────────────────────────
print("\nWilcoxon signed-rank tests (per-citizen MRR):")

def wtest(a, b, label):
    xa = np.array([a[c]['MRR'] for c in cits])
    xb = np.array([b[c]['MRR'] for c in cits])
    if np.all(xa - xb == 0):
        print(f"  {label:<24} identical rankings")
        return None
    stat, p = wilcoxon(xa, xb)
    sig = 'significant' if p < 0.05 else 'ns'
    print(f"  {label:<24} dMRR={np.mean(xa-xb):+.3f}  p={p:.4f}  {sig}")
    return p

wtest(bge,  rand, 'BGE-M3 vs Random')
wtest(bge,  tfid, 'BGE-M3 vs TF-IDF')
wtest(tfid, rand, 'TF-IDF vs Random')

# ─────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────
out = pd.DataFrame(
    [dict(method=n, **a) for n, a in rows]
)
out.to_csv('task2_bge_results.csv', index=False)

per_cit_rows = []
for c in cits:
    per_cit_rows.append(dict(citizen_id=c,
                             random_mrr=rand[c]['MRR'],
                             tfidf_mrr=tfid[c]['MRR'],
                             bge_mrr=bge[c]['MRR']))
pd.DataFrame(per_cit_rows).to_csv('task2_per_citizen_mrr.csv', index=False)

print("\nSaved: task2_bge_results.csv, task2_per_citizen_mrr.csv")
print("Upload both to Claude Code to fold into the paper.")
