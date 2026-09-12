"""
DeBERTa-v3-base Supervised Baseline — CITIZEN-GROUPED CV
=========================================================
5-fold cross-validation GROUPED BY CITIZEN over all 521 gold pairs.
No citizen appears in both train and test of the same fold, so results
measure generalisation to entirely unseen citizens (stricter than the
pair-level CV run). Run alongside — not instead of — the pair-level
script; the paper reports both.

WHY K-FOLD: with only 521 pairs (115 positive) a fixed train/test split
would shrink the test set and break comparability with every existing
result. Under 5-fold CV every pair receives an out-of-fold (held-out)
prediction, so the final metrics are computed on the FULL 521-pair gold
set -- directly comparable with the Protocol A systems in the paper.

DESIGN
  - Model: microsoft/deberta-v3-base (~184M params, fits 16GB GPU easily)
  - Input: "[CITIZEN] <profile text> [SCHEME] <name + eligibility text>"
  - Loss:  class-weighted cross-entropy (imbalance 3.5:1)
  - Seeds: 3 full CV repetitions (42, 7, 2025) -> mean +/- SD
  - Output: out-of-fold predictions per pair (enables McNemar tests
    against the symbolic matcher and LLM baselines later)

SETUP (run once):
  pip install torch transformers sentencepiece protobuf scikit-learn pandas numpy

FILES NEEDED in the same folder:
  data/schemes.json
  data/citizen_profiles.json
  data/gold_eval_pairs.csv

RUN:
  python3 deberta_cv_baseline.py

EXPECTED RUNTIME on a 16GB GPU: ~1.5-2.5 hours total
(3 seeds x 5 folds = 15 short training runs of ~417 examples each).
CPU-only: not recommended (~12+ hours).

OUTPUTS:
  deberta_grouped_oof.csv   (per-pair held-out predictions, all seeds)
  deberta_grouped_results.csv        (per-seed metrics + mean/SD summary)
Upload BOTH to Claude Code for integration and significance testing.
"""

import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, f1_score
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_NAME  = "microsoft/deberta-v3-base"
SEEDS       = [42, 7, 2025]
N_FOLDS     = 5
EPOCHS      = 8
BATCH_SIZE  = 16
LR          = 2e-5
MAX_LEN     = 512
WARMUP_FRAC = 0.1

SCHEMES_FILE  = "data/schemes.json"
CITIZENS_FILE = "data/citizen_profiles.json"
EVAL_FILE     = "data/gold_eval_pairs.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cpu":
    print("WARNING: no GPU detected -- this will be very slow.")

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
with open(SCHEMES_FILE) as f:
    schemes = {s['scheme_id']: s for s in json.load(f)}
with open(CITIZENS_FILE) as f:
    citizens = {c['citizen_id']: c for c in json.load(f)}
eval_df = pd.read_csv(EVAL_FILE)
eval_df['gold_int'] = (eval_df['gold_label'] == 'eligible').astype(int)
print(f"Pairs: {len(eval_df)}  (positive: {eval_df['gold_int'].sum()})")

def citizen_to_text(c):
    parts = []
    for k, v in c.items():
        if k == 'citizen_id':
            continue
        if isinstance(v, bool):
            if v:
                parts.append(k.replace('_', ' '))
        elif v is not None:
            parts.append(f"{k.replace('_', ' ')}: {v}")
    return '; '.join(parts)

def scheme_to_text(s):
    name = s.get('scheme_name', '')
    elig = s.get('eligibility_text', '') or ''
    return f"{name}. Eligibility: {elig}"

texts, labels = [], []
for _, row in eval_df.iterrows():
    c = citizens.get(row['citizen_id'], {})
    s = schemes.get(row['scheme_id'], {})
    texts.append(f"[CITIZEN] {citizen_to_text(c)} [SCHEME] {scheme_to_text(s)}")
    labels.append(int(row['gold_int']))
labels = np.array(labels)

# Class weights for 3.5:1 imbalance
n_neg, n_pos = (labels == 0).sum(), (labels == 1).sum()
class_weights = torch.tensor(
    [len(labels) / (2 * n_neg), len(labels) / (2 * n_pos)],
    dtype=torch.float
).to(DEVICE)
print(f"Class weights: neg={class_weights[0]:.3f} pos={class_weights[1]:.3f}")

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PairDataset(Dataset):
    def __init__(self, idxs):
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        j = self.idxs[i]
        enc = tokenizer(
            texts[j], truncation=True, max_length=MAX_LEN,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(labels[j], dtype=torch.long),
            'row': j
        }

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────
# TRAIN / PREDICT ONE FOLD
# ─────────────────────────────────────────
def run_fold(train_idx, test_idx, seed):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).float().to(DEVICE)   # force fp32 master weights; autocast handles fp16 compute

    train_loader = DataLoader(PairDataset(train_idx), batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader  = DataLoader(PairDataset(test_idx), batch_size=BATCH_SIZE)

    steps = len(train_loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        opt, int(WARMUP_FRAC * steps), steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for batch in train_loader:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE)
                )
                loss = loss_fn(out.logits.float(), batch['label'].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
        # brief progress line per epoch
        print(f"      epoch {epoch+1}/{EPOCHS}  loss={total/len(train_loader):.4f}")

    # Out-of-fold predictions
    model.eval()
    rows, preds, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE)
            )
            p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            preds.extend((p >= 0.5).astype(int).tolist())
            probs.extend(p.tolist())
            rows.extend(batch['row'].tolist())

    del model
    torch.cuda.empty_cache()
    return rows, preds, probs

# ─────────────────────────────────────────
# FULL CV PER SEED
# ─────────────────────────────────────────
def compute_metrics(gold, preds):
    acc = accuracy_score(gold, preds)
    pr, re, f1b, _ = precision_recall_fscore_support(
        gold, preds, average='binary', zero_division=0)
    f1m = f1_score(gold, preds, average='macro', zero_division=0)
    kap = cohen_kappa_score(gold, preds)
    return dict(accuracy=round(acc, 4), precision=round(pr, 4),
                recall=round(re, 4), f1_binary=round(f1b, 4),
                f1_macro=round(f1m, 4), kappa=round(kap, 4))

oof = pd.DataFrame({
    'citizen_id': eval_df['citizen_id'],
    'scheme_id':  eval_df['scheme_id'],
    'gold':       labels
})
seed_results = []

for seed in SEEDS:
    print(f"\n=== SEED {seed} — {N_FOLDS}-fold CV ===")
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_pred = np.full(len(labels), -1)
    oof_prob = np.full(len(labels), -1.0)

    groups = eval_df['citizen_id'].values
    for k, (tr, te) in enumerate(skf.split(texts, labels, groups=groups), 1):
        print(f"   fold {k}/{N_FOLDS}  (train {len(tr)}, test {len(te)})")
        rows, preds, probs = run_fold(tr, te, seed + k)
        for r, p, q in zip(rows, preds, probs):
            oof_pred[r] = p
            oof_prob[r] = q

    assert (oof_pred >= 0).all(), "missing OOF predictions"
    m = compute_metrics(labels, oof_pred)
    m['seed'] = seed
    seed_results.append(m)
    oof[f'pred_seed{seed}'] = oof_pred
    oof[f'prob_seed{seed}'] = np.round(oof_prob, 4)
    print(f"   SEED {seed} OOF → Acc={m['accuracy']}  F1mac={m['f1_macro']}  "
          f"κ={m['kappa']}  Rec={m['recall']}")

# ─────────────────────────────────────────
# SUMMARY + SAVE
# ─────────────────────────────────────────
res = pd.DataFrame(seed_results)
print("\n" + "=" * 60)
print("DeBERTa-v3-base — CITIZEN-GROUPED 5-fold CV (Mean ± SD, 3 seeds)")
print("=" * 60)
summary_rows = []
for col in ['accuracy', 'precision', 'recall', 'f1_binary', 'f1_macro', 'kappa']:
    mu, sd = res[col].mean(), res[col].std()
    print(f"  {col:<12}: {mu:.4f} ± {sd:.4f}")
    summary_rows.append(dict(metric=col, mean=round(mu, 4), std=round(sd, 4)))

res.to_csv('deberta_grouped_results.csv', index=False)
pd.DataFrame(summary_rows).to_csv('deberta_grouped_summary.csv', index=False)
oof.to_csv('deberta_grouped_oof.csv', index=False)

print("\nSaved: deberta_grouped_results.csv, deberta_grouped_summary.csv,")
print("       deberta_grouped_oof.csv")
print("Upload all three to Claude Code for paper integration + McNemar tests.")
