"""
Claude Sonnet 3.5 Baseline for IndiaWelfare-606
Zero-shot and Few-shot evaluation via Anthropic API
Run this on your local machine with ANTHROPIC_API_KEY set

Setup:
    pip install anthropic pandas scikit-learn numpy
    export ANTHROPIC_API_KEY=your_key_here

Usage:
    python3 claude_sonnet_baseline.py
"""

import json
import time
import numpy as np
import pandas as pd
import anthropic
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, f1_score
)

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────

MODEL        = "claude-sonnet-4-5"   # Claude Sonnet 3.5
SEEDS        = [42, 7, 2025]
MAX_RETRIES  = 3
SLEEP_SEC    = 0.5                   # delay between API calls

# ── File paths — update to your local folder ──
SCHEMES_FILE  = "data/schemes.json"
CITIZENS_FILE = "data/citizen_profiles.json"
EVAL_FILE     = "data/gold_eval_pairs.csv"

# ─────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────

with open(SCHEMES_FILE) as f:
    schemes_list = json.load(f)

with open(CITIZENS_FILE) as f:
    citizens_list = json.load(f)

eval_df = pd.read_csv(EVAL_FILE)

schemes  = {s['scheme_id']: s for s in schemes_list}
citizens = {c['citizen_id']: c for c in citizens_list}

print(f"Schemes:  {len(schemes)}")
print(f"Citizens: {len(citizens)}")
print(f"Eval pairs: {len(eval_df)}")
print(f"Gold label dist:\n{eval_df['gold_label'].value_counts()}\n")

# ─────────────────────────────────────────
# 3. PROMPT BUILDERS
# ─────────────────────────────────────────

def build_zeroshot_prompt(citizen: dict, scheme: dict) -> str:
    citizen_text = "\n".join([
        f"  {k.replace('_',' ').title()}: {v}"
        for k, v in citizen.items()
        if k != 'citizen_id' and v is not None
    ])
    eligibility = scheme.get('eligibility_text', 'Not available')
    return f"""You are an expert welfare scheme eligibility assessor for Indian government schemes.

CITIZEN PROFILE:
{citizen_text}

SCHEME: {scheme.get('scheme_name', '')}
ELIGIBILITY CRITERIA:
{eligibility}

Task: Determine if this citizen is eligible for this scheme based strictly on the eligibility criteria.

Reply with ONLY one of these two words:
ELIGIBLE
NOT_ELIGIBLE"""


def build_fewshot_prompt(citizen: dict, scheme: dict, examples: list) -> str:
    few_shot_block = ""
    for ex in examples:
        c = ex['citizen']
        s = ex['scheme']
        label = "ELIGIBLE" if ex['gold_label'] == 'eligible' else "NOT_ELIGIBLE"
        c_text = "\n".join([
            f"  {k.replace('_',' ').title()}: {v}"
            for k, v in c.items()
            if k != 'citizen_id' and v is not None
        ])
        few_shot_block += f"""
--- EXAMPLE ---
CITIZEN:
{c_text}
SCHEME: {s.get('scheme_name','')}
ELIGIBILITY: {s.get('eligibility_text','')[:300]}
ANSWER: {label}
"""

    citizen_text = "\n".join([
        f"  {k.replace('_',' ').title()}: {v}"
        for k, v in citizen.items()
        if k != 'citizen_id' and v is not None
    ])
    eligibility = scheme.get('eligibility_text', 'Not available')

    return f"""You are an expert welfare scheme eligibility assessor for Indian government schemes.

Here are some examples:
{few_shot_block}

--- YOUR TASK ---
CITIZEN PROFILE:
{citizen_text}

SCHEME: {scheme.get('scheme_name', '')}
ELIGIBILITY CRITERIA:
{eligibility}

Reply with ONLY one of these two words:
ELIGIBLE
NOT_ELIGIBLE"""


# ─────────────────────────────────────────
# 4. API CALL WITH RETRY
# ─────────────────────────────────────────

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

def call_claude(prompt: str, seed: int) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=10,
                temperature=0.0,   # deterministic — no temperature randomness
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip().upper()
            # Parse response
            if "NOT_ELIGIBLE" in raw:
                return "not eligible"
            elif "ELIGIBLE" in raw:
                return "eligible"
            else:
                return "unknown"
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)  # exponential backoff
    return "unknown"


# ─────────────────────────────────────────
# 5. BUILD FEW-SHOT EXAMPLES
# ─────────────────────────────────────────

# Use 6 examples — 3 eligible, 3 not eligible — from first 50 pairs
# These must NOT be in your test set
np.random.seed(0)
few_shot_pool = eval_df.head(50)
eligible_ex   = few_shot_pool[few_shot_pool['gold_label'] == 'eligible'].head(3)
ineligible_ex = few_shot_pool[few_shot_pool['gold_label'] == 'not eligible'].head(3)
few_shot_rows = pd.concat([eligible_ex, ineligible_ex])

few_shot_examples = []
for _, row in few_shot_rows.iterrows():
    c = citizens.get(row['citizen_id'])
    s = schemes.get(row['scheme_id'])
    if c and s:
        few_shot_examples.append({
            'citizen': c,
            'scheme': s,
            'gold_label': row['gold_label']
        })

print(f"Few-shot examples prepared: {len(few_shot_examples)}")

# ─────────────────────────────────────────
# 6. EVALUATION FUNCTION
# ─────────────────────────────────────────

def compute_metrics(gold: list, preds: list) -> dict:
    # Exclude unknowns
    clean = [(g, p) for g, p in zip(gold, preds) if p != 'unknown']
    if not clean:
        return {}
    g_clean = [x[0] for x in clean]
    p_clean = [1 if x[1] == 'eligible' else 0 for x in clean]
    g_int   = [1 if g == 'eligible' else 0 for g in g_clean]

    acc    = accuracy_score(g_int, p_clean)
    pr, re, f1b, _ = precision_recall_fscore_support(
        g_int, p_clean, average='binary', zero_division=0
    )
    f1m  = f1_score(g_int, p_clean, average='macro', zero_division=0)
    kap  = cohen_kappa_score(g_int, p_clean)
    n_unk = len(gold) - len(clean)

    return {
        'accuracy':   round(acc, 4),
        'precision':  round(pr, 4),
        'recall':     round(re, 4),
        'f1_binary':  round(f1b, 4),
        'f1_macro':   round(f1m, 4),
        'kappa':      round(kap, 4),
        'n_unknown':  n_unk,
        'n_evaluated': len(clean)
    }


# ─────────────────────────────────────────
# 7. RUN ZERO-SHOT — 3 SEEDS
# ─────────────────────────────────────────

print("\n" + "="*60)
print("ZERO-SHOT EVALUATION")
print("="*60)

zeroshot_results = []

for seed in SEEDS:
    np.random.seed(seed)
    print(f"\nSeed {seed}:")
    preds = []

    for i, (_, row) in enumerate(eval_df.iterrows()):
        cid = row['citizen_id']
        sid = row['scheme_id']
        c   = citizens.get(cid)
        s   = schemes.get(sid)

        if not c or not s:
            preds.append('unknown')
            continue

        prompt = build_zeroshot_prompt(c, s)
        pred   = call_claude(prompt, seed)
        preds.append(pred)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(eval_df)}")

        time.sleep(SLEEP_SEC)

    gold = eval_df['gold_label'].tolist()
    m    = compute_metrics(gold, preds)
    m['seed'] = seed
    zeroshot_results.append(m)
    print(f"  Seed {seed} → F1_mac={m.get('f1_macro','?')}  "
          f"Kappa={m.get('kappa','?')}  Unknown={m.get('n_unknown',0)}")

# Save zero-shot predictions from last seed
eval_df['claude_zeroshot'] = preds
eval_df.to_csv('claude_zeroshot_predictions.csv', index=False)

# Summary
zs_df = pd.DataFrame(zeroshot_results)
print("\nZero-Shot Summary (Mean ± SD):")
for col in ['accuracy','precision','recall','f1_binary','f1_macro','kappa']:
    print(f"  {col:<12}: {zs_df[col].mean():.4f} ± {zs_df[col].std():.4f}")


# ─────────────────────────────────────────
# 8. RUN FEW-SHOT — 3 SEEDS
# ─────────────────────────────────────────

print("\n" + "="*60)
print("FEW-SHOT EVALUATION")
print("="*60)

fewshot_results = []

for seed in SEEDS:
    np.random.seed(seed)
    print(f"\nSeed {seed}:")
    preds = []

    for i, (_, row) in enumerate(eval_df.iterrows()):
        cid = row['citizen_id']
        sid = row['scheme_id']
        c   = citizens.get(cid)
        s   = schemes.get(sid)

        if not c or not s:
            preds.append('unknown')
            continue

        prompt = build_fewshot_prompt(c, s, few_shot_examples)
        pred   = call_claude(prompt, seed)
        preds.append(pred)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(eval_df)}")

        time.sleep(SLEEP_SEC)

    gold = eval_df['gold_label'].tolist()
    m    = compute_metrics(gold, preds)
    m['seed'] = seed
    fewshot_results.append(m)
    print(f"  Seed {seed} → F1_mac={m.get('f1_macro','?')}  "
          f"Kappa={m.get('kappa','?')}  Unknown={m.get('n_unknown',0)}")

# Save few-shot predictions from last seed
eval_df['claude_fewshot'] = preds
eval_df.to_csv('claude_fewshot_predictions.csv', index=False)

# Summary
fs_df = pd.DataFrame(fewshot_results)
print("\nFew-Shot Summary (Mean ± SD):")
for col in ['accuracy','precision','recall','f1_binary','f1_macro','kappa']:
    print(f"  {col:<12}: {fs_df[col].mean():.4f} ± {fs_df[col].std():.4f}")


# ─────────────────────────────────────────
# 9. SAVE ALL RESULTS
# ─────────────────────────────────────────

zs_df['mode'] = 'zero-shot'
fs_df['mode'] = 'few-shot'
all_results = pd.concat([zs_df, fs_df], ignore_index=True)
all_results.to_csv('claude_sonnet_results.csv', index=False)

print("\n" + "="*60)
print("FILES SAVED:")
print("  claude_zeroshot_predictions.csv")
print("  claude_fewshot_predictions.csv")
print("  claude_sonnet_results.csv")
print()
print("Upload claude_sonnet_results.csv and both prediction")
print("CSVs to Claude Code for final table compilation.")
