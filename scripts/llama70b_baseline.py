"""
Llama-3.1-70B Baseline for IndiaWelfare-606 (zero-shot + few-shot)
Runs locally via Ollama; greedy decoding (temperature 0) -> deterministic.

Setup:
    ollama pull llama3.1:70b        (~40GB; use llama3.1:70b-instruct-q4_K_M under 24GB VRAM)
    pip install ollama pandas scikit-learn numpy
Data:
    Download schemes.json, citizen_profiles.json, gold_eval_pairs.csv from
    the HuggingFace dataset into data/ (see data/README.md).
Run:
    python3 scripts/llama70b_baseline.py
Outputs:
    results/llama70b_results.csv, results/llama70b_predictions.csv
"""

import json
import time
import numpy as np
import pandas as pd
import ollama
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, f1_score
)

MODEL = "llama3.1:70b"
SEEDS = [42]          # temperature 0 -> deterministic; one seed suffices
SLEEP = 0.2

with open("data/schemes.json") as f:
    schemes = {s['scheme_id']: s for s in json.load(f)}
with open("data/citizen_profiles.json") as f:
    citizens = {c['citizen_id']: c for c in json.load(f)}
eval_df = pd.read_csv("data/gold_eval_pairs.csv")
print(f"Schemes {len(schemes)} | Citizens {len(citizens)} | Pairs {len(eval_df)}")

def citizen_text(c):
    return "\n".join(f"  {k.replace('_',' ').title()}: {v}"
                     for k, v in c.items()
                     if k != 'citizen_id' and v is not None)

def zeroshot_prompt(c, s):
    return f"""You are a welfare scheme eligibility assessor.

CITIZEN PROFILE:
{citizen_text(c)}

SCHEME: {s.get('scheme_name','')}
ELIGIBILITY CRITERIA:
{s.get('eligibility_text','Not available')}

Is this citizen eligible? Reply with ONLY one word:
ELIGIBLE or NOT_ELIGIBLE"""

def fewshot_prompt(c, s, examples):
    block = ""
    for ex in examples:
        label = "ELIGIBLE" if ex['gold_label'] == 'eligible' else "NOT_ELIGIBLE"
        block += f"""
--- EXAMPLE ---
CITIZEN:
{citizen_text(ex['citizen'])}
SCHEME: {ex['scheme'].get('scheme_name','')}
ELIGIBILITY: {ex['scheme'].get('eligibility_text','')[:300]}
ANSWER: {label}
"""
    return f"""You are a welfare scheme eligibility assessor.

Examples:
{block}

--- YOUR TASK ---
CITIZEN PROFILE:
{citizen_text(c)}

SCHEME: {s.get('scheme_name','')}
ELIGIBILITY CRITERIA:
{s.get('eligibility_text','Not available')}

Reply with ONLY one word: ELIGIBLE or NOT_ELIGIBLE"""

def call_ollama(prompt, seed):
    for attempt in range(3):
        try:
            r = ollama.chat(model=MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            options={"temperature": 0.0, "seed": seed})
            raw = r['message']['content'].strip().upper()
            if "NOT_ELIGIBLE" in raw:
                return "not eligible"
            if "ELIGIBLE" in raw:
                return "eligible"
            return "unknown"
        except Exception as e:
            print(f"  error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return "unknown"

def compute_metrics(gold, preds):
    clean = [(g, p) for g, p in zip(gold, preds) if p != 'unknown']
    if not clean:
        return {}
    g = [1 if x[0] == 'eligible' else 0 for x in clean]
    p = [1 if x[1] == 'eligible' else 0 for x in clean]
    pr, re_, f1b, _ = precision_recall_fscore_support(
        g, p, average='binary', zero_division=0)
    return dict(accuracy=round(accuracy_score(g, p), 4),
                precision=round(pr, 4), recall=round(re_, 4),
                f1_binary=round(f1b, 4),
                f1_macro=round(f1_score(g, p, average='macro',
                                        zero_division=0), 4),
                kappa=round(cohen_kappa_score(g, p), 4),
                n_unknown=len(gold) - len(clean))

# Few-shot examples: 3 eligible + 3 not-eligible from the first 50 pairs
np.random.seed(0)
pool = eval_df.head(50)
ex_rows = pd.concat([pool[pool['gold_label'] == 'eligible'].head(3),
                     pool[pool['gold_label'] == 'not eligible'].head(3)])
few_shot = [{'citizen': citizens[r['citizen_id']],
             'scheme': schemes[r['scheme_id']],
             'gold_label': r['gold_label']}
            for _, r in ex_rows.iterrows()
            if r['citizen_id'] in citizens and r['scheme_id'] in schemes]

all_results = []
for mode, prompt_fn in [("zero-shot", lambda c, s: zeroshot_prompt(c, s)),
                        ("few-shot", lambda c, s: fewshot_prompt(c, s, few_shot))]:
    for seed in SEEDS:
        preds = []
        for i, (_, row) in enumerate(eval_df.iterrows()):
            c = citizens.get(row['citizen_id'])
            s = schemes.get(row['scheme_id'])
            preds.append(call_ollama(prompt_fn(c, s), seed)
                         if c and s else 'unknown')
            if (i + 1) % 50 == 0:
                print(f"  {mode} seed {seed}: {i+1}/{len(eval_df)}")
            time.sleep(SLEEP)
        m = compute_metrics(eval_df['gold_label'].tolist(), preds)
        m.update(mode=mode, seed=seed)
        all_results.append(m)
        col = 'llama70b_zeroshot' if mode == 'zero-shot' else 'llama70b_fewshot'
        eval_df[col] = preds
        print(f"{mode} seed {seed} -> F1mac={m.get('f1_macro')} "
              f"kappa={m.get('kappa')}")

pd.DataFrame(all_results).to_csv('results/llama70b_results.csv', index=False)
eval_df.to_csv('results/llama70b_predictions.csv', index=False)
print("Saved: results/llama70b_results.csv, results/llama70b_predictions.csv")
