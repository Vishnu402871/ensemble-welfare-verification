# IndiaWelfare-606 — Baselines and Analysis

Code and result files for the paper:

> **IndiaWelfare-606: A Benchmark for Structured Eligibility Reasoning and
> Scheme Retrieval in Public Welfare Systems.**
> Sudha Subramanian, Kavya Sri K, Jaslyn Fathima M, Architha R.
> *Under review, Machine Learning (ACML 2026 journal track).*

**Dataset:** https://huggingface.co/datasets/sudhaksmanian/IndiaWelfare-606

## Quick start — reproduce the paper's tables in one command

```bash
pip install -r requirements.txt
# download the three data files from the HuggingFace dataset into data/
python3 analysis.py
```

`analysis.py` reproduces, from the released prediction/score files in
`results/` (no GPU, ~30 seconds): the Protocol-B main results with 95%
bootstrap CIs, ROC/PR-AUC, all McNemar tests, the per-category macro-F1
table, the supervised DeBERTa results (pair-level and citizen-grouped
consensus) with significance tests, the Task-2 judged-subset re-ranking
with Wilcoxon tests, the error composition table, and the 8B-vs-70B
scale summary.

## Repository layout

```
analysis.py                       one-command reproduction of paper tables
requirements.txt
data/                             download from the HuggingFace dataset (see data/README.md)
scripts/
  auto_baseline_v7.py             rule-based symbolic matcher (paper's headline system)
  tfidf_baseline.py               Protocol B lexical baseline (dev-tuned threshold)
  bge_baseline.py                 Protocol B dense baseline (BGE-M3)
  bge_task2_ranking.py            Task 2 judged-subset ranking (all methods)
  llama70b_baseline.py            Llama-3.1-70B zero/few-shot via Ollama
  deberta_cv_baseline.py          Supervised DeBERTa-v3, pair-level 5-fold CV
  deberta_grouped_cv.py           Citizen-grouped 5-fold CV (robustness check)
  claude_sonnet_baseline.py       Optional API-based LLM baseline (not used in paper)
results/                          released scores & predictions backing every table
  auto_v7_predictions.csv         symbolic matcher per-pair predictions
  tfidf_raw_scores.csv            raw cosine similarities (521 pairs)
  bge_raw_scores.csv
  rag_lenient_predictions.csv     lenient-policy RAG predictions (Llama-3.1-8B)
  llama70b_predictions.csv / llama70b_results.csv
  deberta_oof_predictions.csv     out-of-fold predictions, 3 seeds (pair-level CV)
  deberta_cv_results.csv / deberta_cv_summary.csv
  deberta_grouped_oof.csv / *_results.csv / *_summary.csv
  task2_bge_results.csv / task2_per_citizen_mrr.csv
```

## Re-running baselines from scratch

Each script in `scripts/` is standalone; header docstrings give setup,
expected runtime, and outputs. GPU notes: DeBERTa CV needs ~6GB VRAM
(~2h for 15 runs); Llama-70B needs ~40GB (or the q4 quantisation under
24GB) via [Ollama](https://ollama.com). The rule-based symbolic matcher
operates directly on `eligibility_rules_json` — `scripts/auto_baseline_v7.py`
(no GPU; runs in seconds and reproduces the paper's headline row exactly).

## Protocols (summary)

* **Protocol A** — full 521-pair gold set; threshold-free systems
  (symbolic matcher, prompted LLMs). Parse failures map to *ineligible*.
* **Protocol B** — thresholds for similarity scorers tuned once on a
  104-pair dev split (seed 42), frozen, evaluated on the remaining 417
  pairs (TF-IDF 0.03; BGE-M3 0.62).
* **Supervised** — stratified (and citizen-grouped) 5-fold CV over all
  521 pairs; out-of-fold predictions; 3 CV repetitions.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0 with government-source
attribution — see the licensing section of the dataset card.

## Citation

```bibtex
@article{indiawelfare606,
  title   = {IndiaWelfare-606: A Benchmark for Structured Eligibility Reasoning
             and Scheme Retrieval in Public Welfare Systems},
  author  = {Subramanian, Sudha and K, Kavya Sri and M, Jaslyn Fathima and R, Architha},
  journal = {Machine Learning (under review, ACML 2026 journal track)},
  year    = {2026}
}
```
