import pandas as pd

auto = pd.read_csv("results/auto_v7_predictions.csv")
deb = pd.read_csv("results/deberta_oof_predictions.csv")
tfidf = pd.read_csv("results/tfidf_raw_scores.csv")
bge = pd.read_csv("results/bge_raw_scores.csv")

df = auto[["citizen_id","scheme_id","gold_label","auto_pred"]].copy()

df = df.merge(
    deb[["citizen_id","scheme_id","prob_seed42"]],
    on=["citizen_id","scheme_id"]
)

df = df.merge(
    tfidf[["citizen_id","scheme_id","tfidf_score"]],
    on=["citizen_id","scheme_id"]
)

df = df.merge(
    bge[["citizen_id","scheme_id","bge_score"]],
    on=["citizen_id","scheme_id"]
)

print(df.head())
print(df.shape)

df.to_csv("results/ensemble_dataset.csv", index=False)