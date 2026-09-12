import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("results/ensemble_dataset.csv")

# Convert labels
df["gold_label"] = df["gold_label"].map({
    "eligible": 1,
    "not eligible": 0
})

df["auto_pred"] = df["auto_pred"].map({
    "eligible": 1,
    "not eligible": 0
})

X = df[[
    "auto_pred",
    "prob_seed42",
    "tfidf_score",
    "bge_score"
]]

y = df["gold_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

print("\nFeature Importance:")
for f, imp in zip(X.columns, model.feature_importances_):
    print(f"{f:15s} {imp:.4f}")