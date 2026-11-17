"""Evaluate the saved model ONCE on the held-out test split.

    python src/evaluate.py

Writes results/test_metrics.json, results/confusion_matrix.png,
results/test_predictions.csv and results/test_errors.csv.
"""
import json, sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    model = joblib.load(ROOT / "models/model.joblib")
    card = json.loads((ROOT / "models/model_card.json").read_text())
    thr = card["threshold"]
    te = pd.read_csv(ROOT / "data/processed/test.csv")

    p = model.predict_proba(te.text)[:, 1]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(te.label, pred, labels=[0, 1]).ravel()
    metrics = {"n_test": int(len(te)), "n_profanity": int(te.label.sum()), "n_clean": int((te.label == 0).sum()),
               "threshold": thr,
               "precision_profanity": round(precision_score(te.label, pred), 4),
               "recall_profanity": round(recall_score(te.label, pred), 4),
               "f1_profanity": round(f1_score(te.label, pred), 4),
               "average_precision": round(average_precision_score(te.label, p), 4),
               "accuracy": round(float((pred == te.label).mean()), 4),
               "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
    out = te.assign(prob_profanity=p.round(4), pred=pred)
    # false-positive rate on the added product reviews (all clean by construction)
    mk = out[out.source.str.startswith("kazsandra_market")]
    metrics["market_reviews"] = {"n": int(len(mk)), "false_positives": int(mk.pred.sum())}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/test_metrics.json").write_text(json.dumps(metrics, indent=2))
    out.to_csv(ROOT / "results/test_predictions.csv", index=False)
    err = out[out.pred != out.label].copy()
    err["error"] = err.label.map({0: "false positive", 1: "false negative"})
    err.sort_values(["error", "prob_profanity"]).to_csv(ROOT / "results/test_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    cm = [[tn, fp], [fn, tp]]
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i][j] > max(tn, tp) / 2 else "black")
    ax.set_xticks([0, 1], ["0 clean", "1 profanity"]); ax.set_yticks([0, 1], ["0 clean", "1 profanity"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Test set (n = {len(te)}), threshold {thr}")
    fig.tight_layout(); fig.savefig(ROOT / "results/confusion_matrix.png", dpi=150)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
