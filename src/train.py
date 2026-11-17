"""Train the classifier and choose settings on the VALIDATION set only.

    python src/train.py

Model: character n-gram TF-IDF -> logistic regression (one trainable model).
The TF-IDF vocabulary and IDF weights are fitted on the training split only.
Selection: for every (analyzer, n-gram range, C) the decision threshold is tuned on validation;
the setting with the best validation F1 for class 1 is kept. Then the final model is refitted
on the training split with that setting and saved with its threshold to models/.
The test split is not read here.
"""
import ast, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from textnorm import normalize  # noqa: E402

CFG = json.loads((ROOT / "config.json").read_text())
SEED, M = CFG["seed"], CFG["model"]


def build(analyzer, ngram, C):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, analyzer=analyzer, ngram_range=tuple(ngram),
                                  min_df=M["min_df"], sublinear_tf=M["sublinear_tf"])),
        ("clf", LogisticRegression(C=C, class_weight=M["class_weight"], max_iter=M["max_iter"], random_state=SEED)),
    ])


def best_threshold(y, p):
    lo, hi, step = M["threshold_grid"]
    grid = np.round(np.arange(lo, hi + 1e-9, step), 2)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


def main():
    tr = pd.read_csv(ROOT / "data/processed/train.csv")
    va = pd.read_csv(ROOT / "data/processed/val.csv")
    print(f"train {len(tr)} (profanity {tr.label.sum()}), val {len(va)} (profanity {va.label.sum()})")

    rows = []
    for analyzer in M["analyzer"]:
        for ngram in M["ngram_range"]:
            for C in M["C"]:
                pipe = build(analyzer, ngram, C).fit(tr.text, tr.label)
                p = pipe.predict_proba(va.text)[:, 1]
                t, f1 = best_threshold(va.label, p)
                pred = (p >= t).astype(int)
                rows.append(dict(analyzer=analyzer, ngram=str(tuple(ngram)), C=C, threshold=t, val_f1=round(f1, 4),
                                 val_precision=round(precision_score(va.label, pred), 4),
                                 val_recall=round(recall_score(va.label, pred), 4),
                                 val_ap=round(average_precision_score(va.label, p), 4),
                                 vocab=len(pipe.named_steps["tfidf"].vocabulary_)))
                print(rows[-1])
    grid = pd.DataFrame(rows).sort_values(["val_f1", "val_ap"], ascending=False)
    (ROOT / "results").mkdir(exist_ok=True)
    grid.to_csv(ROOT / "results/validation_grid.csv", index=False)
    best = grid.iloc[0]
    print("\nselected on validation:\n", best.to_string())

    t0 = time.time()
    model = build(best.analyzer, ast.literal_eval(best.ngram), float(best.C)).fit(tr.text, tr.label)
    card = {"model": "TF-IDF character n-grams + logistic regression",
            "analyzer": best.analyzer, "ngram_range": ast.literal_eval(best.ngram), "C": float(best.C),
            "min_df": M["min_df"], "sublinear_tf": M["sublinear_tf"], "class_weight": M["class_weight"],
            "threshold": float(best.threshold), "seed": SEED,
            "selected_by": "highest validation F1 (class 1) after tuning the threshold on validation",
            "validation": {k: float(best[k]) for k in ("val_f1", "val_precision", "val_recall", "val_ap")},
            "train_size": int(len(tr)), "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
            "fit_seconds": round(time.time() - t0, 2)}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/model.joblib")
    (ROOT / "models/model_card.json").write_text(json.dumps(card, indent=2))
    print("saved models/model.joblib and models/model_card.json")


if __name__ == "__main__":
    main()
