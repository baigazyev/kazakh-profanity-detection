"""Predict on new comments with the saved model (no retraining).

    python src/predict.py "Мына ойын керемет екен"
    python src/predict.py            # interactive: type comments, empty line to quit

From Python / a notebook:
    from src.predict import load, predict
    model, thr = load()
    predict(["текст 1", "текст 2"], model, thr)
"""
import json, sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # the saved pipeline references textnorm.normalize


def load(model_dir=ROOT / "models"):
    model = joblib.load(Path(model_dir) / "model.joblib")
    thr = json.loads((Path(model_dir) / "model_card.json").read_text())["threshold"]
    return model, thr


def predict(texts, model, thr):
    if isinstance(texts, str):
        texts = [texts]
    probs = model.predict_proba(texts)[:, 1]
    return [{"text": t, "label": int(p >= thr), "prob_profanity": round(float(p), 4)} for t, p in zip(texts, probs)]


def show(r):
    name = "1 = Kazakh profanity present" if r["label"] else "0 = no Kazakh profanity"
    print(f"{name}   (P(profanity) = {r['prob_profanity']:.3f})   | {r['text']}")


if __name__ == "__main__":
    model, thr = load()
    if len(sys.argv) > 1:
        for r in predict(sys.argv[1:], model, thr):
            show(r)
    else:
        print(f"threshold = {thr}. Type a comment (empty line to quit).")
        while True:
            text = input("> ").strip()
            if not text:
                break
            show(predict(text, model, thr)[0])
