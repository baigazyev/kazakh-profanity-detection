"""Build the final labelled dataset and the train / validation / test split.

    python src/prepare_data.py

Steps
1. Source comments: data/source/kazakh_profanity_share.csv (KazSAnDRA comments with profanity labels),
   relabelled by data/label_overrides.csv according to the rules in README (rule C).
2. Product-review negatives: Market-domain reviews from KazSAnDRA (downloaded from its GitHub repo),
   so that the data actually contains product reviews, negative-but-clean reviews and look-alike words.
3. Cleaning: PII placeholders, exact-duplicate removal.
4. Near-duplicate groups (character TF-IDF cosine >= threshold) so that near-duplicates share a split.
5. Stratified, grouped split with a fixed seed. Writes data/processed/*.csv and data/processed/stats.json.
"""
import io, json, re, sys, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lexicon import profane_hits          # noqa: E402  (audit only: keeps profane text out of the added negatives)
from textnorm import mask_pii, normalize  # noqa: E402

CFG = json.loads((ROOT / "config.json").read_text())
SEED = CFG["seed"]
D = CFG["data"]
EXT = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed"

# Full names inside comments ("Нурасыл Толысбаев") -> <name>
_UP, _LO = "А-ЯӘІҢҒҮҰҚӨҺ", "а-яәіңғүұқөһ"
NAME_RE = re.compile(rf"\b(?:[{_UP}][{_LO}]+\s+[{_UP}][{_LO}]+(?:ов|ев|ова|ева|ұлы|улы|қызы|кызы)"
                     rf"|[{_UP}]{{3,}}\s+[{_UP}]{{3,}}(?:ОВ|ЕВ|ОВА|ЕВА|ҰЛЫ|УЛЫ|ҚЫЗЫ|КЫЗЫ))\b")
LOOKALIKE_RE = re.compile(r"(?:есік|бесік|тесік|кесік|көтер|сигнал|шешім|шешендік|қатыным|амал)")


def clean_text(t: str) -> str:
    t = mask_pii(str(t))
    t = NAME_RE.sub("<name>", t)
    return re.sub(r"\s+", " ", t).strip()


def load_kazsandra() -> pd.DataFrame:
    EXT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in D["kazsandra_files"]:
        path = EXT / name
        if not path.exists():
            print("downloading", name)
            urllib.request.urlretrieve(D["kazsandra_base_url"] + name, path)
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if member.endswith(".csv"):
                    frames.append(pd.read_csv(io.BytesIO(z.read(member))))
    return pd.concat(frames, ignore_index=True)


def source_comments() -> pd.DataFrame:
    df = pd.read_csv(ROOT / D["source_csv"])
    ov = pd.read_csv(ROOT / D["overrides_csv"]).set_index("kazsandra_id")
    df["label"] = [int(ov.at[i, "new_label"]) if i in ov.index else int(l) for i, l in zip(df.kazsandra_id, df.label)]
    df["uncertain"] = [int(ov.at[i, "uncertain"]) if i in ov.index else 0 for i in df.kazsandra_id]
    df["source"] = "share_csv"
    return df[["kazsandra_id", "text", "label", "source", "uncertain"]]


def market_negatives(exclude_ids) -> pd.DataFrame:
    k = load_kazsandra()
    m = k[(k["domain"] == "market") & ~k["custom_id"].isin(exclude_ids)].drop_duplicates("custom_id")
    m = m[m["text"].map(lambda t: len(profane_hits(t)) == 0)]          # conservative: no lexicon hit at all
    rng = np.random.RandomState(SEED)
    look = m[m["text"].str.lower().str.contains(LOOKALIKE_RE)]
    look = look.sample(min(D["market_lookalike_reviews"], len(look)), random_state=rng)
    rest = m.drop(look.index)
    neg = rest[rest["label"] == 0].sample(D["market_negative_reviews"], random_state=rng)
    pos = rest[rest["label"] == 1].drop(neg.index, errors="ignore").sample(D["market_positive_reviews"], random_state=rng)
    out = pd.concat([look.assign(kind="look-alike word"), neg.assign(kind="negative review"),
                     pos.assign(kind="positive review")])
    return pd.DataFrame({"kazsandra_id": out["custom_id"], "text": out["text"], "label": 0,
                         "source": "kazsandra_market:" + out["kind"], "uncertain": 0})


def near_duplicate_groups(texts) -> np.ndarray:
    """Union-find over pairs with character TF-IDF cosine >= threshold (or identical normalised text)."""
    norm = [normalize(t) for t in texts]
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(norm)
    nn = NearestNeighbors(radius=1 - D["near_duplicate_cosine"], metric="cosine").fit(X)
    _, idx = nn.radius_neighbors(X)
    parent = list(range(len(norm)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, nbrs in enumerate(idx):
        for j in nbrs:
            parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(norm))])


def main():
    src = source_comments()
    mk = market_negatives(set(src.kazsandra_id))
    df = pd.concat([src, mk], ignore_index=True)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].map(normalize).str.len() > 0]

    key = df["text"].map(normalize).str.replace(" ", "", regex=False)
    conflict = key[df.groupby(key)["label"].transform("nunique") > 1]
    n_before = len(df)
    df = df[~key.isin(conflict)]                                   # identical text with different labels: drop
    df = df.loc[~df["text"].map(normalize).str.replace(" ", "", regex=False).duplicated()].reset_index(drop=True)
    print(f"removed {n_before - len(df)} exact duplicates / conflicting copies")

    df["group"] = near_duplicate_groups(df["text"].tolist())
    n_groups = df["group"].nunique()
    print(f"{len(df)} rows in {n_groups} near-duplicate groups ({len(df) - n_groups} rows share a group)")

    # stratified + grouped: test first, then validation from the rest
    df["split"] = "train"
    n_test = round(1 / D["test_fraction"])
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=SEED)
    _, test_idx = next(sgkf.split(df, df["label"], df["group"]))
    df.loc[test_idx, "split"] = "test"
    rest = df[df["split"] == "train"]
    n_val = round((1 - D["test_fraction"]) / D["val_fraction"])
    sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=SEED)
    _, val_pos = next(sgkf.split(rest, rest["label"], rest["group"]))
    df.loc[rest.index[val_pos], "split"] = "val"
    assert df.groupby("group")["split"].nunique().max() == 1, "a near-duplicate group crosses splits"

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["kazsandra_id", "text", "label", "source", "uncertain", "group", "split"]
    df[cols].to_csv(OUT / "dataset.csv", index=False)
    for s in ("train", "val", "test"):
        df[df["split"] == s][cols].to_csv(OUT / f"{s}.csv", index=False)

    stats = {"seed": SEED, "total": int(len(df)),
             "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
             "by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
             "splits": {s: {"n": int((df.split == s).sum()), "profanity": int(((df.split == s) & (df.label == 1)).sum()),
                            "clean": int(((df.split == s) & (df.label == 0)).sum())} for s in ("train", "val", "test")},
             "near_duplicate_groups": int(n_groups), "uncertain_labels": int(df["uncertain"].sum()),
             "synthetic_examples": 0}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
