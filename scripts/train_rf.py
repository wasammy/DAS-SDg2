"""Train RF-v2 (11 features) on seismograph data: all distances and per distance bin."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, ".")
from src.compare import FEATURES_11   #13-peak and crest_factor

BINS = [("<5", 0, 5), ("5-10", 5, 10), ("10-20", 10, 20.01)]
OUT_DIR = "data/models"
N_FOLDS = 5

def load_table(une_csv="data/features/gtune_une.csv", eq_csv="data/features/eq.csv"):
    df = pd.concat([pd.read_csv(une_csv), pd.read_csv(eq_csv)], ignore_index=True)
    df["event_id"] = df["event_id"].astype(str)
    df["y"] = (df["label"] == "UNE").astype(int)   # 1 = explosion, 0 = earthquake
    return df.dropna(subset=FEATURES_11)

def split_by_event(df, test_frac=0.2, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(gss.split(df, df["y"], groups=df["event_id"]))
    return df.iloc[tr], df.iloc[te]

def train(df):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=0, n_jobs=-1)
    return rf.fit(df[FEATURES_11], df["y"])


def accuracy(rf, df):
    return np.nan if len(df) == 0 else float((rf.predict(df[FEATURES_11]) == df["y"]).mean())


def class_rates(rf, df):
    pred = rf.predict(df[FEATURES_11])
    eq, une = df["y"] == 0, df["y"] == 1
    return float((pred[eq] == 0).mean()), float((pred[une] == 1).mean())


def in_bin(df, lo, hi):
    return df[(df["dist_deg"] >= lo) & (df["dist_deg"] < hi)]


def cross_distance_cv(df, n_folds=N_FOLDS, seed=0):
    """Cross-distance accuracy over n_folds GroupKFold folds (grouped by event,
    so no event is in both train and test). Returns (mean, std, n) tables:
    rows = trained on, columns = tested on, n = folds that had test data."""
    names = [b[0] for b in BINS]
    scores = {(r, c): [] for r in names for c in names}
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in gkf.split(df, df["y"], groups=df["event_id"]):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        for tr_name, lo, hi in BINS:
            sub = in_bin(tr_df, lo, hi)
            if sub["y"].nunique() < 2:
                continue   # only one class in this training bin
            m = train(sub)
            for te_name, lo2, hi2 in BINS:
                acc = accuracy(m, in_bin(te_df, lo2, hi2))
                if not np.isnan(acc):
                    scores[(tr_name, te_name)].append(acc)
    mean = pd.DataFrame(index=names, columns=names, dtype=float)
    std, n = mean.copy(), mean.copy()
    for (r, c), v in scores.items():
        mean.loc[r, c] = np.mean(v) if v else np.nan
        std.loc[r, c] = np.std(v) if v else np.nan
        n.loc[r, c] = len(v)
    return mean, std, n

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_table()
    train_df, test_df = split_by_event(df)
    for name, d in (("train", train_df), ("test", test_df)):
        print(f"{name}: {len(d)} traces, {d['event_id'].nunique()} events, "
              f"{int(d['y'].sum())} UNE / {int((d['y'] == 0).sum())} EQ")

    # all distances
    rf = train(train_df)
    print(f"\nAll-distance model, held-out accuracy: {accuracy(rf, test_df):.1%}")
    eq_ok, une_ok = class_rates(rf, test_df)
    print(f"  earthquakes correct: {eq_ok:.1%}   explosions correct: {une_ok:.1%}")
    joblib.dump(rf, f"{OUT_DIR}/rf_all.joblib")
    imp = pd.Series(rf.feature_importances_, index=FEATURES_11).sort_values(ascending=False)
    print("Top features:\n", imp.head(5).round(3).to_string())

    #for Himi's synthetic DAS test
    held = test_df[test_df["y"] == 1][["event_id", "net", "station", "dist_deg"]]
    held.to_csv("data/features/heldout_une.csv", index=False)
    print(f"\nSaved {len(held)} held-out explosion traces for the synthetic DAS test")

    

    # per-bin models from the fixed split, saved for predict_das.py
    for tr_name, lo, hi in BINS:
        sub = in_bin(train_df, lo, hi)
        if sub["y"].nunique() < 2:
            print(f"bin {tr_name}: only one class in training, no model saved")
            continue
        joblib.dump(train(sub), f"{OUT_DIR}/rf_{tr_name.replace('<', 'lt')}.joblib")

    # traces and events per bin and class, so thin cells are visible
    print("\nTraces (events) per distance bin")
    for name, lo, hi in BINS:
        b = in_bin(df, lo, hi)
        eq, une = b[b["y"] == 0], b[b["y"] == 1]
        print(f"  {name:>5}: EQ {len(eq)} ({eq['event_id'].nunique()})   "
              f"UNE {len(une)} ({une['event_id'].nunique()})")

    # cross-distance table, 5-fold GroupKFold by event (rows = trained on, columns = tested on)
    mean, std, n = cross_distance_cv(df)
    # "+/-" rather than the plus-minus sign: the Windows console can't print it
    cell = (mean * 100).round(1).astype(str) + " +/- " + (std * 100).round(1).astype(str)
    cell = cell.where(n > 0, "n/a")
    print(f"\nCross-distance accuracy (%), mean +/- std over {N_FOLDS} GroupKFold folds")
    print("rows = trained on, columns = tested on")
    print(cell.to_string())
    if (n < N_FOLDS).to_numpy().any():
        print("folds with test data per cell (fewer than 5 = a bin was empty in some folds):")
        print(n.astype(int).to_string())