"""Train RF-v2 (11 features) on seismograph traces that pass the SNR rule, all distances and per bin.
Final models leave out the held-out explosion events; the cross-distance table is GroupKFold on everything."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, ".")
from src.compare import FEATURES_11  # 13 minus peak and crest_factor

BINS = [("<5", 0, 5), ("5-10", 5, 10), ("10-20", 10, 20.01)]
OUT_DIR = "data/models"
N_FOLDS = 5
HELDOUT = "data/features/heldout_une.csv"

def load_table(une_csv="data/features/gtune_une.csv", eq_csv="data/features/eq.csv"):
    df = pd.concat([pd.read_csv(une_csv), pd.read_csv(eq_csv)], ignore_index=True)
    df["event_id"] = df["event_id"].astype(str)
    df["y"] = (df["label"] == "UNE").astype(int)   # 1 = explosion, 0 = earthquake
    df = df[df["snr_pass"].astype(bool)].reset_index(drop=True)
    return df.dropna(subset=FEATURES_11)

def train(df, seed=0):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    return rf.fit(df[FEATURES_11], df["y"])


def accuracy(rf, df):
    return np.nan if len(df) == 0 else float((rf.predict(df[FEATURES_11]) == df["y"]).mean())


def in_bin(df, lo, hi):
    return df[(df["dist_deg"] >= lo) & (df["dist_deg"] < hi)]


def cross_distance_cv(df, n_folds=N_FOLDS, seed=0):
    # grouped by event; returns (mean, std, n folds with test data), rows = trained on
    names = [b[0] for b in BINS]
    scores = {(r, c): [] for r in names for c in names}
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in gkf.split(df, df["y"], groups=df["event_id"]):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        for tr_name, lo, hi in BINS:
            sub = in_bin(tr_df, lo, hi)
            if sub["y"].nunique() < 2:
                continue  # only one class in this bin
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
    held = set(pd.read_csv(HELDOUT)["event_id"].astype(str))
    train_df, test_df = df[~df["event_id"].isin(held)], df[df["event_id"].isin(held)]
    print(f"{len(held)} held-out explosion events ({len(test_df)} traces) kept out of the final models")
    print(f"train: {len(train_df)} traces, {train_df['event_id'].nunique()} events, "
          f"{int(train_df['y'].sum())} UNE / {int((train_df['y'] == 0).sum())} EQ")

    rf = train(train_df)
    print(f"\nAll-distance model, held-out explosions called UNE: {accuracy(rf, test_df):.1%} of {len(test_df)}")
    joblib.dump(rf, f"{OUT_DIR}/rf_all.joblib")
    imp = pd.Series(rf.feature_importances_, index=FEATURES_11).sort_values(ascending=False)
    print("Top features:\n", imp.head(5).round(3).to_string())

    for tr_name, lo, hi in BINS:
        sub = in_bin(train_df, lo, hi)
        if sub["y"].nunique() < 2:
            print(f"bin {tr_name}: only one class in training, no model saved")
            continue
        m = train(sub)
        joblib.dump(m, f"{OUT_DIR}/rf_{tr_name.replace('<', 'lt')}.joblib")
        te = in_bin(test_df, lo, hi)
        print(f"bin {tr_name} model ({len(sub)} traces), held-out explosions called UNE: "
              + (f"{accuracy(m, te):.1%} of {len(te)}" if len(te) else "none in bin"))

    print("\nTraces (events) per distance bin")
    for name, lo, hi in BINS:
        b = in_bin(df, lo, hi)
        eq, une = b[b["y"] == 0], b[b["y"] == 1]
        print(f"  {name:>5}: EQ {len(eq)} ({eq['event_id'].nunique()})   "
              f"UNE {len(une)} ({une['event_id'].nunique()})")

    mean, std, n = cross_distance_cv(df)
    # plain +/-, the Windows console chokes on the real sign
    cell = (mean * 100).round(1).astype(str) + " +/- " + (std * 100).round(1).astype(str)
    cell = cell.where(n > 0, "n/a")
    print(f"\nCross-distance accuracy (%), mean +/- std over {N_FOLDS} GroupKFold folds")
    print("rows = trained on, columns = tested on")
    print(cell.to_string())
    if (n < N_FOLDS).to_numpy().any():
        print("folds with test data per cell (fewer than 5 = a bin was empty in some folds):")
        print(n.astype(int).to_string())
