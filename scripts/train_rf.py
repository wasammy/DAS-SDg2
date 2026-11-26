"""Train RF-v2 (11 features) on seismograph data: all distances and per distance bin."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, ".")
from src.compare import FEATURES_11   #13-peak and crest_factor

BINS = [("<5", 0, 5), ("5-10", 5, 10), ("10-20", 10, 20.01)]
OUT_DIR = "data/models" 

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

    

    #cross-distance table (rows = trained on, columns = tested on)
    names = [b[0] for b in BINS]
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for tr_name, lo, hi in BINS:
        sub = in_bin(train_df, lo, hi)
        if sub["y"].nunique() < 2:
            print(f"bin {tr_name}: only one class in training, skipped")
            continue
        m = train(sub)
        joblib.dump(m, f"{OUT_DIR}/rf_{tr_name.replace('<', 'lt')}.joblib")
        for te_name, lo2, hi2 in BINS:
            table.loc[tr_name, te_name] = accuracy(m, in_bin(test_df, lo2, hi2))
    print("\nCross-distance accuracy (%), rows = trained on, columns = tested on")
    print((table * 100).round(1).to_string())