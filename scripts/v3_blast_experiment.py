"""v3 blast experiment, step 3: blast vs earthquake near Ridgecrest (v3b data). Baselines and CV on v3 alone,
then a 50-model under-5 deg ensemble on v2 + v3, run once on Ridgecrest. Experiment only; v2 is the headline model.
Usage: python scripts/v3_blast_experiment.py   -> data/predictions/ridgecrest_v3b_ensemble.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path[:0] = [".", "scripts"]
import build_features as B
import train_rf as T
from src.compare import FEATURES_11
pd.set_option("display.width", 200)

N_SEEDS, FRAC, FAR_KM = 50, 0.8, 90
DATA = "data/gtune"
OUT = "data/predictions/ridgecrest_v3b_ensemble.csv"


def v3_table(eq_file="v3b_eqs.h5"):
    parts = []
    for f, label in (("v3_blasts.h5", "UNE"), (eq_file, "EQ")):
        raw = pd.read_hdf(os.path.join(DATA, f), "df").drop_duplicates(["evid", "net", "station"]).reset_index(drop=True)
        t = B.with_snr(raw.drop(columns=["year"]), label)
        info = raw.assign(event_id=raw["evid"].astype(str))[["event_id", "net", "station", "mag", "year"]]
        parts.append(t.merge(info, on=["event_id", "net", "station"]))
    df = pd.concat(parts, ignore_index=True).dropna(subset=FEATURES_11)
    df["y"] = (df["label"] == "UNE").astype(int)
    df["source"] = np.where(df["y"] == 1, "quarry blast", "EQ-v3")
    return df


def rf(seed=0):
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)


def cv(df, inputs, n_folds=5, seed=0):
    res = {k: [] for k in inputs}
    for tr, te in GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df, df["y"], groups=df["event_id"]):
        ytr, yte = df["y"].iloc[tr], df["y"].iloc[te].to_numpy()
        for k, X in inputs.items():
            p = rf().fit(X.iloc[tr], ytr).predict(X.iloc[te])
            res[k].append(((p == yte).mean(), (p[yte == 0] == 0).mean(), (p[yte == 1] == 1).mean()))
    out = {k: [f"{a.mean():.1f} +/- {a.std():.1f}" for a in (np.array(v) * 100).T] for k, v in res.items()}
    return pd.DataFrame(out, index=["accuracy", "EQ correct", "UNE correct"]).T


def cv_by_source(df, n_folds=5, seed=0):
    pred = pd.Series(index=df.index, dtype=float)
    for tr, te in GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df, df["y"], groups=df["event_id"]):
        pred.iloc[te] = rf().fit(df[FEATURES_11].iloc[tr], df["y"].iloc[tr]).predict(df[FEATURES_11].iloc[te])
    ok = pred == df["y"]
    print(f"  overall {ok.mean():.1%} of {len(df)}")
    for s, g in df.groupby("source"):
        print(f"  {s:>14}: {ok[g.index].mean():.1%} correct of {len(g)}")


def describe(df):
    for lab, g in df.groupby("label"):
        p = g[g["snr_pass"]]
        print(f"  {lab}: {len(g)} traces, {g['event_id'].nunique()} events, {g['station'].nunique()} stations; "
              f"pass SNR {len(p)} ({len(p) / len(g):.0%}), {p['event_id'].nunique()} events; "
              f"M{p['mag'].min():.1f}-{p['mag'].max():.1f} (median {p['mag'].median():.2f}); "
              f"dist {p['dist_deg'].min():.2f}-{p['dist_deg'].max():.2f} deg (median {p['dist_deg'].median():.2f}); "
              f"years {p['year'].min()}-{p['year'].max()}")


def event_probs(m, blocks):
    return blocks.assign(p=m.predict_proba(blocks[FEATURES_11])[:, 1]).groupby("event_id", sort=False)["p"].median()


if __name__ == "__main__":
    v3 = v3_table()
    print("v3 data:")
    describe(v3)
    d = v3[v3["snr_pass"]].reset_index(drop=True)
    inputs = {"11 features": d[FEATURES_11], "station only": pd.get_dummies(d["station"]).astype(float),
              "year only": d[["year"]], "distance only": d[["dist_deg"]], "magnitude only": d[["mag"]]}
    print("\nv3 alone (SNR-passing), 5-fold GroupKFold by event (mean +/- std, %):")
    print(cv(d, inputs).to_string())

    v2 = T.load_table()
    held = set(pd.read_csv(T.HELDOUT)["event_id"].astype(str))
    v2 = T.in_bin(v2[~v2["event_id"].isin(held)], 0, 5)
    v2["source"] = np.where(v2["y"] == 1, "GTUNE", "EQ-v2")
    both = pd.concat([v2, d[v2.columns.intersection(d.columns)]], ignore_index=True)
    print(f"\nv2 + v3 under-5 training data: {len(both)} traces, {both['event_id'].nunique()} events; "
          + str(both.groupby("source").size().to_dict()))
    print("5-fold GroupKFold, 11 features, share correct by source:")
    cv_by_source(both)

    blocks = pd.read_csv("data/features/ridgecrest_blocks.csv")
    checks = pd.read_csv("data/stack_checks_ridgecrest.csv")
    checks["event_id"] = checks["file"].str.replace(".h5", "", regex=False)
    events = both["event_id"].unique()
    probs = []
    for s in range(N_SEEDS):
        pick = np.random.default_rng(s).choice(events, size=int(round(FRAC * len(events))), replace=False)
        sub = both[both["event_id"].isin(pick)]
        probs.append(event_probs(rf(s).fit(sub[FEATURES_11], sub["y"]), blocks))
    P = pd.concat(probs, axis=1)
    c = (P <= 0.5).sum().to_numpy()
    print(f"\nRidgecrest, {N_SEEDS} v2+v3 models: correct of {len(P)} median {np.median(c):.0f}, "
          f"5th-95th percentile {np.percentile(c, 5):.0f}-{np.percentile(c, 95):.0f}, min {c.min()}, max {c.max()}")
    ens = P.mean(axis=1)
    pd.DataFrame({"event_id": ens.index, "dataset": "ridgecrest", "true": "EQ",
                  "pred": np.where(ens > 0.5, "UNE", "EQ"), "prob_une": ens.round(3).values}
                 ).to_csv(OUT, index=False)
    v2e = pd.read_csv("data/predictions/ridgecrest_lt5_ensemble.csv").set_index("event_id")["prob_une"]
    per = pd.DataFrame({"share_EQ": (P <= 0.5).mean(axis=1), "v3_P_UNE": ens, "v2_P_UNE": v2e.reindex(ens.index)})
    per = per.join(checks.set_index("event_id")[["mag", "dist_km"]])
    per["group"] = np.where(per["dist_km"] >= FAR_KM, f"far (>={FAR_KM} km)", "close")
    per = per.sort_values(["group", "share_EQ"], ascending=[False, True])
    for g, x in per.groupby("group", sort=False):
        print(f"\n{g}: {len(x)} events; v3 ensemble calls EQ for {int((x['v3_P_UNE'] <= 0.5).sum())} "
              f"(v2 ensemble: {int((x['v2_P_UNE'] <= 0.5).sum())})")
        print(x[["mag", "dist_km", "share_EQ", "v3_P_UNE", "v2_P_UNE"]].round(2).to_string())
