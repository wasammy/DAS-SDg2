"""Strain-rate experiment: convert every training seismogram to strain rate (das_convert) before preprocessing,
retrain the v2 and v2+v3 50-model ensembles on the same traces and seeds (Himi's 12 held-out events excluded),
and score the raw Ridgecrest DAS blocks.
Usage: python scripts/strainrate_experiment.py   -> data/models/rf_lt5_srtrained_ensemble.joblib,
       data/predictions/ridgecrest_{v2,v3b}_srtrained_ensemble.csv
"""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SR_MODELS = "data/models/rf_lt5_srtrained_ensemble.joblib"  # list of 50 RFs, average P(UNE)
sys.path[:0] = [".", "scripts"]
import build_features as B
import predict_das as PD
import train_rf as T
import v3_blast_experiment as V
from src.das_convert import velocity_to_strain_rate
from src.training_set import load_gtune, feature_rows
from src.compare import FEATURES_11


def converted(raw, label):
    raw = raw.reset_index(drop=True)
    sr = raw.copy()
    sr["waveform"] = [velocity_to_strain_rate(np.asarray(w, dtype=float), float(fs))
                      for w, fs in zip(raw["waveform"], raw["sampling_rate"])]
    t = feature_rows(sr, label)
    keep = B.snr_keys(raw)  # SNR decided on the original trace, so the trace set is unchanged
    t["snr_pass"] = [k in keep for k in zip(t["event_id"], t["net"], t["station"])]
    return t


def check_same_rows(new, committed_csv):
    old = pd.read_csv(committed_csv)
    k = ["event_id", "net", "station"]
    same = len(old) == len(new) and (old[k].astype(str).values == new[k].astype(str).values).all() \
        and (old["snr_pass"].values == new["snr_pass"].values).all()
    print(f"  {os.path.basename(committed_csv)}: {len(new)} rows, same rows/order/snr_pass as committed: {same}")
    return same


def table(une, eq):
    df = pd.concat([une, eq], ignore_index=True)
    df["event_id"] = df["event_id"].astype(str)
    df["y"] = (df["label"] == "UNE").astype(int)
    df = df[df["snr_pass"].astype(bool)].reset_index(drop=True)
    return df.dropna(subset=FEATURES_11)


def ensemble(both, label):
    events = both["event_id"].unique()
    models = []
    for s in range(V.N_SEEDS):
        pick = np.random.default_rng(s).choice(events, size=int(round(V.FRAC * len(events))), replace=False)
        sub = both[both["event_id"].isin(pick)]
        models.append(T.train(sub, seed=s) if label == "v2" else V.rf(s).fit(sub[FEATURES_11], sub["y"]))
    return models


if __name__ == "__main__":
    print("converting v2 training seismograms:")
    une = converted(load_gtune(), "UNE")
    eq = converted(B.load_eqs(), "EQ")
    ok = check_same_rows(une, "data/features/gtune_une.csv") & check_same_rows(eq, "data/features/eq.csv")
    v2 = table(une, eq)
    held = set(pd.read_csv(T.HELDOUT)["event_id"].astype(str))
    v2 = T.in_bin(v2[~v2["event_id"].isin(held)], 0, 5)
    ref = T.in_bin(T.load_table(), 0, 5)
    ref = ref[~ref["event_id"].isin(held)]
    print(f"  v2 <5 training: {len(v2)} traces; event order same as committed: {list(v2['event_id'].unique()) == list(ref['event_id'].unique())}")

    print("converting v3 seismograms:")
    parts = []
    for f, label in (("v3_blasts.h5", "UNE"), ("v3b_eqs.h5", "EQ")):
        raw = pd.read_hdf(os.path.join(V.DATA, f), "df").drop_duplicates(["evid", "net", "station"]).reset_index(drop=True)
        t = converted(raw.drop(columns=["year"]), label)
        info = raw.assign(event_id=raw["evid"].astype(str))[["event_id", "net", "station", "mag", "year"]]
        parts.append(t.merge(info, on=["event_id", "net", "station"]))
    d = pd.concat(parts, ignore_index=True).dropna(subset=FEATURES_11)
    d["y"] = (d["label"] == "UNE").astype(int)
    d["source"] = np.where(d["y"] == 1, "quarry blast", "EQ-v3")
    d = d[d["snr_pass"]].reset_index(drop=True)
    d0 = V.v3_table()
    d0 = d0[d0["snr_pass"]].reset_index(drop=True)
    print(f"  v3 SNR-passing: {len(d)} traces; same rows as committed v3: {(d[['event_id','station']].values == d0[['event_id','station']].values).all()}")
    v2s = v2.copy()
    v2s["source"] = np.where(v2s["y"] == 1, "GTUNE", "EQ-v2")
    both = pd.concat([v2s, d[v2s.columns.intersection(d.columns)]], ignore_index=True)

    blocks = pd.read_csv("data/features/ridgecrest_blocks.csv")
    checks = pd.read_csv("data/stack_checks_ridgecrest.csv")
    far = set(checks.loc[checks["dist_km"] >= PD.FAR_KM, "file"].str.replace(".h5", "", regex=False))
    print("\nspectral_centroid median (IQR):")
    for name, x in (("seismometer <5 EQ, strain rate", v2[v2.y == 0]), ("seismometer <5 UNE, strain rate", v2[v2.y == 1]),
                    ("v3 quarry blasts, strain rate", d[d.y == 1]), ("v3 EQs, strain rate", d[d.y == 0]),
                    ("Ridgecrest DAS blocks (raw)", blocks)):
        s = x["spectral_centroid"]
        print(f"  {name:>32}: {s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})")

    print("\nRidgecrest raw blocks, ensembles trained on strain-rate seismograms:")
    for tag, train_df in (("v2", v2), ("v3b", both)):
        models = ensemble(train_df, tag)
        if tag == "v2":
            joblib.dump(models, SR_MODELS, compress=3)
            print(f"  saved {SR_MODELS}")
        P = pd.concat([PD.event_probs(m, blocks) for m in models], axis=1)
        ens = P.mean(axis=1)
        okk = ens <= 0.5
        c = (P <= 0.5).sum().to_numpy()
        print(f"  {tag:>4}: {int(okk.sum())}/41 EQ (far {int(okk[ens.index.isin(far)].sum())}/10, close "
              f"{int(okk[~ens.index.isin(far)].sum())}/31); 50 models median {np.median(c):.0f}, 5th-95th "
              f"{np.percentile(c, 5):.0f}-{np.percentile(c, 95):.0f}; P(UNE) {ens.min():.2f}-{ens.max():.2f}")
        pd.DataFrame({"event_id": ens.index, "dataset": "ridgecrest", "true": "EQ",
                      "pred": np.where(okk, "EQ", "UNE"), "prob_une": ens.round(3).values}
                     ).to_csv(f"data/predictions/ridgecrest_{tag}_srtrained_ensemble.csv", index=False)
