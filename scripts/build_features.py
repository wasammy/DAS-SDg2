"""Build data/features/gtune_une.csv and eq.csv for train_rf.py, with an snr_pass column
(after/before ratio at P >= 1.2x the best fake window, the same rule as the DAS test set).
Earthquake records whose USGS catalog type is not "earthquake" are dropped.
Usage: python scripts/build_features.py [--eq-only]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from src.preprocess import bandpass
from src.training_set import load_gtune, feature_rows, MAX_DIST_DEG, MIN_FS, P_OFFSET_S

EQ_FILES = ["data/gtune/eqs_iris.h5", "data/gtune/eqs_near.h5", "data/gtune/eqs_matched.h5"]
OUT_DIR = "data/features"
RATIO_FACTOR = 1.2
WIN_S, GAP_S = 3.0, 1.0
FAKE_SPLITS_S = (4.0, 5.0, 6.0)
TYPES_CSV = "data/gtune/eq_catalog_types.csv"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def catalog_type(t):
    r = requests.get(USGS, timeout=60, params=dict(
        format="geojson", minmagnitude=2.5,
        starttime=pd.Timestamp(t - 2, unit="s").isoformat(),
        endtime=pd.Timestamp(t + 2, unit="s").isoformat()))
    feats = r.json().get("features", [])
    if not feats:
        return "not found"
    return min(feats, key=lambda f: abs(f["properties"]["time"] / 1000 - t))["properties"]["type"]


def earthquakes_only(df):
    known = {}
    if os.path.exists(TYPES_CSV):
        c = pd.read_csv(TYPES_CSV)
        known = dict(zip(c["evid"].astype(str), c["type"]))
    ev = df.drop_duplicates("evid")
    for e, t in zip(ev["evid"].astype(str), ev["origin_time"]):
        if e not in known:
            known[e] = catalog_type(float(t))
    pd.DataFrame({"evid": list(known), "type": list(known.values())}).to_csv(TYPES_CSV, index=False)
    bad = {e: ty for e, ty in known.items() if ty not in ("earthquake", "not found")}
    drop = df["evid"].astype(str).isin(bad)
    for e in sorted(set(df.loc[drop, "evid"].astype(str))):
        print(f"dropped {e}: catalog type {bad[e]}")
    return df[~drop]


def load_eqs():
    parts = []
    for f in EQ_FILES:
        if os.path.exists(f):
            d = pd.read_hdf(f, "df")
            print(f"{f}: {len(d)} records")
            parts.append(d)
        else:
            print(f"{f}: not found, skipped")
    df = pd.concat(parts, ignore_index=True)
    df = df[df["station_dist"] <= MAX_DIST_DEG]
    df = df[df["sampling_rate"] >= MIN_FS - 0.01]
    df = df[df["waveform"].apply(lambda w: w is not None and len(w) > 0)]
    df = df.drop_duplicates(subset=["evid", "net", "station", "chn"])
    df = df[df["waveform"].apply(len) >= 59.9 * df["sampling_rate"]]
    df = earthquakes_only(df)
    return df.reset_index(drop=True)


def after_before(x, fs, split_s):
    i = lambda s: int(round(s * fs))
    after = x[i(split_s):i(split_s + WIN_S)].std()
    before = x[i(split_s - GAP_S - WIN_S):i(split_s - GAP_S)].std()
    return after / before if before > 0 else np.nan


def snr_keys(df):
    keep = set()
    for _, r in df.iterrows():
        fs = float(r["sampling_rate"])
        x = np.asarray(r["waveform"], dtype=float)[: int(60 * fs)]
        if x.std() == 0:
            continue
        y = bandpass(x, fs)
        fake = np.nanmax([after_before(y, fs, s) for s in FAKE_SPLITS_S])
        if after_before(y, fs, P_OFFSET_S) >= RATIO_FACTOR * fake:
            keep.add((str(r["evid"]), r["net"], r["station"]))
    return keep


def with_snr(raw, label):
    t = feature_rows(raw, label)
    keep = snr_keys(raw)
    t["snr_pass"] = [k in keep for k in zip(t["event_id"], t["net"], t["station"])]
    return t


def summary(name, t):
    bins = pd.cut(t["dist_deg"], [0, 5, 10, 20.01], right=False,
                  labels=["<5", "5-10", "10-20"]).value_counts().sort_index()
    print(f"{name}: {len(t)} traces, {t['event_id'].nunique()} events, "
          f"{t['station'].nunique()} stations; per bin " + str(bins.to_dict())
          + f"; {int(t['snr_pass'].sum())} pass SNR")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.eq_only:
        une = with_snr(load_gtune(), "UNE")
        une.to_csv(f"{OUT_DIR}/gtune_une.csv", index=False)
        summary("UNE", une)

    eq = with_snr(load_eqs(), "EQ")
    eq.to_csv(f"{OUT_DIR}/eq.csv", index=False)
    summary("EQ", eq)
