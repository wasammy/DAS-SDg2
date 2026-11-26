"""Build the seismograph feature tables for train_rf.py.

  data/features/gtune_une.csv : GTUNE explosions (load_gtune), label UNE
  data/features/eq.csv        : EarthScope earthquakes from download_eqs.py and
                                download_eqs_near.py, label EQ

The earthquake records get the same quality filters as load_gtune (<= 20 deg,
>= 20 Hz, full 60 s, no duplicates), so both classes pass the same checks.

Usage, from the top of the repo:
    python scripts/build_features.py            # both tables
    python scripts/build_features.py --eq-only  # keep gtune_une.csv as it is
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.training_set import load_gtune, feature_rows, MAX_DIST_DEG, MIN_FS

EQ_FILES = ["data/gtune/eqs_iris.h5", "data/gtune/eqs_near.h5"]
OUT_DIR = "data/features"


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
    return df.reset_index(drop=True)


def summary(name, t):
    bins = pd.cut(t["dist_deg"], [0, 5, 10, 20.01], right=False,
                  labels=["<5", "5-10", "10-20"]).value_counts().sort_index()
    print(f"{name}: {len(t)} traces, {t['event_id'].nunique()} events, "
          f"{t['station'].nunique()} stations; per bin " + str(bins.to_dict()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.eq_only:
        une = feature_rows(load_gtune(), "UNE")
        une.to_csv(f"{OUT_DIR}/gtune_une.csv", index=False)
        summary("UNE", une)

    eq = feature_rows(load_eqs(), "EQ")
    eq.to_csv(f"{OUT_DIR}/eq.csv", index=False)
    summary("EQ", eq)
