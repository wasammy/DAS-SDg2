"""Classify Ridgecrest DAS events: 50-channel blocks with a detected arrival, median P(UNE) per event.
Saved models (all, <5) plus a 50-model <5 ensemble; outputs in evaluate.py's format.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.das_io import load_das_event
from src.stacking import substack
from src.preprocess import bandpass, preprocess_trace
from src.features import extract_features
from src.compare import FEATURES_11
sys.path.insert(0, "scripts")
from stack_checks import after_before, fake_splits_before
from train_rf import load_table, in_bin, train, HELDOUT

DATA_DIR = "data/ridgecrest_north/data"
CHECKS = "data/stack_checks_ridgecrest.csv"
MODELS = {"all": "data/models/rf_all.joblib", "lt5": "data/models/rf_lt5.joblib"}

RATIO_FACTOR = 1.2  # ratio at P must be >= 1.2x the best fake window
MAX_ONSET_OFFSET_S = 2.0  # observed onset within 2 s of predicted P
N_SEEDS = 50
FRAC = 0.8  # share of training events each ensemble model sees
FAR_KM = 90

def test_events():
    c = pd.read_csv(CHECKS)
    keep = c[(c["fits_window"] == True)
             & (c["ratio_at_P"] >= RATIO_FACTOR * c["fake_same_file_max"])
             & (c["onset_minus_P_s"].abs() <= MAX_ONSET_OFFSET_S)]
    print(f"{len(keep)} of {len(c)} Ridgecrest files pass the strict rule "
          f"(ratio >= {RATIO_FACTOR}x best fake, onset within {MAX_ONSET_OFFSET_S:.0f} s)")
    return keep

def block_features(path, p_s):
    data, meta = load_das_event(path)
    fs = meta["fs"]
    blocks, counts = substack(data, fs, group=50, filtered=False)  # preprocess_trace filters it once

    filt = np.array([bandpass(b, fs) for b in blocks])  # filtered copy only for detection
    splits = fake_splits_before(p_s)
    if not splits:
        return pd.DataFrame()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = after_before(filt, fs, p_s)
        fakes = np.array([after_before(filt, fs, s) for s in splits])
    detected = ratio >= RATIO_FACTOR * fakes.max(axis=0)

    rows = []
    for b, (x, n) in enumerate(zip(blocks, counts)):
        if n == 0 or not detected[b]:
            continue   # dead block, or no arrival visible in it
        try:
            win, _ = preprocess_trace(x, fs, p_index=round(p_s * fs))
        except ValueError:
            continue
        rows.append({"block": b, "ratio": float(ratio[b]), **extract_features(win)})
    return pd.DataFrame(rows)

def ensemble_models():
    df = load_table()
    held = set(pd.read_csv(HELDOUT)["event_id"].astype(str))
    tr5 = in_bin(df[~df["event_id"].isin(held)], 0, 5)
    events = tr5["event_id"].unique()
    models = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(s)
        picked = rng.choice(events, size=int(round(FRAC * len(events))), replace=False)
        sub = tr5[tr5["event_id"].isin(picked)]
        models.append(train(sub, seed=s))
    return models

def event_probs(m, blocks):
    p = blocks.assign(p=m.predict_proba(blocks[FEATURES_11])[:, 1])
    return p.groupby("event_id", sort=False)["p"].median()

def run_ensemble(blocks, info):
    P = pd.concat([event_probs(m, blocks) for m in ensemble_models()], axis=1)
    correct = (P <= 0.5).sum().to_numpy()
    print(f"\n{N_SEEDS} under-5 models, each on a random {FRAC:.0%} of training events: "
          f"correct of {len(P)} median {np.median(correct):.0f}, "
          f"5th-95th percentile {np.percentile(correct, 5):.0f}-{np.percentile(correct, 95):.0f}")

    ens = P.mean(axis=1)
    pd.DataFrame({"event_id": ens.index, "dataset": "ridgecrest", "true": "EQ",
                  "pred": np.where(ens > 0.5, "UNE", "EQ"), "prob_une": ens.round(3).values}
                 ).to_csv("data/predictions/ridgecrest_lt5_ensemble.csv", index=False)

    per = pd.DataFrame({"share_EQ": (P <= 0.5).mean(axis=1), "ens_P_UNE": ens}).join(info)
    per["group"] = np.where(per["dist_km"] >= FAR_KM, f"far (>={FAR_KM} km)", "close")
    per = per.sort_values(["group", "share_EQ"], ascending=[False, True])
    per.to_csv("data/predictions/ridgecrest_lt5_ensemble_events.csv")
    for g, d in per.groupby("group", sort=False):
        print(f"\n{g}: {len(d)} events, ensemble calls EQ for {int((d['ens_P_UNE'] <= 0.5).sum())}")
        print(d[["mag", "dist_km", "share_EQ", "ens_P_UNE"]].round(2).to_string())

if __name__ == "__main__":
    os.makedirs("data/predictions", exist_ok=True)
    models = {k: joblib.load(v) for k, v in MODELS.items()}
    preds = {k: [] for k in models}
    feats, info = [], []
    for _, r in test_events().iterrows():
        f = block_features(os.path.join(DATA_DIR, r["file"]), r["p_s"])
        if f.empty:
            print(r["file"], "skipped, no block with a detected arrival")
            continue
        eid = r["file"].replace(".h5", "")
        f = f.assign(event_id=eid, label="EQ", dist_deg=r["dist_km"] / 111.19)
        feats.append(f)
        info.append(dict(event_id=eid, mag=r["mag"], dist_km=r["dist_km"]))
        for k, m in models.items():
            p = float(np.median(m.predict_proba(f[FEATURES_11])[:, 1]))
            preds[k].append(dict(event_id=eid, dataset="ridgecrest", true="EQ",
                                 pred="UNE" if p > 0.5 else "EQ", prob_une=round(p, 3)))
        print(f"{eid}: {len(f)} blocks, M{r['mag']} {r['dist_km']:.0f} km, "
              f"P(UNE) all={preds['all'][-1]['prob_une']} lt5={preds['lt5'][-1]['prob_une']}")

    blocks = pd.concat(feats)
    blocks.to_csv("data/features/ridgecrest_blocks.csv", index=False)
    for k, rows in preds.items():
        pd.DataFrame(rows).to_csv(f"data/predictions/ridgecrest_{k}.csv", index=False)
    print("saved predictions and block features")
    run_ensemble(blocks, pd.DataFrame(info).set_index("event_id"))
