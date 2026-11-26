"""Classify Ridgecrest DAS events with the seismograph-trained RF.

One prediction per event: substack 50 channels, preprocess each block like the
seismograph data, predict each block, event probability==median over blocks.
Writes predictions in evaluate.py's format, plus block features for Test 6.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.das_io import load_das_event
from src.stacking import substack
from src.preprocess import preprocess_trace
from src.features import extract_features
from src.compare import FEATURES_11
sys.path.insert(0, "scripts")
from stack_checks import after_before, fake_splits_before
from src.preprocess import bandpass, preprocess_trace 

DATA_DIR = "data/ridgecrest_north/data"
CHECKS = "data/stack_checks_ridgecrest.csv"
MODELS = {"all": "data/models/rf_all.joblib", "lt5": "data/models/rf_lt5.joblib"}

RATIO_FACTOR = 1.2 # ratio at P must be >= 1.2x the best fake window
MAX_ONSET_OFFSET_S = 2.0 # observed onset within 2s of predicted P

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
    blocks, counts = substack(data, fs, group=50, filtered=False)  # filtered once, below

    # detection test on a 0.5-10 Hz copy
    filt = np.array([bandpass(b, fs) for b in blocks])
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

if __name__ == "__main__":
    os.makedirs("data/predictions", exist_ok=True)
    models = {k: joblib.load(v) for k, v in MODELS.items()}
    preds = {k: [] for k in models}
    feats = []
    for _, r in test_events().iterrows():
        f = block_features(os.path.join(DATA_DIR, r["file"]), r["p_s"])
        if f.empty:
            print(r["file"], "skipped, no block with a detected arrival")
            continue
        eid = r["file"].replace(".h5", "")
        f = f.assign(event_id=eid, label="EQ", dist_deg=r["dist_km"] / 111.19)
        feats.append(f)
        for k, m in models.items():
            p = float(np.median(m.predict_proba(f[FEATURES_11])[:, 1]))
            preds[k].append(dict(event_id=eid, dataset="ridgecrest", true="EQ",
                                 pred="UNE" if p > 0.5 else "EQ", prob_une=round(p, 3)))
        print(f"{eid}: {len(f)} blocks, M{r['mag']} {r['dist_km']:.0f} km, "
              f"P(UNE) all={preds['all'][-1]['prob_une']} lt5={preds['lt5'][-1]['prob_une']}")

    pd.concat(feats).to_csv("data/features/ridgecrest_blocks.csv", index=False)
    for k, rows in preds.items():
        pd.DataFrame(rows).to_csv(f"data/predictions/ridgecrest_{k}.csv", index=False)
    print("saved predictions and block features")