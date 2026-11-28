"""Far-event experiment: Ridgecrest DAS events at 90-500 km with M>=3 (the QuakeFlow catalog stops at 252 km), through
the strict rule and the committed block pipeline, scored once with the strain-rate-trained v2 ensemble.
Run stack_checks.py --dataset ridgecrest first, so every downloaded file has a row.
Usage: python scripts/far_experiment.py   -> data/predictions/ridgecrest_far_srv2_ensemble.csv
"""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path[:0] = [".", "scripts"]
import predict_das as PD

MIN_KM, MAX_KM, MIN_MAG = 90, 500, 3.0
FAR_CHECKS = "data/stack_checks_ridgecrest_far.csv"

if __name__ == "__main__":
    c = pd.read_csv("data/stack_checks_ridgecrest.csv")
    far = c[c["dist_km"].between(MIN_KM, MAX_KM) & (c["mag"] >= MIN_MAG)]
    far.to_csv(FAR_CHECKS, index=False)
    print(f"{len(far)} downloaded files at {MIN_KM}-{MAX_KM} km, M>={MIN_MAG}")
    PD.CHECKS = FAR_CHECKS
    models = joblib.load("data/models/rf_lt5_srtrained_ensemble.joblib")
    rows = []
    for _, r in PD.test_events().iterrows():
        eid = r["file"].replace(".h5", "")
        f = PD.block_features(os.path.join(PD.DATA_DIR, r["file"]), r["p_s"])
        if f.empty:
            print(eid, "skipped, no block with a detected arrival")
            continue
        p = float(np.mean([np.median(m.predict_proba(f[PD.FEATURES_11])[:, 1]) for m in models]))
        rows.append(dict(event_id=eid, dataset="ridgecrest_far", true="EQ", pred="UNE" if p > 0.5 else "EQ",
                         prob_une=round(p, 3), mag=r["mag"], dist_km=round(r["dist_km"], 1)))
        print(f"{eid}: M{r['mag']} {r['dist_km']:.0f} km, {len(f)} blocks, P(UNE) {p:.3f}", flush=True)
    pd.DataFrame(rows).to_csv("data/predictions/ridgecrest_far_srv2_ensemble.csv", index=False)
