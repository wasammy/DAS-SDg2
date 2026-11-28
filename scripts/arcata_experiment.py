"""Arcata experiment: Arcata DAS events through the strict rule and the committed block pipeline (predict_das),
scored once with the strain-rate-trained v2 ensemble. Run stack_checks.py --dataset arcata first.
Usage: python scripts/arcata_experiment.py   -> data/features/arcata_blocks.csv, data/predictions/arcata_srv2_ensemble.csv
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

PD.CHECKS = "data/stack_checks_arcata.csv"
PD.DATA_DIR = "data/arcata/data"

if __name__ == "__main__":
    models = joblib.load("data/models/rf_lt5_srtrained_ensemble.joblib")
    rows, feats = [], []
    for _, r in PD.test_events().iterrows():
        f = PD.block_features(os.path.join(PD.DATA_DIR, r["file"]), r["p_s"])
        eid = r["file"].replace(".h5", "")
        if f.empty:
            print(eid, "skipped, no block with a detected arrival")
            continue
        f = f.assign(event_id=eid)
        feats.append(f)
        p = float(np.mean([np.median(m.predict_proba(f[PD.FEATURES_11])[:, 1]) for m in models]))
        rows.append(dict(event_id=eid, dataset="arcata", true="EQ", pred="UNE" if p > 0.5 else "EQ", prob_une=round(p, 3)))
        print(f"{eid}: M{r['mag']} {r['dist_km']:.0f} km, {len(f)} blocks, P(UNE) {p:.3f}", flush=True)
    pd.concat(feats).to_csv("data/features/arcata_blocks.csv", index=False)
    pd.DataFrame(rows).to_csv("data/predictions/arcata_srv2_ensemble.csv", index=False)
