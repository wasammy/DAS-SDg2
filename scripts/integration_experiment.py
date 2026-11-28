"""Integration experiment: integrate Ridgecrest DAS blocks (strain rate -> velocity-like) before preprocessing,
on the same detected blocks, and score the committed v2 and v3 50-model ensembles on both versions.
Usage: python scripts/integration_experiment.py   -> data/features/ridgecrest_blocks_integrated.csv,
       data/predictions/ridgecrest_{v2,v3b}_integrated_ensemble.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path[:0] = [".", "scripts"]
import predict_das as PD
import train_rf as T
import v3_blast_experiment as V
from src.das_io import load_das_event
from src.das_convert import integrate_strain_rate
from src.stacking import substack
from src.preprocess import bandpass, preprocess_trace
from src.features import extract_features
from src.compare import FEATURES_11
from stack_checks import after_before, fake_splits_before

SPECTRAL = ["dominant_frequency", "spectral_centroid", "bandwidth"]
SHOW = SPECTRAL + ["energy_centroid_time", "duration_20"]


def both_features(path, p_s):
    data, meta = load_das_event(path)
    fs = meta["fs"]
    blocks, counts = substack(data, fs, group=50, filtered=False)
    filt = np.array([bandpass(b, fs) for b in blocks])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = after_before(filt, fs, p_s)
        fakes = np.array([after_before(filt, fs, s) for s in fake_splits_before(p_s)])
    detected = ratio >= PD.RATIO_FACTOR * fakes.max(axis=0)  # detection stays on strain rate
    raw, integ = [], []
    for b, (x, n) in enumerate(zip(blocks, counts)):
        if n == 0 or not detected[b]:
            continue
        try:
            w0, _ = preprocess_trace(x, fs, p_index=round(p_s * fs))
            w1, _ = preprocess_trace(integrate_strain_rate(x, fs), fs, p_index=round(p_s * fs))
        except ValueError:
            continue
        raw.append({"block": b, **extract_features(w0)})
        integ.append({"block": b, **extract_features(w1)})
    return pd.DataFrame(raw), pd.DataFrame(integ)


def v3_models():
    d = V.v3_table()
    d = d[d["snr_pass"]].reset_index(drop=True)
    v2 = T.load_table()
    held = set(pd.read_csv(T.HELDOUT)["event_id"].astype(str))
    v2 = T.in_bin(v2[~v2["event_id"].isin(held)], 0, 5)
    v2["source"] = np.where(v2["y"] == 1, "GTUNE", "EQ-v2")
    both = pd.concat([v2, d[v2.columns.intersection(d.columns)]], ignore_index=True)
    events = both["event_id"].unique()
    models = []
    for s in range(V.N_SEEDS):
        pick = np.random.default_rng(s).choice(events, size=int(round(V.FRAC * len(events))), replace=False)
        sub = both[both["event_id"].isin(pick)]
        models.append(V.rf(s).fit(sub[FEATURES_11], sub["y"]))
    return models


def score(name, models, blocks, far, out=None):
    P = pd.concat([PD.event_probs(m, blocks) for m in models], axis=1)
    ens = P.mean(axis=1)
    ok = ens <= 0.5
    c = (P <= 0.5).sum().to_numpy()
    print(f"  {name:>34}: {int(ok.sum())}/41 EQ (far {int(ok[ens.index.isin(far)].sum())}/10, "
          f"close {int(ok[~ens.index.isin(far)].sum())}/31); 50 models median {np.median(c):.0f}, "
          f"5th-95th {np.percentile(c, 5):.0f}-{np.percentile(c, 95):.0f}")
    if out:
        pd.DataFrame({"event_id": ens.index, "dataset": "ridgecrest", "true": "EQ",
                      "pred": np.where(ok, "EQ", "UNE"), "prob_une": ens.round(3).values}).to_csv(out, index=False)
    return ens


def spread(s):
    return f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})"


if __name__ == "__main__":
    R, I, far = [], [], set()
    for _, r in PD.test_events().iterrows():
        eid = r["file"].replace(".h5", "")
        a, b = both_features(os.path.join(PD.DATA_DIR, r["file"]), r["p_s"])
        R.append(a.assign(event_id=eid))
        I.append(b.assign(event_id=eid))
        if r["dist_km"] >= PD.FAR_KM:
            far.add(eid)
    R, I = pd.concat(R, ignore_index=True), pd.concat(I, ignore_index=True)
    I.to_csv("data/features/ridgecrest_blocks_integrated.csv", index=False)
    old = pd.read_csv("data/features/ridgecrest_blocks.csv")
    same = len(old) == len(R) and np.allclose(old[FEATURES_11].to_numpy(), R[FEATURES_11].to_numpy(), equal_nan=True)
    print(f"{len(R)} blocks; strain-rate features identical to ridgecrest_blocks.csv: {same}")

    spec_only = R.copy()
    spec_only[SPECTRAL] = I[SPECTRAL].values
    for tag, models in (("v2", PD.ensemble_models()), ("v3b", v3_models())):
        print(f"\n{tag} ensemble (committed models):")
        score("strain rate", models, R, far)
        score("integrated", models, I, far, f"data/predictions/ridgecrest_{tag}_integrated_ensemble.csv")
        score("only spectral features integrated", models, spec_only, far)

    s5 = T.in_bin(T.load_table(), 0, 5)
    print("\nmedian (IQR): seismometer <5 EQ | <5 UNE | DAS strain rate | DAS integrated")
    for f in SHOW:
        print(f"  {f:>20}: {spread(s5[s5.y == 0][f])} | {spread(s5[s5.y == 1][f])} | {spread(R[f])} | {spread(I[f])}")
    eqm, r0, r1 = s5[s5.y == 0]["spectral_centroid"].median(), R["spectral_centroid"].median(), I["spectral_centroid"].median()
    lo, hi = s5["spectral_centroid"].quantile([0.05, 0.95])
    print(f"spectral_centroid gap to seismometer EQ closed: {(r0 - r1) / (r0 - eqm):.0%}; blocks inside the seismometer "
          f"5-95% range: {R['spectral_centroid'].between(lo, hi).mean():.0%} -> {I['spectral_centroid'].between(lo, hi).mean():.0%}")
