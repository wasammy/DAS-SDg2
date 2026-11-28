"""Alignment experiment: shift each Ridgecrest channel by its predicted P (iasp91, event location, SCEDC channel
positions in data/ridgecrest_north/meta/das_info.csv) before the 50-channel substack. Test 6 comparison for close
events first; the strain-rate-trained v2 ensemble is scored once only if the pre-declared rule passes.
Usage: python scripts/align_experiment.py   -> data/features/ridgecrest_blocks_aligned.csv, ridgecrest_align_stats.csv
"""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

warnings.filterwarnings("ignore")
sys.path[:0] = [".", "scripts"]
import predict_das as PD
import train_rf as T
import strainrate_experiment as S
import build_features as B
from src.das_io import load_das_event
from src.stacking import substack
from src.preprocess import bandpass, preprocess_trace
from src.features import extract_features
from src.compare import FEATURES_11
from src.training_set import load_gtune
from stack_checks import after_before, fake_splits_before, CABLE_LATLON

INFO = "data/ridgecrest_north/meta/das_info.csv"
SR_MODELS = "data/models/rf_lt5_srtrained_ensemble.joblib"
SPECTRAL = ["spectral_centroid", "bandwidth"]
TIMING = ["energy_centroid_time", "duration_20", "duration_10"]
model = TauPyModel("iasp91")


def ttp(depth, d):
    a = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=d, phase_list=["ttp"])
    return a[0].time if a else np.nan


def channel_shifts(meta, lat, lon):
    depth = max(float(meta["depth_km"]), 0.0)
    d = locations2degrees(lat, lon, meta["latitude"], meta["longitude"])
    grid = np.linspace(d.min(), d.max(), 40)
    tc = np.interp(d, grid, [ttp(depth, g) for g in grid])
    tref = ttp(depth, locations2degrees(*CABLE_LATLON, meta["latitude"], meta["longitude"]))
    return tc - tref


def shift_rows(data, n):
    out = np.zeros_like(data)
    T_ = data.shape[1]
    for c, s in enumerate(n):
        if s >= 0:
            out[c, :T_ - s] = data[c, s:]
        else:
            out[c, -s:] = data[c, :T_ + s]
    return out


def blocks_features(data, fs, p_s):
    blocks, counts = substack(data, fs, group=50, filtered=False)
    filt = np.array([bandpass(b, fs) for b in blocks])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = after_before(filt, fs, p_s)
        fakes = np.array([after_before(filt, fs, s) for s in fake_splits_before(p_s)])
    detected = ratio >= PD.RATIO_FACTOR * fakes.max(axis=0)
    rows = []
    for b, (x, n) in enumerate(zip(blocks, counts)):
        if n == 0 or not detected[b]:
            continue
        try:
            w, _ = preprocess_trace(x, fs, p_index=round(p_s * fs))
        except ValueError:
            continue
        rows.append({"block": b, "ratio": float(ratio[b]), **extract_features(w)})
    return pd.DataFrame(rows), np.nanmedian(ratio)


def med(s):
    return f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})"


if __name__ == "__main__":
    info = pd.read_csv(INFO)
    lat, lon = info["latitude"].to_numpy(), info["longitude"].to_numpy()
    A, stats = [], []
    for _, r in PD.test_events().iterrows():
        eid = r["file"].replace(".h5", "")
        data, meta = load_das_event(os.path.join(PD.DATA_DIR, r["file"]))
        assert data.shape[0] == len(info), (data.shape, len(info))
        fs = meta["fs"]
        sh = channel_shifts(meta, lat, lon)
        n = np.round(sh * fs).astype(int)
        f0, r0 = blocks_features(data, fs, r["p_s"])
        f1, r1 = blocks_features(shift_rows(data, n), fs, r["p_s"])
        within = pd.Series(sh).groupby(np.arange(len(sh)) // 50).agg(lambda x: x.max() - x.min())
        stats.append(dict(event_id=eid, dist_km=r["dist_km"], mag=r["mag"], cable_spread_s=sh.max() - sh.min(),
                          block_spread_s=within.max(), ratio_unaligned=r0, ratio_aligned=r1,
                          blocks_unaligned=len(f0), blocks_aligned=len(f1)))
        A.append(f1.assign(event_id=eid))
        print(eid, f"spread {sh.max() - sh.min():.2f} s, ratio {r0:.2f} -> {r1:.2f}, blocks {len(f0)} -> {len(f1)}", flush=True)
    A = pd.concat(A, ignore_index=True)
    A.to_csv("data/features/ridgecrest_blocks_aligned.csv", index=False)
    st = pd.DataFrame(stats).set_index("event_id")
    st.to_csv("data/features/ridgecrest_align_stats.csv")
    close = st[st["dist_km"] < PD.FAR_KM]
    print(f"\nP spread along the cable: close {med(close['cable_spread_s'])} s, far {med(st[st.dist_km >= PD.FAR_KM]['cable_spread_s'])} s; "
          f"largest spread within one 50-channel block {st['block_spread_s'].max():.3f} s")
    print(f"median ratio at P (geometry check): unaligned {med(st['ratio_unaligned'])} -> aligned {med(st['ratio_aligned'])}; "
          f"higher after alignment in {int((st['ratio_aligned'] > st['ratio_unaligned']).sum())} of {len(st)} events")
    print(f"detected blocks: {int(st['blocks_unaligned'].sum())} -> {int(st['blocks_aligned'].sum())}; events with none after alignment: "
          f"{int((st['blocks_aligned'] == 0).sum())}")

    U = pd.read_csv("data/features/ridgecrest_blocks.csv")
    cid = set(close.index)
    Uc, Ac = U[U["event_id"].isin(cid)], A[A["event_id"].isin(cid)]
    une = S.converted(load_gtune(), "UNE")
    eq = S.converted(B.load_eqs(), "EQ")
    seis = S.table(une, eq)
    held = set(pd.read_csv(T.HELDOUT)["event_id"].astype(str))
    seis = T.in_bin(seis[~seis["event_id"].isin(held)], 0, 5)
    print("\nTest 6, close events (31), per block. Seismometer = strain-rate <5 deg training data.")
    print(f"{'feature':>22} | {'seis <5 all':>18} | {'seis <5 EQ':>18} | {'DAS unaligned':>18} | {'DAS aligned':>18} | toward?")
    toward = {}
    for f in SPECTRAL + TIMING + ["dominant_frequency"]:
        m = seis[f].median()
        toward[f] = abs(Ac[f].median() - m) < abs(Uc[f].median() - m)
        print(f"{f:>22} | {med(seis[f]):>18} | {med(seis[seis.y == 0][f]):>18} | {med(Uc[f]):>18} | {med(Ac[f]):>18} | {toward[f]}")
    ok = all(toward[f] for f in SPECTRAL) and sum(toward[f] for f in TIMING) >= 2
    print(f"\nrule (both spectral toward, >=2 of 3 timing toward): {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("not scored")
        sys.exit()
    models = joblib.load(SR_MODELS)
    P = pd.concat([PD.event_probs(m, A) for m in models], axis=1)
    ens = P.mean(axis=1)
    far = set(st.index[st["dist_km"] >= PD.FAR_KM])
    good = ens <= 0.5
    print(f"aligned, strain-rate-trained v2: {int(good.sum())}/{len(ens)} EQ (far {int(good[ens.index.isin(far)].sum())}/10, "
          f"close {int(good[~ens.index.isin(far)].sum())}/{len(ens) - len(far & set(ens.index))})")
    pd.DataFrame({"event_id": ens.index, "dataset": "ridgecrest", "true": "EQ", "pred": np.where(good, "EQ", "UNE"),
                  "prob_une": ens.round(3).values}).to_csv("data/predictions/ridgecrest_aligned_srv2_ensemble.csv", index=False)
