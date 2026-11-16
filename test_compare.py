"""Smoke tests for src/compare.py (Test 6 feature comparison), using synthetic data."""
import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.features import feature_table, RF_V2_FEATURES
from src.compare import (FEATURES_11, normalize_label, load_features, handout_errors,
                         distance_bin, bin_medians, range_table, plot_overlays)

rng = np.random.default_rng(1)
ok = lambda name: print(f"  PASS  {name}")


def windows(n, kind):
    """Fake 1200-sample, peak-normalised windows with P at sample 200."""
    out = []
    t = np.arange(1200) / 20.0
    for _ in range(n):
        x = rng.normal(0, 0.05, 1200)
        if kind == "signal":
            k = np.arange(1000)
            x[200:] += np.exp(-k / 80) * np.sin(2 * np.pi * 2 * k / 20)
        x /= np.abs(x).max()
        out.append(x)
    return out


print("\n[1] 11 features")
assert len(FEATURES_11) == 11 and "peak" not in FEATURES_11 and "crest_factor" not in FEATURES_11
assert all(f in RF_V2_FEATURES for f in FEATURES_11)
ok("13 handout features minus peak and crest_factor")

print("\n[2] labels")
assert [normalize_label(x) for x in ["earthquake", "EQ", " eq ", "explosion", "UNE"]] == ["EQ", "EQ", "EQ", "UNE", "UNE"]
try:
    normalize_label("noise"); raise SystemExit("FAIL: unknown label should raise")
except ValueError:
    ok("earthquake/EQ and explosion/UNE map to EQ/UNE; anything else raises")

print("\n[3] handout errors")
X, names = feature_table(windows(20, "signal"))
df = pd.DataFrame(X, columns=names)
e = handout_errors(df)
assert e["peak_is_constant"] and e["peak_values"] == [1.0]
ok("peak is 1.0 on every peak-normalised window")
assert e["crest_is_duplicate"] and e["crest_minus_ratio"] == 0.0
ok("crest_factor equals peak_rms_ratio")

print("\n[4] distance bins")
assert [distance_bin(d) for d in [0.1, 4.99, 5.0, 9.9, 15, 20, 25]] == ["<5", "<5", "5-10", "5-10", "10-20", None, None]
ok("0-5, 5-10, 10-20 degrees; 20 and beyond are outside")

print("\n[5] range table")
seis = pd.DataFrame(feature_table(windows(200, "signal"))[0], columns=names)
seis["label"] = "EQ"
same = pd.DataFrame(feature_table(windows(30, "signal"))[0], columns=names)
noise = pd.DataFrame(feature_table(windows(30, "noise"))[0], columns=names)
t_same = range_table(seis, same)
t_noise = range_table(seis, noise)
assert (t_same["verdict"] == "inside").sum() >= 9, t_same["verdict"].to_dict()
ok(f"signal vs signal: {(t_same['verdict'] == 'inside').sum()}/11 features inside")
assert t_noise.loc["duration_10", "verdict"] == "OUTSIDE"
assert t_noise.loc["energy_centroid_time", "verdict"] == "OUTSIDE"
ok("noise vs signal: duration_10 and energy_centroid_time flagged OUTSIDE")

print("\n[6] per-bin medians")
seis["dist_deg"] = rng.uniform(0, 20, len(seis))
bm = bin_medians(seis)
assert list(bm.columns[:2]) == ["ours <5", "handout <5"] and bm.loc["duration_10", "handout <5"] == 28.10
ok("ours vs handout medians laid out per bin")

print("\n[7] load + plot")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "a_features.csv")
    same.assign(label="earthquake").to_csv(p, index=False)
    back = load_features(os.path.join(d, "*_features.csv"), source="DAS")
    assert len(back) == 30 and set(back["label"]) == {"EQ"} and set(back["source"]) == {"DAS"}
    assert load_features(os.path.join(d, "nothing*.csv")).empty
    ok("loads a glob of CSVs; a missing input gives an empty table, not an error")
    out = os.path.join(d, "fig.png")
    plot_overlays({"Seismometer EQ": seis, "DAS": same, "empty": pd.DataFrame()}, path=out)
    assert os.path.getsize(out) > 10_000
    ok("overlay figure saved")

print("\nAll comparison checks passed.")
