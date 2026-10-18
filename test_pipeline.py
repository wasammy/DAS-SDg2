"""Smoke tests for the preprocessing + feature pipeline, using synthetic data."""
import numpy as np
import h5py
import tempfile, os

from src.preprocess import bandpass, resample, pick_p, cut_window, normalize, preprocess_trace
from src.features import extract_features, feature_table, RF_V2_FEATURES
from src.das_io import load_das_event, select_channels, channel_snr

rng = np.random.default_rng(0)
ok = lambda name: print(f"  PASS  {name}")

print("\n[1] bandpass")
fs = 100.0
t = np.arange(0, 10, 1 / fs)
sig = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 30 * t) + 500.0
y = bandpass(sig, fs)
assert abs(y.mean()) < 0.1, y.mean()
assert 0.9 < np.abs(y[100:-100]).max() < 1.3
ok("removes DC offset and the 30 Hz component, keeps 1 Hz")

print("\n[2] resample")
x2, fs2 = resample(np.sin(2 * np.pi * 1 * t), 100.0)
assert fs2 == 20.0 and len(x2) == 200, (fs2, len(x2))
ok("100 Hz -> 20 Hz gives the right length and rate")
x3, fs3 = resample(np.zeros(100), 19.99993515)
assert fs3 == 20.0 and len(x3) == 100
ok("19.99993515 Hz is treated as already 20 Hz (no resample)")
x4, fs4 = resample(np.sin(2 * np.pi * 1 * np.arange(0, 10, 1/39.999999)), 39.999999)
assert fs4 == 20.0 and abs(len(x4) - 200) <= 1, len(x4)
ok("39.999999 Hz handled without integer error")
try:
    resample(np.zeros(100), 15.0); raise SystemExit("FAIL: 15 Hz should raise")
except ValueError:
    ok("15 Hz raises ValueError instead of fabricating data")

print("\n[3] pick_p")
fs = 20.0
n = 2000
trace = rng.normal(0, 1, n) * 0.05
p_true = 800
decay = np.exp(-np.arange(n - p_true) / 60.0)
trace[p_true:] += 3.0 * decay * np.sin(2 * np.pi * 3 * np.arange(n - p_true) / fs)
pick = pick_p(trace, fs)
assert pick is not None, "no pick"
assert abs(pick - p_true) < 40, f"pick {pick} vs true {p_true}"
ok(f"finds a synthetic arrival at {pick} (true {p_true})")
assert pick_p(rng.normal(0, 1, 2000) * 0.05, fs, threshold=8.0) is None
ok("returns None on pure noise with a high threshold")

print("\n[4] cut_window")
w = cut_window(trace, p_true)
assert len(w) == 1200
assert np.allclose(w[200], trace[p_true])
ok("1200 samples with P landing exactly on sample 200")
try:
    cut_window(trace, 50); raise SystemExit("FAIL: should not fit")
except ValueError:
    ok("raises when the window runs off the start of the trace")

print("\n[5] preprocess_trace end to end")
fs_raw = 100.0
n = 12000
raw = rng.normal(0, 1, n) * 0.05 + 1200.0 # DC offset like real counts
p_raw = 4000
d = np.exp(-np.arange(n - p_raw) / 300.0)
raw[p_raw:] += 3.0 * d * np.sin(2 * np.pi * 3 * np.arange(n - p_raw) / fs_raw)
win, info = preprocess_trace(raw, fs_raw)
assert len(win) == 1200
assert np.isclose(np.abs(win).max(), 1.0)
assert info["p_source"] == "sta_lta"
ok(f"100 Hz raw trace -> 1200 samples, peak 1.0, pick at {info['p_index']} (expected ~800)")
win2, info2 = preprocess_trace(raw, fs_raw, p_index=p_raw)
assert info2["p_index"] == 800, info2["p_index"]
ok("a known P index in original samples is rescaled to the 20 Hz grid")

print("\n[6] features")
f = extract_features(win)
assert len(f) == 13, len(f)
for k in RF_V2_FEATURES:
    assert k in f, k
    assert np.isfinite(f[k]), (k, f[k])
ok("all 13 features computed and finite")
assert np.isclose(f["peak"], 1.0)
assert f["duration_10"] >= f["duration_20"]
ok("peak == 1 after normalising; duration_10 >= duration_20")
assert 0 <= f["dominant_frequency"] <= 10
ok(f"dominant frequency {f['dominant_frequency']:.2f} Hz is inside the passband")

# an impulsive trace should look more impulsive than a drawn-out one
imp = np.zeros(1200); imp[200:210] = 1.0; imp += rng.normal(0, 0.01, 1200)
spread = np.zeros(1200); spread[200:900] = 1.0; spread += rng.normal(0, 0.01, 1200)
fi, fsp = extract_features(normalize(imp)), extract_features(normalize(spread))
assert fi["crest_factor"] > fsp["crest_factor"]
assert fi["duration_20"] < fsp["duration_20"]
assert fi["energy_centroid_time"] < fsp["energy_centroid_time"]
ok("impulsive vs drawn-out separates on crest factor, duration and energy centroid")

X, names = feature_table([win, win2, normalize(imp)])
assert X.shape == (3, 13), X.shape
ok(f"feature_table -> {X.shape} matrix with {len(names)} named columns")

print("\n[7] das_io on a synthetic DAS file")
# 120 s at 100 Hz, P around 30 s -- the real quakeflow_das window geometry
nch, nt = 60, 12000
das = rng.normal(0, 1, (nch, nt)) * 0.05
for c in range(10, 40): # only these see the quake
    pp = 3000 + (c - 10) * 3 # moveout across channels
    dd = np.exp(-np.arange(nt - pp) / 400.0)
    das[c, pp:] += 2.0 * dd * np.sin(2 * np.pi * 4 * np.arange(nt - pp) / 100.0)

tmp = os.path.join(tempfile.mkdtemp(), "synthetic_event.h5")
with h5py.File(tmp, "w") as fh:
    d = fh.create_dataset("data", data=das.astype("float32"))
    d.attrs["event_id"] = "test001"
    d.attrs["event_time"] = "2024-01-01T00:00:00 +00:00"
    d.attrs["event_time_index"] = 2800
    d.attrs["latitude"] = 36.7
    d.attrs["longitude"] = -122.1
    d.attrs["depth_km"] = 8.0
    d.attrs["magnitude"] = 3.9
    d.attrs["dt_s"] = 0.01
    d.attrs["dx_m"] = 8.0
    d.attrs["unit"] = "microstrain/s"

data, meta = load_das_event(tmp)
assert data.shape == (nch, nt) and meta["fs"] == 100.0
assert meta["event_id"] == "test001" and meta["magnitude"] == 3.9
ok(f"loaded {meta['n_channels']} channels x {meta['n_time']} samples at {meta['fs']} Hz")

keep, snrs = select_channels(data, split_index=3000, n=20, min_snr=1.5)
assert len(keep) > 0
assert all(10 <= c < 40 for c in keep), f"picked quiet channels: {sorted(keep)}"
ok(f"selected {len(keep)} channels, all from the signal-bearing range, best SNR {snrs[0]:.1f}")

print("\n[8] DAS channels through the full pipeline")
wins, infos = [], []
for c in keep[:10]:
    try:
        w, i = preprocess_trace(data[c], meta["fs"])
        wins.append(w); infos.append(i)
    except ValueError as e:
        print("   skipped channel", c, "-", e)
assert len(wins) >= 8, len(wins)
X, names = feature_table(wins)
assert X.shape[1] == 13 and np.isfinite(X).all()
ok(f"{len(wins)} DAS channels -> feature matrix {X.shape}, all finite")
picks = [i["p_index"] for i in infos]
ok(f"picks ranged {min(picks)}-{max(picks)} samples (moveout across the cable)")

print("\nAll checks passed.\n")
