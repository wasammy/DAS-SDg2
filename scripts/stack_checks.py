"""
Is there an earthquake where we expect P? Per-file checks on DAS event files.

P time per file:
  monterey   : catalog P (p_in_window_s) from Osman's first-pass match CSV.
               Files without a match are "no catalog P". Monterey files are NOT
               P-aligned at 10 s (Nov 15 finding).
  ridgecrest : origin time from the file's metadata (event_time_index) + iasp91
               P travel time from the event to the cable (CABLE_LATLON, approximate).

For each file:
  - ratio at P: 0.5-10 Hz substacks (group=50), median over substacks of
    std(P..P+6 s) / std(P-8..P-2 s)
  - fake ratios: the same ratio at split times that end before the event
    (same file), and, if some files have no P, at the same time in those files
  - observed onset: first time the 2-40 Hz envelope stack rises clearly above
    its pre-P level (median + 5 robust std, held for 0.5 s), searched P-5..P+20 s
  - window fit: needs >= 10 s before P and >= 50 s after P (1200 samples at 20 Hz),
    with no more than 1% of that window zero-filled (data gaps)

Usage, from the top of the repo:
    python scripts/stack_checks.py --dataset monterey
    python scripts/stack_checks.py --dataset ridgecrest
Writes data/stack_checks_<dataset>.csv and .png (data/ is not in Git).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.das_io import load_das_event  # noqa: E402
from src.stacking import substack      # noqa: E402

DATASETS = {
    "monterey": ROOT / "data" / "monterey_bay" / "data",
    "ridgecrest": ROOT / "data" / "ridgecrest_north" / "data",
}
CATALOG_CSV = ROOT / "data" / "catalog_match.csv"   # Osman's first pass (v1)
# Ridgecrest DAS cable, approximate (city of Ridgecrest). The cable is ~9 km long,
# so this is good to about +-1 s in P. Checked against the observed onset.
CABLE_LATLON = (35.65, -117.66)

PRE_NEEDED_S, POST_NEEDED_S = 10.0, 50.0
MAX_GAP_FRAC = 0.01   # window fails if more than 1% of it is zero-filled
AFTER_S, GAP_S, BEFORE_S = 6.0, 2.0, 6.0


# ---------- ratios and envelope ----------

def after_before(x, fs, split_s, win_s=AFTER_S, gap_s=GAP_S):
    """
    std of x in [split, split + win] over std in [split - gap - win, split - gap].
    x: 1D trace or 2D array (time on the last axis). Returns float or array.
    """
    x = np.asarray(x, dtype=float)
    i = lambda s: int(round(s * fs))
    after = x[..., i(split_s):i(split_s + win_s)]
    before = x[..., i(split_s - gap_s - win_s):i(split_s - gap_s)]
    return after.std(axis=-1) / before.std(axis=-1)


def envelope_stack(data, fs, step=3, fmin=2.0, fmax=40.0, smooth_s=0.5):
    """
    Energy-envelope stack (notebook 02 cell 15): bandpass, square, divide by the
    channel's median energy, median across channels, moving average.
    fmax is capped below Nyquist. Dead channels are skipped. The first and last
    smooth_s are dropped. returns: (t, stack)
    """
    hi = min(fmax, 0.45 * fs)
    sos = butter(4, [fmin, hi], btype="bandpass", fs=fs, output="sos")
    env = []
    for c in range(0, data.shape[0], step):
        y = sosfiltfilt(sos, data[c] - data[c].mean())
        e = y ** 2
        m = np.median(e)
        if np.isfinite(m) and m > 0:
            env.append(e / m)
    med = np.median(np.array(env), axis=0)
    w = int(smooth_s * fs)
    stack = np.convolve(med, np.ones(w) / w, mode="same")
    t = np.arange(len(stack)) / fs
    return t[w:-w], stack[w:-w]


def observed_onset(t, env, p_s, search=(-5.0, 20.0), k=5.0, hold_s=0.5):
    """
    First time in [P + search[0], P + search[1]] where the envelope stays above
    median + k * robust std of the pre-P level for hold_s. NaN if none.
    The pre-P level is taken from [P - 10, P - 2] s (or what exists of it).
    """
    base = env[(t >= max(t[0], p_s - 10.0)) & (t < p_s - 2.0)]
    if base.size < 20:
        return np.nan
    med = np.median(base)
    thr = med + k * 1.4826 * np.median(np.abs(base - med))
    dt = t[1] - t[0]
    hold = max(1, int(hold_s / dt))
    idx = np.where((t >= p_s + search[0]) & (t <= p_s + search[1]))[0]
    above = env[idx] > thr
    run = 0
    for j, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= hold:
            return float(t[idx[j - hold + 1]])
    return np.nan


def zero_filled(data, chan_frac=0.9):
    """
    Boolean per time sample: True where nearly all channels are exactly 0.
    That marks a data gap filled with zeros (e.g. ci39494008 has no data after
    ~70 s), not a quiet period. Real DAS noise is never exactly 0 on every channel.
    """
    return (np.asarray(data) == 0).mean(axis=0) > chan_frac


def fake_splits_before(p_s, step_s=1.0):
    """Split times whose whole after/before measurement ends >= 1 s before P."""
    lo = GAP_S + BEFORE_S
    hi = p_s - 1.0 - AFTER_S
    return list(np.arange(lo, hi + 1e-9, step_s)) if hi >= lo else []


# ---------- P per file ----------

def p_from_catalog(csv_path=CATALOG_CSV):
    """file name -> dict(p_s, mag, dist_km) for matched rows of Osman's v1 CSV."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df[df["matched"]].iterrows():
        out[r["file"]] = dict(p_s=float(r["p_in_window_s"]), mag=float(r["mag"]),
                              dist_km=float(r["dist_km"]))
    return out


def p_from_metadata(meta, cable=CABLE_LATLON):
    """
    P time (s from window start) = origin index / fs + iasp91 P travel time.
    Returns dict(p_s, mag, dist_km, origin_s) or None if the metadata is missing.
    Note event_time_index = -1 means "missing" (and -1 is truthy).
    """
    from obspy.geodetics import locations2degrees
    from obspy.taup import TauPyModel

    idx = meta.get("event_time_index")
    if idx is None or idx < 0 or meta.get("latitude") is None:
        return None
    origin_s = idx / meta["fs"]
    dist_deg = locations2degrees(cable[0], cable[1], meta["latitude"], meta["longitude"])
    arr = TauPyModel("iasp91").get_travel_times(
        source_depth_in_km=max(float(meta.get("depth_km", 10.0)), 0.0),
        distance_in_degree=dist_deg, phase_list=["ttp"])
    if not arr:
        return None
    return dict(p_s=origin_s + arr[0].time, mag=meta.get("magnitude"),
                dist_km=dist_deg * 111.19, origin_s=origin_s)


# ---------- run ----------

def main():
    ap = argparse.ArgumentParser(description="per-file P checks on DAS event files")
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--group", type=int, default=50, help="channels per substack")
    args = ap.parse_args()

    files = sorted(DATASETS[args.dataset].glob("*.h5"))
    if not files:
        sys.exit(f"no .h5 files in {DATASETS[args.dataset]}")
    catalog = p_from_catalog() if args.dataset == "monterey" else None

    # pass 1: load each file once, keep only the small things we need
    rec = []
    for f in files:
        data, meta = load_das_event(f)
        fs = meta["fs"]
        info = catalog.get(f.name) if catalog is not None else p_from_metadata(meta)
        st, _ = substack(data, fs, group=args.group)
        t, env = envelope_stack(data, fs)
        rec.append(dict(file=f.name, fs=fs, dur=meta["n_time"] / fs, info=info,
                        st=st, t=t, env=env, zero=zero_filled(data)))
        print(f"loaded {f.name}", flush=True)

    no_p = [r for r in rec if r["info"] is None]
    rows = []
    for r in rec:
        fs, info = r["fs"], r["info"]
        row = dict(file=r["file"], duration_s=round(r["dur"], 1),
                   env_peak_s=round(float(r["t"][np.argmax(r["env"])]), 1),
                   env_max_over_median=round(float(r["env"].max() / np.median(r["env"])), 2))
        if info is None:
            row["note"] = "no catalog P"
            rows.append(row)
            continue
        p = info["p_s"]
        row.update(mag=info["mag"], dist_km=round(info["dist_km"], 0), p_s=round(p, 1))
        if "origin_s" in info:
            row["origin_s"] = round(info["origin_s"], 1)

        row["pre_P_s"] = round(p, 1)
        row["post_P_s"] = round(r["dur"] - p, 1)
        long_enough = p >= PRE_NEEDED_S and r["dur"] - p >= POST_NEEDED_S
        i0 = max(0, int(round((p - PRE_NEEDED_S) * fs)))
        i1 = int(round((p + POST_NEEDED_S) * fs))
        gap_frac = float(r["zero"][i0:i1].mean()) if i1 > i0 else 1.0
        row["zero_filled_in_window_pct"] = round(100 * gap_frac, 1)
        row["fits_window"] = bool(long_enough and gap_frac <= MAX_GAP_FRAC)

        on = observed_onset(r["t"], r["env"], p)
        row["onset_s"] = round(on, 1) if np.isfinite(on) else np.nan
        row["onset_minus_P_s"] = round(on - p, 1) if np.isfinite(on) else np.nan

        if p - GAP_S - BEFORE_S >= 0 and p + AFTER_S <= r["dur"]:
            row["ratio_at_P"] = round(float(np.median(after_before(r["st"], fs, p))), 2)
        splits = fake_splits_before(p)
        fakes = [float(np.median(after_before(r["st"], fs, s))) for s in splits]
        row["n_fake_same_file"] = len(fakes)
        if fakes:
            row["fake_same_file_max"] = round(max(fakes), 2)
        other = [float(np.median(after_before(q["st"], q["fs"], p))) for q in no_p
                 if p - GAP_S - BEFORE_S >= 0 and p + AFTER_S <= q["dur"]]
        if other:
            row["fake_other_files_max"] = round(max(other), 2)
        ref = max([v for v in (row.get("fake_same_file_max"), row.get("fake_other_files_max"))
                   if v is not None], default=np.nan)
        row["above_all_fakes"] = bool(np.isfinite(ref) and row.get("ratio_at_P", 0) > ref)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = ROOT / "data" / f"stack_checks_{args.dataset}.csv"
    df.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string(index=False))
    print(f"\nsaved {out_csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rec)
    ncol = 3 if n > 6 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 1.9 * nrow + 0.8),
                            sharex=True, squeeze=False)
    axs = axs.T.ravel()
    for ax, r, row in zip(axs, rec, rows):
        ax.semilogy(r["t"], r["env"] / np.median(r["env"]), lw=0.8)
        title = r["file"].replace(".h5", "")
        if r["info"] is not None:
            ax.axvline(r["info"]["p_s"], color="k", lw=1.5)
            if np.isfinite(row.get("onset_s", np.nan)):
                ax.axvline(row["onset_s"], color="crimson", ls="--", lw=1)
            title += f"  M{r['info']['mag']} {r['info']['dist_km']:.0f} km"
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle(f"{args.dataset}: envelope stack 2-40 Hz (log). "
                 "black = expected P, red dashed = observed onset", fontsize=10)
    fig.tight_layout()
    out_png = ROOT / "data" / f"stack_checks_{args.dataset}.png"
    fig.savefig(out_png, dpi=90)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
