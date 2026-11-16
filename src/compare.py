"""
Test 6: compare RF-v2 feature values between seismometer data (GTUNE + CREW)
and DAS data, to explain why the classifier transfers to DAS or fails to.

Idea: the model only ever sees the 11 feature values. If a feature on DAS sits
inside the range it had on seismometers, the model has seen values like it
before. If it sits outside, the model is guessing there, and that feature is
the first place to look when a DAS event is misclassified.

Inputs are feature tables, one row per waveform window, with one column per
feature plus:
    label     earthquake / explosion (any of EQ, UNE, earthquake, explosion)
    dist_deg  epicentral distance in degrees (seismometer rows; optional)
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.features import RF_V2_FEATURES

# peak is always 1.00 after peak normalisation, and crest_factor is the same
# formula as peak_rms_ratio (handout Appendix A vs B). Oct 20 decision: drop both.
DROPPED = ("peak", "crest_factor")
FEATURES_11 = [f for f in RF_V2_FEATURES if f not in DROPPED]

# Handout, "Distance-Regime Discovery" table: seismometer feature medians per
# distance bin (EQ and UNE together; the handout does not split them).
# crest_factor = peak/rms, so its row is the same quantity as peak_rms_ratio.
DIST_BINS = [("<5", 0.0, 5.0), ("5-10", 5.0, 10.0), ("10-20", 10.0, 20.0)]
HANDOUT_MEDIANS = {
    "dominant_frequency": {"<5": 1.68, "5-10": 0.92, "10-20": 0.78},
    "spectral_centroid": {"<5": 3.39, "5-10": 2.65, "10-20": 2.34},
    "bandwidth": {"<5": 2.10, "5-10": 1.91, "10-20": 1.75},
    "peak_rms_ratio": {"<5": 5.10, "5-10": 4.03, "10-20": 3.90},
    "duration_20": {"<5": 13.95, "5-10": 22.10, "10-20": 23.15},
    "duration_10": {"<5": 28.10, "5-10": 36.10, "10-20": 36.95},
}

_LABELS = {"eq": "EQ", "earthquake": "EQ", "une": "UNE", "explosion": "UNE"}


def normalize_label(x):
    """'earthquake' / 'EQ' -> 'EQ', 'explosion' / 'UNE' -> 'UNE'."""
    key = str(x).strip().lower()
    if key not in _LABELS:
        raise ValueError(f"unknown label {x!r}; expected EQ/UNE/earthquake/explosion")
    return _LABELS[key]


def load_features(pattern, label=None, source=None):
    """
    Read one or more feature CSVs (a path or a glob) into one DataFrame.

    label  : force this label on every row (e.g. "EQ" for the DAS earthquakes)
    source : value for a 'source' column, e.g. "seismometer" or "DAS stacked"
    Returns an empty DataFrame when nothing matches, so a notebook can run
    before every input exists.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if label is not None:
        df["label"] = label
    if "label" in df:
        df["label"] = df["label"].map(normalize_label)
    if source is not None:
        df["source"] = source
    missing = [f for f in FEATURES_11 if f not in df]
    if missing:
        raise ValueError(f"{pattern}: missing feature columns {missing}")
    return df


def handout_errors(df):
    """
    Check the two handout errors on real feature rows.

    Returns a dict:
        peak_values         distinct peak values (all 1.0 if normalised)
        peak_is_constant    True when peak never changes
        crest_minus_ratio   largest |crest_factor - peak_rms_ratio|
        crest_is_duplicate  True when the two columns are the same
    Columns that are absent come back as None.
    """
    out = {"peak_values": None, "peak_is_constant": None,
           "crest_minus_ratio": None, "crest_is_duplicate": None}
    if "peak" in df:
        vals = np.unique(np.round(df["peak"].to_numpy(dtype=float), 6))
        out["peak_values"] = vals.tolist()
        out["peak_is_constant"] = bool(len(vals) == 1)
    if "crest_factor" in df and "peak_rms_ratio" in df:
        diff = np.abs(df["crest_factor"].to_numpy(float) - df["peak_rms_ratio"].to_numpy(float))
        out["crest_minus_ratio"] = float(diff.max())
        out["crest_is_duplicate"] = bool(np.allclose(diff, 0.0))
    return out


def distance_bin(deg):
    """Distance in degrees -> '<5', '5-10', '10-20', or None outside 0-20."""
    for name, lo, hi in DIST_BINS:
        if lo <= deg < hi:
            return name
    return None


def bin_medians(seis, features=None):
    """
    Seismometer feature medians per distance bin, laid out like the handout's
    table, with the handout value beside ours. Needs a dist_deg column.
    """
    features = features or list(HANDOUT_MEDIANS)
    if "dist_deg" not in seis:
        raise ValueError("seismometer table has no dist_deg column")
    bins = seis["dist_deg"].map(distance_bin)
    rows = []
    for f in features:
        row = {"feature": f}
        for name, _, _ in DIST_BINS:
            sel = seis.loc[bins == name, f]
            row[f"ours {name}"] = float(sel.median()) if len(sel) else np.nan
            row[f"handout {name}"] = HANDOUT_MEDIANS.get(f, {}).get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def range_table(seis, das, features=None, lo=5, hi=95, ref_label="EQ"):
    """
    For each feature: where the DAS values sit relative to the seismometer
    values of the same class (DAS events are earthquakes, so ref_label='EQ').

    Columns:
        seis p5, seis p95    the middle 90% of seismometer values
        DAS median
        DAS pct              percentile of the DAS median within the seismometer
                             values (50 = right in the middle)
        DAS inside           share of DAS rows inside [p5, p95]
        verdict              'inside', 'edge' or 'OUTSIDE'
    """
    features = features or FEATURES_11
    ref = seis[seis["label"] == ref_label] if "label" in seis else seis
    if ref.empty:
        raise ValueError(f"no seismometer rows with label {ref_label}")
    rows = []
    for f in features:
        s = ref[f].to_numpy(float)
        d = das[f].to_numpy(float)
        p_lo, p_hi = np.percentile(s, [lo, hi])
        med = float(np.median(d))
        pct = float((s < med).mean() * 100)
        inside = float(((d >= p_lo) & (d <= p_hi)).mean())
        if p_lo <= med <= p_hi and inside >= 0.5:
            verdict = "inside"
        elif p_lo <= med <= p_hi or inside >= 0.25:
            verdict = "edge"
        else:
            verdict = "OUTSIDE"
        rows.append({"feature": f, f"seis p{lo}": p_lo, f"seis p{hi}": p_hi,
                     "DAS median": med, "DAS pct": pct,
                     "DAS inside": inside, "verdict": verdict})
    return pd.DataFrame(rows).set_index("feature")


def plot_overlays(groups, features=None, handout=True, ncols=4, title=None, path=None):
    """
    One panel per feature, with a histogram for each group.

    groups  : dict name -> DataFrame, e.g. {"Seismometer EQ": ..., "DAS": ...}
              (empty frames are skipped)
    handout : draw the handout's per-bin medians as dashed lines
    The x range covers the 1st to 99th percentile of all groups together, so
    one extreme row does not squash everything else.
    """
    features = features or FEATURES_11
    groups = {k: v for k, v in groups.items() if v is not None and not v.empty}
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, f in zip(axes, features):
        allv = np.concatenate([g[f].to_numpy(float) for g in groups.values()])
        x0, x1 = np.percentile(allv, [1, 99])
        if x0 == x1:
            x0, x1 = x0 - 0.5, x1 + 0.5
        bins = np.linspace(x0, x1, 40)
        for (name, g), c in zip(groups.items(), colors):
            v = np.clip(g[f].to_numpy(float), x0, x1)
            ax.hist(v, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.35, color=c, label=f"{name} (n={len(g)})")
            ax.hist(v, bins=bins, density=True, histtype="step", color=c, lw=1)
        if handout and f in HANDOUT_MEDIANS:
            for name, val in HANDOUT_MEDIANS[f].items():
                if x0 <= val <= x1:
                    ax.axvline(val, color="k", ls="--", lw=0.8)
                    ax.text(val, ax.get_ylim()[1] * 0.95, f" {name}°",
                            fontsize=7, va="top")
        ax.set_title(f, fontsize=10)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)

    for ax in axes[len(features):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, frameon=False)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
