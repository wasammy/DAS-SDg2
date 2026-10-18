"""
The RF-v2 feature set, from Appendices A and B of the EQ/UNE
machine learning handout (Sep 2026).

Expects a 1200-sample, 20 Hz, peak-normalised window with the P arrival at
sample 200 (output of preprocess.preprocess_trace).

Window layout (Appendix A), boundaries define what early_e, middle_e
and late_e mean, so do not change them without changing the handout's
definitions too:
      0 -  199 pre-P noise
    200 -  299 early signal (P arrival at sample 200)
    300 -  599 middle signal
    600 - 1199 late signal / coda

Not done here: the original RF's noise-referenced features (noise_e,
early/middle/late_noise_ratio). RF-v2 dropped them because they depend on
station noise. Add them back only if we decide to reproduce the original RF
for comparison.
"""

import numpy as np

# segment boundaries, in samples
EARLY = slice(200, 300)
MIDDLE = slice(300, 600)
LATE = slice(600, 1200)

FS = 20.0

# the 13 features RF-v2 uses, in the handout's order
RF_V2_FEATURES = [
    "rms",
    "peak",
    "peak_rms_ratio",
    "early_e",
    "middle_e",
    "late_e",
    "dominant_frequency",
    "spectral_centroid",
    "bandwidth",
    "crest_factor",
    "duration_20",
    "duration_10",
    "energy_centroid_time",
]


def extract_features(x, fs=FS, duration_in_samples=True):
    """
    Compute the 13 RF-v2 features for one window.

    x: 1D array, 1200 samples at 20 Hz with P at sample 200
    duration_in_samples: if True, duration_10/duration_20 count samples above
        the threshold; if False, they are converted to seconds. The handout
        never states the unit. reported values (duration_20 ~14-23,
        duration_10 ~28-37) match sample counts at 20 Hz, so that is the
        default.

    returns: dict of feature name -> float
    """
    x = np.asarray(x, dtype=float)
    f = {}

    f["rms"] = float(np.sqrt(np.mean(x ** 2)))
    f["peak"] = float(np.abs(x).max())
    f["peak_rms_ratio"] = f["peak"] / f["rms"] if f["rms"] > 0 else 0.0

    f["early_e"] = float(np.mean(x[EARLY] ** 2))
    f["middle_e"] = float(np.mean(x[MIDDLE] ** 2))
    f["late_e"] = float(np.mean(x[LATE] ** 2))

    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    amp = np.abs(np.fft.rfft(x))
    total = amp.sum()
    if total > 0:
        centroid = float((freq * amp).sum() / total)
        f["dominant_frequency"] = float(freq[np.argmax(amp)])
        f["spectral_centroid"] = centroid
        f["bandwidth"] = float(np.sqrt((amp * (freq - centroid) ** 2).sum() / total))
    else:
        f["dominant_frequency"] = f["spectral_centroid"] = f["bandwidth"] = 0.0

    # Appendix B defines CrestFactor = Peak / RMS, which is the same formula
    # Appendix A gives for PeakRMSRatio. Both appear in the handout's list of
    # 13, and Appendix C reports identical importance for them (0.016 each).
    # Kept as duplicates so our feature set matches his exactly.
    # Do not "fix" this without asking the professor first.
    f["crest_factor"] = f["peak_rms_ratio"]

    scale = 1.0 if duration_in_samples else 1.0 / fs
    f["duration_20"] = float(np.count_nonzero(np.abs(x) > 0.20 * f["peak"]) * scale)
    f["duration_10"] = float(np.count_nonzero(np.abs(x) > 0.10 * f["peak"]) * scale)

    energy = x ** 2
    e_sum = energy.sum()
    if e_sum > 0:
        t = np.arange(len(x)) / fs
        f["energy_centroid_time"] = float((t * energy).sum() / e_sum)
    else:
        f["energy_centroid_time"] = 0.0

    return f

def feature_vector(x, fs=FS, names=None, **kwargs):
    """Features as a 1D array in a fixed order (default: RF_V2_FEATURES)."""
    names = names or RF_V2_FEATURES
    f = extract_features(x, fs, **kwargs)
    return np.array([f[n] for n in names], dtype=float)


def feature_table(windows, fs=FS, names=None, **kwargs):
    """
    Build a feature matrix from many windows.

    windows : iterable of 1D arrays
    returns : (array of shape (n_windows, n_features), list of feature names)
    """
    names = names or RF_V2_FEATURES
    rows = [feature_vector(w, fs, names, **kwargs) for w in windows]
    return (np.vstack(rows) if rows else np.empty((0, len(names)))), names
