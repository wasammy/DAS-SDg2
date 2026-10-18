"""
Preprocessing to put any waveform into the format the EQ/UNE classifiers
expect, as handout suggested:

    0.5-10 Hz bandpass, 20 Hz, 60 s window (10 s pre-P + 50 s post-P)
    = 1200 samples with the P arrival at sample 200, peak-normalised.

The same functions run on GTUNE seismograms and on DAS channels. Both classes
must go through SAME processing, or a difference between them could come
from the way we wrote it rather than the physics.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

TARGET_FS = 20.0
PRE_P_SAMPLES = 200 # 10 s at 20 Hz
WINDOW_SAMPLES = 1200  # 60 s at 20 Hz


def bandpass(x, fs, fmin=0.5, fmax=10.0, order=4):
    """
    Bandpass filter a 1D waveform. Returns the filtered trace, same length.

    fmax is nudged below the Nyquist frequency (fs/2) when it would land on
    or above it -- at exactly 20 Hz, a literal 10 Hz edge is not
    constructible. This is why filtering must happen BEFORE downsampling.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)  # raw counts sit far from zero

    hi = min(fmax, 0.995 * fs / 2.0)
    if not (0 < fmin < hi):
        raise ValueError(f"cannot build a {fmin}-{fmax} Hz band at fs={fs} Hz")

    sos = butter(order, [fmin, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def resample(x, fs, target=TARGET_FS):
    """
    Resample to the target rate. Returns (waveform, new rate).

    Raises ValueError below the target.
    """
    x = np.asarray(x, dtype=float)

    if np.isclose(fs, target, atol=0.01): # GTUNE has rates like 19.99993
        return x, target
    if fs < target:
        raise ValueError(f"sampling rate {fs} Hz is below the target {target} Hz")

    # resample_poly needs integers; x100 keeps two decimals so 39.999999 Hz
    # becomes 4000 rather than going down to 40
    return resample_poly(x, int(round(target * 100)), int(round(fs * 100))), target


def pick_p(x, fs, sta_s=1.0, lta_s=10.0, threshold=3.0):
    """
    P arrival with STA/LTA: the ratio of a short-term average of
    signal energy to a long-term average, which spikes when energy arrives
    suddenly. Same method as last semester's DAS notebook.

    Returns the sample index of the first trigger, or None if nothing fires.
    """
    x = np.asarray(x, dtype=float)
    nsta = max(1, int(sta_s * fs))
    nlta = max(nsta + 1, int(lta_s * fs))
    if len(x) < nlta + nsta:
        return None

    csum = np.cumsum(np.insert(x ** 2, 0, 0.0)) # running without a loop
    sta = (csum[nsta:] - csum[:-nsta]) / nsta
    lta = (csum[nlta:] - csum[:-nlta]) / nlta

    n = min(len(sta) - (nlta - nsta), len(lta))
    if n <= 0:
        return None
    sta = sta[nlta - nsta: nlta - nsta + n]
    lta = lta[:n]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lta > 0, sta / lta, 0.0)

    hits = np.flatnonzero(ratio > threshold)
    return int(hits[0] + nlta) if hits.size else None


def cut_window(x, p_index, pre=PRE_P_SAMPLES, total=WINDOW_SAMPLES):
    """
    Cut the 1200-sample window so the P lands on sample `pre` (200).

    Raises ValueError if the trace is too short either side. Padding with
    zeros would distort the energy and duration features.
    """
    x = np.asarray(x, dtype=float)
    start = p_index - pre
    stop = start + total
    if start < 0 or stop > len(x):
        raise ValueError(
            f"window [{start}:{stop}] does not fit in a trace of {len(x)} samples"
        )
    return x[start:stop]


def normalize(x):
    """Scale so the largest absolute value is 1. Raises ValueError on a flat trace."""
    x = np.asarray(x, dtype=float)
    peak = np.abs(x).max()
    if peak == 0:
        raise ValueError("trace is all zeros")
    return x / peak


def preprocess_trace(x, fs, p_index=None, do_normalize=True):
    """
    Full chain: bandpass -> resample -> pick P -> cut -> normalise.

    p_index : sample index of the P in the ORIGINAL trace if known;
              None to find it with STA/LTA.

    Returns (1200-sample window, info dict). Raises ValueError if the trace
    cannot be processed.
    """
    x = np.asarray(x, dtype=float)
    original_fs = float(fs)
    info = {"original_fs": original_fs, "original_samples": len(x)}

    if original_fs < TARGET_FS and not np.isclose(original_fs, TARGET_FS, atol=0.01):
        raise ValueError(f"sampling rate {original_fs} Hz cannot support a 0.5-10 Hz band")

    # filter first, then downsample: removing everything above 10 Hz means the
    # downsample cannot alias, and the 10 Hz edge is unbuildable at 20 Hz
    x = bandpass(x, original_fs)
    x, fs = resample(x, original_fs)
    info["resampled_samples"] = len(x)

    if p_index is not None:
        p_index = int(round(p_index * fs / original_fs)) # rescale
        info["p_source"] = "metadata"
    else:
        p_index = pick_p(x, fs)
        info["p_source"] = "sta_lta"
        if p_index is None:
            raise ValueError("STA/LTA found no arrival")

    info["p_index"] = int(p_index)
    info["p_time_s"] = p_index / fs

    x = cut_window(x, p_index)
    if do_normalize:
        x = normalize(x)

    return x, info





