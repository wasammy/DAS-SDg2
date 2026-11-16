"""
Stacking DAS channels into one waveform per event.

Step 1 (this file so far): substack() averages blocks of neighboring channels.
Within ~50 channels (~250 m of cable) the P arrival is only ~0.04 s out of step,
so the channels can be averaged directly without cancelling the signal.
"""
import numpy as np
from src.preprocess import bandpass

DEAD_STD = 1e-12  # channels with std below this (or NaN) are treated as dead


def substack(data, fs, group=50, filtered=True):
    """
    Average blocks of `group` neighboring channels.

    data: array (n_channels, n_time), as returned by load_das_event
    fs: sampling rate in Hz
    group: channels per block; leftover channels at the end are dropped
    filtered: if True, bandpass each channel 0.5-10 Hz before normalizing.
        If False, only remove the mean (use this for the final stack, so the
        signal is filtered once, inside preprocess_trace).

    Each channel is divided by its own std so loud channels don't dominate.
    Dead channels (std ~ 0 or NaN) are skipped. A block with no live
    channels stays all zeros.

    returns: (stacks, counts)
        stacks: array (n_groups, n_time), n_groups = n_channels // group
        counts: array (n_groups,), number of live channels used per block
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_channels, n_time), got shape {data.shape}")

    n_ch, n_time = data.shape
    n_groups = n_ch // group
    if n_groups == 0:
        raise ValueError(f"only {n_ch} channels, fewer than group={group}")

    stacks = np.zeros((n_groups, n_time))
    counts = np.zeros(n_groups, dtype=int)

    for g in range(n_groups):
        total = np.zeros(n_time)
        for c in range(g * group, (g + 1) * group):
            # loop per channel: bandpass() removes np.mean over the whole input,
            # so passing a 2D block would subtract one shared mean
            if filtered:
                x = bandpass(data[c], fs)
            else:
                x = np.asarray(data[c], dtype=float)
                x = x - np.mean(x)
            s = x.std()
            if not np.isfinite(s) or s < DEAD_STD:
                continue
            total += x / s
            counts[g] += 1
        if counts[g] > 0:
            stacks[g] = total / counts[g]

    return stacks, counts
