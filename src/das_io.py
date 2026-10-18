"""
Reading DAS event files in the AI4EPS "Seismic Event Format for DAS", which is
what the quakeflow_das datasets use (Monterey Bay, Arcata, Ridgecrest).

Each file has one dataser, "data", of shape(n_channels, n_time) in microstrain/s, plus event metadata as attributes.

Formar reference: https://ai4eps.github.io/homepage/ml4earth/seismic_event_format_das/
"""

import numpy as np
import h5py


def load_das_event(path):
    """
    Load one DAS event file.

    returns: (data, meta)
        data : 2D array (n_channels, n_time), float
        meta : dict of the file's attributes, plus derived keys
               'fs' (Hz), 'n_channels', 'n_time'
    """
    with h5py.File(path, "r") as f:
        dset = f["data"]
        data = np.asarray(dset[:], dtype=float)
        meta = {}
        for key, val in dset.attrs.items():
            # h5py hands back numpy bytes/scalars; make them plain Python
            if isinstance(val, bytes):
                val = val.decode()
            elif isinstance(val, np.generic):
                val = val.item()
            meta[key] = val

    dt = float(meta.get("dt_s", 0.01))
    meta["fs"] = 1.0 / dt
    meta["n_channels"], meta["n_time"] = data.shape
    return data, meta


def channel_snr(trace, split_index):
    """
    Crude signal-to-noise ratio for one channel: RMS after split_index over
    RMS before it. split_index should be at or just before the P arrival.

    Returns 0.0 if either side is empty or the noise window is silent.
    """
    trace = np.asarray(trace, dtype=float)
    noise = trace[:split_index]
    signal = trace[split_index:]
    if noise.size == 0 or signal.size == 0:
        return 0.0
    n_rms = np.sqrt(np.mean(noise ** 2))
    s_rms = np.sqrt(np.mean(signal ** 2))
    if n_rms == 0:
        return 0.0
    return float(s_rms / n_rms)


def select_channels(data, split_index, n=100, min_snr=2.0):
    """
    n best-SNR channels from a DAS waterfall.

    Matches the appraoch of the reference work we used.

    returns: (indices of chosen channels, their SNR values), best first
    """
    snrs = np.array([channel_snr(tr, split_index) for tr in data])
    order = np.argsort(snrs)[::-1]
    keep = [i for i in order if snrs[i] >= min_snr][:n]
    keep = np.array(keep, dtype=int)
    return keep, snrs[keep]


def event_time_index(meta):
    """
    Sample index of the event origin time, if the file records it.

    Note this is the ORIGIN time, not the P arrival: the P arrives later by
    the travel time. We use it as a lower bound for where to start looking for
    the P, not as the pick itself.
    """
    idx = meta.get("event_time_index")
    return int(idx) if idx is not None else None
