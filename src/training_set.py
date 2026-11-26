"""seismograph training set: GTUNE explosions+IRIS earthquakes."""
import numpy as np
import pandas as pd
from src.preprocess import preprocess_trace
from src.features import extract_features

P_OFFSET_S = 10.0  # GTUNE traces are cut with P at 10s (README+handout)

MAX_DIST_DEG = 20.0
MIN_FS = 20.0


def load_gtune(path="data/gtune/Digital_data_fdsn.h5"):
    """GTUNE explosion records within 20 deg, sampled at 20 Hz or more, full 60 s."""
    df = pd.read_hdf(path, "/df")
    df = df[df["station_dist"] <= MAX_DIST_DEG]
    df = df[df["sampling_rate"] >= MIN_FS - 0.01]
    df = df[df["waveform"].apply(lambda w: w is not None and len(w) > 0)]
    df = df.drop_duplicates(subset=["evid", "net", "station", "chn"])
    n = df["waveform"].apply(len)
    df = df[n >= 59.9 * df["sampling_rate"]]   # need the full 60 s
    return df.reset_index(drop=True)

def feature_rows(df, label):
    """One row per trace: ids, distance, label, and the 13 features."""
    rows = []
    for _, r in df.iterrows():
        fs = r["sampling_rate"]
        try:
            win, _ = preprocess_trace(r["waveform"], fs, p_index=round(P_OFFSET_S * fs))
        except ValueError:
            continue  # too short after resampling, or flat
        rows.append({"event_id": str(r["evid"]), "net": r["net"],
                     "station": r["station"], "dist_deg": r["station_dist"],
                     "label": label, **extract_features(win)})
    return pd.DataFrame(rows)