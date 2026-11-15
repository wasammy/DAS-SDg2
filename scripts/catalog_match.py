"""Match the 33 Monterey Bay DAS windows to USGS earthquakes."""
import glob
import os

import h5py
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

# Approximate cable location (Moss Landing). Confirm with the team.
CABLE_LAT, CABLE_LON = 36.80, -121.79
TOLERANCE_S = 10
# (max distance in degrees, min magnitude, seconds to look back before window)
TIERS = [(3, 0.0, 120), (10, 1.5, 300), (30, 3.0, 900), (180, 4.5, 1800)]

client = Client("USGS")
model = TauPyModel("iasp91")


def read_window(path):
    with h5py.File(path, "r") as f:
        a = f["data"].attrs
        begin = UTCDateTime(str(a["begin_time"]).replace("+00:00", ""))
        end = UTCDateTime(str(a["end_time"]).replace("+00:00", ""))
    return begin, end


def find_candidates(begin, end):
    found = {}
    for max_deg, min_mag, lookback in TIERS:
        try:
            cat = client.get_events(
                starttime=begin - lookback, endtime=end,
                latitude=CABLE_LAT, longitude=CABLE_LON,
                maxradius=max_deg, minmagnitude=min_mag)
        except FDSNNoDataException:
            continue
        for ev in cat:
            o = ev.preferred_origin() or ev.origins[0]
            m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            dist = locations2degrees(CABLE_LAT, CABLE_LON, o.latitude, o.longitude)
            depth_km = (o.depth or 10000) / 1000.0
            arr = model.get_travel_times(
                source_depth_in_km=depth_km, distance_in_degree=dist,
                phase_list=["ttp"])
            if not arr:
                continue
            p_time = o.time + arr[0].time
            if begin - TOLERANCE_S <= p_time <= end + TOLERANCE_S:
                found[str(ev.resource_id)] = dict(
                    origin=str(o.time), mag=m.mag if m else None,
                    lat=o.latitude, lon=o.longitude, depth_km=depth_km,
                    dist_deg=dist, dist_km=dist * 111.19,
                    p_in_window_s=round(float(p_time - begin), 1))
    return list(found.values())


def dist_bin(d):
    return "<5" if d < 5 else "5-10" if d < 10 else "10-20" if d < 20 else ">20"


rows = []
for path in sorted(glob.glob("data/monterey_bay/data/*.h5")):
    begin, end = read_window(path)
    cands = find_candidates(begin, end)
    row = {"file": os.path.basename(path), "window_start": str(begin),
           "n_candidates": len(cands), "matched": bool(cands)}
    if cands:
        best = min(cands, key=lambda c: c["dist_deg"])
        row.update(best)
        row["bin"] = dist_bin(best["dist_deg"])
    rows.append(row)
    print(row["file"], "->", "match" if cands else "NO MATCH", flush=True)

df = pd.DataFrame(rows)
df.to_csv("data/catalog_match.csv", index=False)
cols = ["file", "n_candidates", "origin", "mag", "dist_deg", "bin", "p_in_window_s"]
print(df[[c for c in cols if c in df.columns]].to_string())
print("\nMatched:", int(df["matched"].sum()), "of", len(df))
if "bin" in df.columns:
    print(df["bin"].value_counts().to_string())