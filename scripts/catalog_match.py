"""Match the 33 Monterey Bay DAS windows to catalog earthquakes (P arrival at ~10 s)."""
import glob
import os

import h5py
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

# Coordinates are the approximate SHORE END (Moss Landing), not the cable itself.
# The cable runs offshore, so distances to small local events (<1 deg) are
# approximate. Replace with the cable midpoint if the team has the geometry.
CABLE_LAT, CABLE_LON = 36.80, -121.79
P_TARGET_S = 10   # the Monterey files are cut with P at 10 s
P_TOL_S = 3       # a match needs P within 10 +/- 3 s
KEEP_S = 30       # near misses within 30 s are kept for diagnosis
# (max distance in degrees, min magnitude, seconds to look back before window)
TIERS = [(3, 0.0, 120), (10, 1.5, 300), (30, 3.0, 900), (180, 4.5, 1800)]

clients = []
for name in ("USGS", "NCEDC"):
    try:
        clients.append((name, Client(name)))
    except Exception as e:
        print("Could not connect to", name, "-", e)

model = TauPyModel("iasp91")


def read_window(path):
    with h5py.File(path, "r") as f:
        a = f["data"].attrs
        begin = UTCDateTime(str(a["begin_time"]).replace("+00:00", ""))
        end = UTCDateTime(str(a["end_time"]).replace("+00:00", ""))
    return begin, end


def find_candidates(begin, end):
    found = {}
    for cname, client in clients:
        for max_deg, min_mag, lookback in TIERS:
            try:
                cat = client.get_events(
                    starttime=begin - lookback, endtime=end,
                    latitude=CABLE_LAT, longitude=CABLE_LON,
                    maxradius=max_deg, minmagnitude=min_mag)
            except FDSNNoDataException:
                continue
            except Exception as e:
                if "Could not parse" not in str(e):
                    print("  warning:", cname, "query failed:", str(e)[:80])
                continue
            for ev in cat:
                o = ev.preferred_origin() or ev.origins[0]
                m = ev.preferred_magnitude() or (
                    ev.magnitudes[0] if ev.magnitudes else None)
                dist = locations2degrees(
                    CABLE_LAT, CABLE_LON, o.latitude, o.longitude)
                depth_km = (o.depth or 10000) / 1000.0
                arr = model.get_travel_times(
                    source_depth_in_km=depth_km, distance_in_degree=dist,
                    phase_list=["ttp"])
                if not arr:
                    continue
                p_in = float(o.time + arr[0].time - begin)
                dev = abs(p_in - P_TARGET_S)
                if dev <= KEEP_S:
                    found[cname + ":" + str(ev.resource_id)] = dict(
                        catalog=cname, origin=str(o.time),
                        mag=m.mag if m else None,
                        lat=o.latitude, lon=o.longitude, depth_km=depth_km,
                        dist_deg=dist, dist_km=dist * 111.19,
                        p_in_window_s=round(p_in, 1), dev_s=round(dev, 1))
    return list(found.values())


def dist_bin(d):
    return "<5" if d < 5 else "5-10" if d < 10 else "10-20" if d < 20 else ">20"


rows = []
for path in sorted(glob.glob("data/monterey_bay/data/*.h5")):
    begin, end = read_window(path)
    cands = find_candidates(begin, end)
    row = {"file": os.path.basename(path), "window_start": str(begin),
           "n_candidates": len(cands), "matched": False}
    if cands:
        best = min(cands, key=lambda c: c["dev_s"])
        row.update(best)
        row["bin"] = dist_bin(best["dist_deg"])
        row["matched"] = best["dev_s"] <= P_TOL_S
    rows.append(row)
    status = "MATCH" if row["matched"] else ("near miss" if cands else "none")
    print(row["file"], "->", status, flush=True)

df = pd.DataFrame(rows)
df.to_csv("data/catalog_match_v2.csv", index=False)
cols = ["file", "catalog", "n_candidates", "mag", "dist_deg", "bin",
        "p_in_window_s", "dev_s", "matched"]
print(df[[c for c in cols if c in df.columns]].to_string())
print("\nMatched (P within 10 +/- 3 s):", int(df["matched"].sum()), "of", len(df))
for tol in (3, 5, 6):
    ok = df["dev_s"] <= tol
    print(f"Within +/-{tol} s of P=10 s:", int(ok.sum()), "of", len(df))