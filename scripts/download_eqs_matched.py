"""Earthquakes matched to the under-5 deg GTUNE explosions: same station, same year, same count (1.5-5 deg, M>=4, M3-4 to fill).
Usage: python scripts/download_eqs_matched.py [start_target]   -> data/gtune/eqs_matched.h5
"""
import os
import random
import sys

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import download_eqs as base
from src.training_set import load_gtune

OUT = "data/gtune/eqs_matched.h5"
MIN_DIST, MAX_DIST = 1.5, 5.0
CLIENTS = {"EARTHSCOPE": base.wave_client, "NCEDC": Client("NCEDC", timeout=60)}
random.seed(5)


def targets():
    g = load_gtune()
    g = g[g["station_dist"] < MAX_DIST].copy()
    g["year"] = pd.to_datetime(g["origin_time"], unit="s").dt.year
    t = g.groupby(["net", "station", "year"]).agg(need=("evid", "size"), stla=("stla", "first"),
                                                 stlo=("stlo", "first"))
    return t.reset_index()


def events(lat, lon, year, min_mag, max_mag):
    cat = base.events_client.get_events(
        starttime=UTCDateTime(year, 1, 1), endtime=UTCDateTime(year + 1, 1, 1),
        latitude=lat, longitude=lon, minradius=MIN_DIST, maxradius=MAX_DIST,
        minmagnitude=min_mag, maxmagnitude=max_mag, maxdepth=base.MAX_DEPTH_KM,
        eventtype="earthquake", limit=2000)
    ev = list(cat)
    random.shuffle(ev)
    return ev


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = []
    if start > 1 and os.path.exists(OUT):
        rows = pd.read_hdf(OUT, "df").to_dict("records")
        print("resuming with", len(rows), "saved records")
    done = {(r["evid"], r["net"], r["station"]) for r in rows}
    for f in ("data/gtune/eqs_iris.h5", "data/gtune/eqs_near.h5"):
        if os.path.exists(f):
            d = pd.read_hdf(f, "df")
            done |= set(zip(d["evid"], d["net"], d["station"]))

    t = targets()
    for i, r in enumerate(t.itertuples(), 1):
        if i < start:
            continue
        base.wave_client = CLIENTS["NCEDC" if r.net == "BK" else "EARTHSCOPE"]
        kept = tried = 0
        for lo, hi in ((4.0, None), (3.0, 3.999)):
            if kept >= r.need:
                break
            try:
                evs = events(r.stla, r.stlo, r.year, lo, hi)
            except Exception:
                continue
            for ev in evs[:max(30, 6 * r.need)]:
                if kept >= r.need:
                    break
                tried += 1
                try:
                    rec = base.fetch_record(r.net, r.station, r.stla, r.stlo, ev)
                except Exception:
                    rec = None
                if rec and (rec["evid"], rec["net"], rec["station"]) not in done:
                    rec["mag"] = ev.preferred_magnitude().mag if ev.preferred_magnitude() else None
                    rows.append(rec)
                    done.add((rec["evid"], rec["net"], rec["station"]))
                    kept += 1
        print(f"[{i}/{len(t)}] {r.net}.{r.station} {r.year}: need {r.need}, kept {kept}, tried {tried}", flush=True)
        if kept:
            pd.DataFrame(rows).to_hdf(OUT, key="df", mode="w")
    if rows:
        pd.DataFrame(rows).to_hdf(OUT, key="df", mode="w")
    print("saved", len(rows), "matched earthquake records")
