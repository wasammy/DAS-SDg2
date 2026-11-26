"""Earthquakes 1.5-5 deg from GTUNE stations that recorded an explosion under 5 deg.
Usage: python scripts/download_eqs_near.py [start_station]   -> data/gtune/eqs_near.h5
"""
import os
import random
import sys

import pandas as pd
from obspy import UTCDateTime

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import download_eqs as base

PER_STATION = 12
MIN_MAG, MAX_DEPTH_KM = base.MIN_MAG, base.MAX_DEPTH_KM
MIN_DIST, MAX_DIST = 1.5, 5.0
OUT = "data/gtune/eqs_near.h5"
random.seed(1)


def events_near(lat, lon):
    cat = base.events_client.get_events(
        starttime=UTCDateTime(2000, 1, 1), endtime=UTCDateTime(2020, 1, 1),
        latitude=lat, longitude=lon, minradius=MIN_DIST, maxradius=MAX_DIST,
        minmagnitude=MIN_MAG, maxdepth=MAX_DEPTH_KM, limit=200)
    events = list(cat)
    random.shuffle(events)
    return events


def near_explosion_stations():
    g = base.load_gtune()
    g = g[g["station_dist"] < MAX_DIST]
    return g.drop_duplicates(["net", "station"])[["net", "station", "stla", "stlo"]]


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = []
    if start > 1 and os.path.exists(OUT):
        rows = pd.read_hdf(OUT, "df").to_dict("records")
        print("resuming with", len(rows), "saved records")
    
    done = set()
    if os.path.exists(base.OUT):
        old = pd.read_hdf(base.OUT, "df")
        done = set(zip(old["evid"], old["net"], old["station"]))
    stations = near_explosion_stations()
    print(len(stations), "stations recorded an explosion under", MAX_DIST, "deg")
    for i, s in enumerate(stations.itertuples(), 1):
        if i < start:
            continue
        try:
            events = events_near(s.stla, s.stlo)
        except Exception as e:
            print(f"[{i}/{len(stations)}] {s.net}.{s.station}: no events ({str(e)[:60]})")
            continue
        kept = 0
        for ev in events:
            if kept >= PER_STATION:
                break
            try:
                rec = base.fetch_record(s.net, s.station, s.stla, s.stlo, ev)
            except Exception:
                rec = None
            if rec and (rec["evid"], rec["net"], rec["station"]) not in done:
                rows.append(rec)
                kept += 1
        print(f"[{i}/{len(stations)}] {s.net}.{s.station}: {kept} records", flush=True)
        if kept:
            pd.DataFrame(rows).to_hdf(OUT, key="df", mode="w")
    if rows:
        pd.DataFrame(rows).to_hdf(OUT, key="df", mode="w")
    print("saved", len(rows), "near earthquake records")
