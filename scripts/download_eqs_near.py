"""Earthquakes within 5 deg of the GTUNE stations, to balance the <5 deg bin.

download_eqs.py draws events from 1.5-20 deg at random, and most of that area is
far away, so it returns few near events (7 of 85 below 5 deg vs 190 explosions).
This script asks only for 1.5-5 deg events, with the same magnitude, depth,
channel and 60 s window as download_eqs.py.

Why not closer than 1.5 deg: the closest GTUNE explosion is 1.8 deg. Earthquakes
with no explosions at the same distance would let the model learn "close means
earthquake", which would make the DAS test (events < 1 deg) look good for the
wrong reason.

Saves after every station. Resume with the station number it stopped at.
Usage: python scripts/download_eqs_near.py [start_station]
Writes data/gtune/eqs_near.h5
"""
import os
import random
import sys

import pandas as pd
from obspy import UTCDateTime

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import download_eqs as base   # reuses its clients, station list and fetch_record

PER_STATION = 5
MIN_MAG, MAX_DEPTH_KM = base.MIN_MAG, base.MAX_DEPTH_KM   # same as download_eqs.py
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


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = []
    if start > 1 and os.path.exists(OUT):
        rows = pd.read_hdf(OUT, "df").to_dict("records")
        print("resuming with", len(rows), "saved records")
    # skip (event, station) pairs that download_eqs.py already has
    done = set()
    if os.path.exists(base.OUT):
        old = pd.read_hdf(base.OUT, "df")
        done = set(zip(old["evid"], old["net"], old["station"]))
    stations = base.gtune_stations()
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
