"""Earthquakes at the GTUNE stations (BHZ, 60 s, P at 10 s), in the GTUNE layout.
Usage: python scripts/download_eqs.py [start_station]   -> data/gtune/eqs_iris.h5
"""
import os
import random
import sys
import time

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

sys.path.insert(0, ".")
from src.training_set import load_gtune

PER_STATION = 5
MIN_MAG, MAX_DEPTH_KM = 4.0, 50
MIN_DIST, MAX_DIST = 1.5, 20.0
OUT = "data/gtune/eqs_iris.h5"

events_client = Client("USGS", timeout=60)
wave_client = Client("EARTHSCOPE", timeout=60)  # formerly IRIS
model = TauPyModel("iasp91")
random.seed(0)


def gtune_stations():
    g = load_gtune()
    return g.drop_duplicates(["net", "station"])[["net", "station", "stla", "stlo"]]


def events_near(lat, lon):
    cat = events_client.get_events(
        starttime=UTCDateTime(2000, 1, 1), endtime=UTCDateTime(2020, 1, 1),
        latitude=lat, longitude=lon, minradius=MIN_DIST, maxradius=MAX_DIST,
        minmagnitude=MIN_MAG, maxdepth=MAX_DEPTH_KM, eventtype="earthquake", limit=200)
    events = list(cat)
    random.shuffle(events)
    return events


def fetch_record(net, sta, lat, lon, ev):
    o = ev.preferred_origin() or ev.origins[0]
    dist = locations2degrees(lat, lon, o.latitude, o.longitude)
    arr = model.get_travel_times(source_depth_in_km=(o.depth or 0) / 1000,
                                 distance_in_degree=dist, phase_list=["ttp"])
    if not arr:
        return None
    p = o.time + arr[0].time
    time.sleep(0.5)  # go easy on the server
    st = wave_client.get_waveforms(net, sta, "*", "BHZ", p - 10, p + 50)
    if len(st) == 0:
        return None
    tr = st[0]
    if abs(tr.stats.starttime - (p - 10)) > 0.5:  # gap at the start, P wouldn't be at 10 s
        return None
    return dict(origin_time=o.time.timestamp, trace_start_time=tr.stats.starttime.timestamp,
                p_arrival=arr[0].time, sampling_rate=tr.stats.sampling_rate,
                station_dist=dist, stla=lat, stlo=lon, station=sta,
                evid="EQ" + o.time.strftime("%Y%m%d%H%M%S"), net=net,
                chn=tr.stats.channel, waveform=tr.data.astype(float))


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = []
    if start > 1 and os.path.exists(OUT):
        rows = pd.read_hdf(OUT, "df").to_dict("records")
        print("resuming with", len(rows), "saved records")
    stations = gtune_stations()
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
                rec = fetch_record(s.net, s.station, s.stla, s.stlo, ev)
            except Exception:
                rec = None
            if rec:
                rows.append(rec)
                kept += 1
        print(f"[{i}/{len(stations)}] {s.net}.{s.station}: {kept} records", flush=True)
        if i % 20 == 0:
            pd.DataFrame(rows).to_hdf(OUT, key="df", mode="w")
    pd.DataFrame(rows).to_hdf(OUT, key="df", mode="w")
    print("saved", len(rows), "earthquake records")
