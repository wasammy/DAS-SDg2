"""v3 blast experiment, step 1: quarry blasts (M1.5-3.5) at CI BHZ stations within 1.5 deg of the Ridgecrest cable,
plus the same number of earthquakes per station-year; Ridgecrest DAS catalog events left out.
Usage: python scripts/v3_blast_download.py   -> data/gtune/v3_blasts.h5, v3_eqs.h5
"""
import os
import random
import re
import sys
import time
import warnings
from types import SimpleNamespace

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

warnings.filterwarnings("ignore")
OUT_DIR = "data/gtune"
CABLE = (35.65, -117.66)
R_STA, R_EV = 1.5, 1.5
MAG = (1.5, 3.5)
YEARS = (2000, 2022)
PER_STATION = 4
sc, us = Client("SCEDC", timeout=60), Client("USGS", timeout=60)
model = TauPyModel("iasp91")
random.seed(7)

rc = pd.read_csv("data/ridgecrest_catalog.csv")
RC_IDS = set(rc["id"])
RC_T = pd.to_datetime(rc["time"], format="ISO8601").astype("int64").to_numpy() / 1e9


def ev_id(ev):
    m = re.search(r"eventid=([a-z]{2}\d+)", str(ev.resource_id))
    return m.group(1) if m else ""


def is_ridgecrest_test(ev):
    t = ev.origins[0].time.timestamp
    return ev_id(ev) in RC_IDS or (abs(RC_T - t) < 5).any()


def fetch(net, sta, lat, lon, ev, prefix):
    o = ev.preferred_origin() or ev.origins[0]
    dist = locations2degrees(lat, lon, o.latitude, o.longitude)
    arr = model.get_travel_times(source_depth_in_km=max(0.0, (o.depth or 0) / 1000),
                                 distance_in_degree=dist, phase_list=["ttp"])
    if not arr:
        return None
    p = o.time + arr[0].time
    time.sleep(0.3)
    st = sc.get_waveforms(net, sta, "*", "BHZ", p - 10, p + 50)
    if len(st) == 0:
        return None
    tr = st[0]
    if abs(tr.stats.starttime - (p - 10)) > 0.5 or tr.stats.npts < 59.9 * tr.stats.sampling_rate:
        return None
    m = ev.preferred_magnitude() or ev.magnitudes[0]
    return dict(origin_time=o.time.timestamp, trace_start_time=tr.stats.starttime.timestamp,
                p_arrival=arr[0].time, sampling_rate=tr.stats.sampling_rate, station_dist=dist,
                stla=lat, stlo=lon, station=sta, net=net, chn=tr.stats.channel,
                evid=prefix + o.time.strftime("%Y%m%d%H%M%S"), comcat_id=ev_id(ev), mag=m.mag,
                event_type=str(ev.event_type), year=o.time.year, waveform=tr.data.astype(float))


def stations():
    inv = sc.get_stations(network="CI", channel="BHZ", latitude=CABLE[0], longitude=CABLE[1],
                          maxradius=R_STA, level="station")
    rows = [(s.code, s.latitude, s.longitude) for n in inv for s in n]
    return pd.DataFrame(rows, columns=["station", "stla", "stlo"]).drop_duplicates("station").reset_index(drop=True)


def run_station(s, events, need, prefix, out):
    kept = tried = 0
    for ev in events:
        if kept >= need or tried >= max(15, 5 * need):
            break
        tried += 1
        try:
            rec = fetch("CI", s.station, s.stla, s.stlo, ev, prefix)
        except Exception:
            rec = None
        if rec:
            out.append(rec)
            kept += 1
    return kept, tried


if __name__ == "__main__":
    st = stations()
    print(len(st), "CI BHZ stations within", R_STA, "deg of the cable", flush=True)
    blasts = us.get_events(starttime=UTCDateTime(YEARS[0], 1, 1), endtime=UTCDateTime(YEARS[1], 1, 1),
                           latitude=CABLE[0], longitude=CABLE[1], maxradius=R_STA + R_EV,
                           minmagnitude=MAG[0], maxmagnitude=MAG[1], eventtype="quarry blast", limit=20000)
    blasts = list(blasts)
    print(len(blasts), "quarry blasts in the catalog", flush=True)

    B = []
    for i, s in enumerate(st.itertuples(), 1):
        near = [e for e in blasts if locations2degrees(s.stla, s.stlo, e.origins[0].latitude, e.origins[0].longitude) <= R_EV]
        random.shuffle(near)
        k, t = run_station(s, near, PER_STATION, "QB", B)
        print(f"[blast {i}/{len(st)}] CI.{s.station}: {len(near)} nearby, kept {k}, tried {t}", flush=True)
        if k:
            pd.DataFrame(B).to_hdf(os.path.join(OUT_DIR, "v3_blasts.h5"), key="df", mode="w")

    need = pd.DataFrame(B).groupby(["station", "year"]).size()
    loc = st.set_index("station")
    E = []
    for j, ((sta, yr), n) in enumerate(need.items(), 1):
        s = loc.loc[sta]
        try:
            cat = us.get_events(starttime=UTCDateTime(yr, 1, 1), endtime=UTCDateTime(yr + 1, 1, 1),
                                latitude=s.stla, longitude=s.stlo, maxradius=R_EV, minmagnitude=MAG[0],
                                maxmagnitude=MAG[1], maxdepth=50, eventtype="earthquake", limit=2000)
        except Exception:
            print(f"[eq {j}/{len(need)}] CI.{sta} {yr}: catalog query failed", flush=True)
            continue
        evs = [e for e in cat if not is_ridgecrest_test(e)]
        random.shuffle(evs)
        k, t = run_station(SimpleNamespace(station=sta, stla=s.stla, stlo=s.stlo), evs, int(n), "EQ", E)
        print(f"[eq {j}/{len(need)}] CI.{sta} {yr}: need {n}, kept {k}, tried {t} "
              f"({len(cat) - len(evs)} Ridgecrest test events skipped)", flush=True)
        if k:
            pd.DataFrame(E).to_hdf(os.path.join(OUT_DIR, "v3_eqs.h5"), key="df", mode="w")
    print(f"done: {len(B)} blast records, {len(E)} earthquake records", flush=True)
