"""v3 blast experiment, step 2 (v3b, final): drop earthquakes within 50 km of the Ridgecrest sequence (2019-07 to 2021)
and above M2.5, then top up each station-year with M1.5-2.5 earthquakes from outside that zone.
Usage: python scripts/v3_blast_filter_eqs.py   -> data/gtune/v3b_eqs.h5
"""
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import v3_blast_download as D

MAINSHOCKS = [(35.705, -117.504), (35.770, -117.599)]  # M6.4 and M7.1
ZONE_KM = 50
WIN = (UTCDateTime(2019, 7, 1).timestamp, UTCDateTime(2022, 1, 1).timestamp)
MAX_MAG = 2.5
random.seed(11)


def zone_km(lat, lon):
    return min(gps2dist_azimuth(lat, lon, a, b)[0] / 1000 for a, b in MAINSHOCKS)


def locate(ids):
    out = {}
    for i in ids:
        try:
            o = D.us.get_events(eventid=i)[0].origins[0]
            out[i] = (o.latitude, o.longitude)
        except Exception:
            out[i] = (np.nan, np.nan)
    return out


if __name__ == "__main__":
    blasts = pd.read_hdf(os.path.join(D.OUT_DIR, "v3_blasts.h5"), "df")
    eqs = pd.read_hdf(os.path.join(D.OUT_DIR, "v3_eqs.h5"), "df")
    in_win = eqs["origin_time"].between(*WIN)
    loc = locate(eqs.loc[in_win, "comcat_id"].unique())
    eqs["zone_km"] = [zone_km(*loc[c]) if w else np.nan for c, w in zip(eqs["comcat_id"], in_win)]
    unlocated = in_win & eqs["zone_km"].isna()
    in_zone = in_win & ((eqs["zone_km"] <= ZONE_KM) | unlocated)
    too_big = eqs["mag"] > MAX_MAG
    print(f"v3 earthquakes: {len(eqs)}; in the 2019-07..2021 window {int(in_win.sum())}, of which within "
          f"{ZONE_KM} km {int(in_zone.sum())} ({int(unlocated.sum())} could not be located, dropped too); "
          f"above M{MAX_MAG} {int(too_big.sum())}; both {int((in_zone & too_big).sum())}")
    keep = eqs[~in_zone & ~too_big].drop(columns=["zone_km"]).copy()
    print(f"kept {len(keep)} of {len(eqs)}")

    need = blasts.groupby(["station", "year"]).size()
    have = keep.groupby(["station", "year"]).size().reindex(need.index, fill_value=0)
    gap = (need - have)[lambda s: s > 0]
    print(f"{len(gap)} station-years short, {int(gap.sum())} earthquakes to fill", flush=True)

    st = D.stations().set_index("station")
    done = set(zip(eqs["comcat_id"], eqs["station"]))
    new = []
    for j, ((sta, yr), n) in enumerate(gap.items(), 1):
        s = st.loc[sta]
        try:
            cat = D.us.get_events(starttime=UTCDateTime(yr, 1, 1), endtime=UTCDateTime(yr + 1, 1, 1),
                                  latitude=s.stla, longitude=s.stlo, maxradius=D.R_EV, minmagnitude=D.MAG[0],
                                  maxmagnitude=MAX_MAG, maxdepth=50, eventtype="earthquake", limit=2000)
        except Exception:
            print(f"[fill {j}/{len(gap)}] CI.{sta} {yr}: catalog query failed", flush=True)
            continue
        evs = [e for e in cat if not D.is_ridgecrest_test(e) and (D.ev_id(e), sta) not in done
               and zone_km(e.origins[0].latitude, e.origins[0].longitude) > ZONE_KM]
        random.shuffle(evs)
        got = []
        k, t = D.run_station(D.SimpleNamespace(station=sta, stla=s.stla, stlo=s.stlo), evs, int(n), "EQ", got)
        new += got
        print(f"[fill {j}/{len(gap)}] CI.{sta} {yr}: need {n}, {len(evs)} candidates outside the zone, kept {k}, tried {t}", flush=True)
    out = pd.concat([keep, pd.DataFrame(new)], ignore_index=True)
    out.to_hdf(os.path.join(D.OUT_DIR, "v3b_eqs.h5"), key="df", mode="w")
    short = (need - out.groupby(["station", "year"]).size().reindex(need.index, fill_value=0))[lambda s: s > 0]
    print(f"done: {len(out)} earthquakes ({len(keep)} kept + {len(new)} new) for {len(blasts)} blasts; "
          f"still short {int(short.sum())} at {len(short)} station-years; max M {out['mag'].max():.2f}", flush=True)
