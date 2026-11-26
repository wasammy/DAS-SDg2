"""
Pick and download Ridgecrest DAS event files (AI4EPS/quakeflow_das, ridgecrest_north).

Events are chosen from the catalog by fixed rules, not by looking at the data,
so the DAS test set is not hand-picked:
  A: every M>=3.0 event within 100 km of the cable (detection threshold is ~M3 at 93 km)
  B: every M>=2.0 event within 30 km
  C: the next-largest events within 30 km, until --total new files are chosen
Whether each event is actually detected is decided later by stack_checks.py.

Needs data/ridgecrest_catalog.csv (ComCat lookup of the 751 event IDs, with
distance to the cable). Files already in data/ridgecrest_north/data are skipped.
Stops if free disk space drops below 2 GB. Each file is ~110 MB.

Usage, from the top of the repo:
    python scripts/download_ridgecrest.py --total 60
Writes data/ridgecrest_selection.csv and data/ridgecrest_north/data/<id>.h5
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "ridgecrest_catalog.csv"
DATA_DIR = ROOT / "data" / "ridgecrest_north" / "data"
REPO = "AI4EPS/quakeflow_das"
MIN_FREE_GB = 2.0


def select(cat, have, total):
    new = cat[~cat["id"].isin(have)]
    a = new[(new["mag"] >= 3.0) & (new["dist_km"] <= 100)].assign(rule="A")
    rest = new[~new["id"].isin(a["id"])]
    b = rest[(rest["mag"] >= 2.0) & (rest["dist_km"] <= 30)].assign(rule="B")
    rest = rest[~rest["id"].isin(b["id"])]
    n_c = max(0, total - len(a) - len(b))
    c = (rest[rest["dist_km"] <= 30].sort_values("mag", ascending=False)
         .head(n_c).assign(rule="C"))
    return pd.concat([a, b, c]).head(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=60, help="new files to download")
    args = ap.parse_args()

    cat = pd.read_csv(CATALOG)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in DATA_DIR.glob("*.h5")}
    sel = select(cat, have, args.total)
    sel.to_csv(ROOT / "data" / "ridgecrest_selection.csv", index=False)
    print(f"selected {len(sel)} new events: "
          + ", ".join(f"{k}={v}" for k, v in sel["rule"].value_counts().sort_index().items()))

    for i, r in enumerate(sel.itertuples(), 1):
        free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
        if free_gb < MIN_FREE_GB:
            sys.exit(f"stopping: only {free_gb:.1f} GB free")
        hf_hub_download(REPO, f"ridgecrest_north/data/{r.id}.h5", repo_type="dataset",
                        local_dir=ROOT / "data")
        print(f"[{i}/{len(sel)}] {r.id} rule {r.rule}: M{r.mag:.2f}, {r.dist_km:.0f} km",
              flush=True)


if __name__ == "__main__":
    main()
