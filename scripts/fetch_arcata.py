"""
Download Arcata DAS event files (AI4EPS/quakeflow_das) and crop each one to
origin-30 s .. origin+90 s, the same 120 s layout as the Ridgecrest files.

Arcata originals are 120 s or 420 s long and up to ~800 MB, with the origin
~5 min into the 420 s files. Cropping keeps disk use small. Each original is
downloaded, cropped, saved compressed, then deleted.

The cropped file keeps every attribute of the original, with begin_time and
end_time updated to the crop, plus:
    source_file       original file name
    crop_start_index  first sample kept, in the original

Usage, from the top of the repo:
    python scripts/fetch_arcata.py 20230101T183020Z 20231016T101513Z ...
Writes data/arcata/data/<name>.h5 (data/ is not in Git).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "arcata" / "data"
REPO = "AI4EPS/quakeflow_das"
BEFORE_S, AFTER_S = 30.0, 90.0


def attr_dict(dset):
    out = {}
    for k, v in dset.attrs.items():
        out[k] = v.decode() if isinstance(v, bytes) else v
    return out


def crop_one(name, tmp):
    out = OUT_DIR / f"{name}.h5"
    if out.exists():
        print(f"{name}: already cropped, skipping")
        return
    src = hf_hub_download(REPO, f"arcata/data/{name}.h5", repo_type="dataset",
                          local_dir=tmp)
    with h5py.File(src, "r") as f:
        dset = f["data"]
        a = attr_dict(dset)
        if not a.get("event_time"):
            print(f"{name}: no event_time in the file, skipping")
            return
        dt = float(a["dt_s"])
        begin = pd.Timestamp(a["begin_time"])
        origin_s = (pd.Timestamp(a["event_time"]) - begin).total_seconds()
        nt = dset.shape[1]
        i0 = max(0, int(round((origin_s - BEFORE_S) / dt)))
        i1 = min(nt, int(round((origin_s + AFTER_S) / dt)))
        x = dset[:, i0:i1]
    a["begin_time"] = (begin + pd.Timedelta(seconds=i0 * dt)).isoformat()
    a["end_time"] = (begin + pd.Timedelta(seconds=(i1 - 1) * dt)).isoformat()
    a["source_file"] = f"{name}.h5"
    a["crop_start_index"] = i0
    with h5py.File(out, "w") as g:
        d = g.create_dataset("data", data=x.astype(np.float32), compression="gzip",
                             compression_opts=4, chunks=(min(64, x.shape[0]), min(4096, x.shape[1])))
        for k, v in a.items():
            d.attrs[k] = v
    Path(src).unlink()
    print(f"{name}: {x.shape[0]} ch, {x.shape[1] * dt:.0f} s kept "
          f"(origin at {origin_s - i0 * dt:.1f} s), {out.stat().st_size / 1e6:.0f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="file names without .h5, e.g. 20230101T183020Z")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="arcata_"))
    try:
        for n in args.names:
            crop_one(n.replace(".h5", ""), tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
