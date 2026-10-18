"""
Download the Monterey Bay DAS earthquake dataset into data/monterey_bay/.

33 earthquake waterfalls, about 137 MB in size (each), so 4.5 GB total. This is the smallest of
the three DAS datasets available to us (for now). 

Source: https://huggingface.co/datasets/AI4EPS/quakeflow_das (monterey_bay/data)

Usage, from the top of the repo:
    python scripts/download_monterey.py              # all 33 files
    python scripts/download_monterey.py --limit 3    # just the first 3 to try it out
    python scripts/download_monterey.py --list       # show what would be downloaded

Needs: pip install huggingface_hub
if re-running: files already present are skipped.
"""

import argparse
from pathlib import Path

REPO_ID = "AI4EPS/quakeflow_das"
SUBDIR = "monterey_bay/data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "monterey_bay"

FILES = [
    "20220730T233258Z.h5", "20220802T054858Z.h5", "20220808T174058Z.h5",
    "20220809T005458Z.h5", "20220809T080958Z.h5", "20220820T225358Z.h5",
    "20220828T062358Z.h5", "20220828T140758Z.h5", "20230407T074209Z.h5",
    "20230505T133353Z.h5", "20230714T024558Z.h5", "20230801T023158Z.h5",
    "20231028T013810Z.h5", "20231029T061810Z.h5", "20231103T154210Z.h5",
    "20231109T130910Z.h5", "20231109T132510Z.h5", "20231110T105410Z.h5",
    "20231116T130210Z.h5", "20231121T195910Z.h5", "20231124T104510Z.h5",
    "20231130T194610Z.h5", "20231205T154310Z.h5", "20240123T123210Z.h5",
    "20240204T004410Z.h5", "20240301T070310Z.h5", "20240428T100822Z.h5",
    "20240712T042952Z.h5", "20240913T221528Z.h5", "20241122T195328Z.h5",
    "20241205T172428Z.h5", "20241206T180928Z.h5", "20241217T092528Z.h5",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only download the first N files")
    ap.add_argument("--list", action="store_true",
                    help="print the file list and exit")
    args = ap.parse_args()

    wanted = FILES[: args.limit] if args.limit else FILES

    if args.list:
        print(f"{len(wanted)} files from {REPO_ID}/{SUBDIR}:")
        for name in wanted:
            print("  ", name)
        print(f"\nwould go to: {OUT_DIR}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Run:  python -m pip install huggingface_hub"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(wanted)} files to {OUT_DIR}\n")

    done = skipped = failed = 0
    for i, name in enumerate(wanted, 1):
        target = OUT_DIR / name
        if target.exists():
            print(f"[{i:2d}/{len(wanted)}] skip     {name} (already here)")
            skipped += 1
            continue
        print(f"[{i:2d}/{len(wanted)}] download {name} ...", flush=True)
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{SUBDIR}/{name}",
                repo_type="dataset",
                local_dir=OUT_DIR.parent, #keeps the monterey_bay/data layout
            )
            print(f"  -> {Path(path).stat().st_size / 1e6:.0f} MB")
            done += 1
        except Exception as e:
            print(f" FAILED: {e}")
            failed += 1

    print(f"\ndownloaded {done}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
