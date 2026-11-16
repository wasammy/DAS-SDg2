"""
Presentation figure: envelope stacks for the six Ridgecrest events with a clear
arrival, 2 x 3 panels, predicted P marked.

Usage, from the top of the repo:
    python scripts/fig_ridgecrest_detection.py
Writes assets/fig_ridgecrest_detection.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.das_io import load_das_event                    # noqa: E402
from stack_checks import envelope_stack, p_from_metadata  # noqa: E402

DATA_DIR = ROOT / "data" / "ridgecrest_north" / "data"
EVENTS = ["ci39493944", "ci39494008", "ci39490952",      # M5.8, 4.6, 4.7 at 93 km
          "ci39491208", "ci39492720", "ci38595986"]      # M3.8 at 93 km, M2.9-3.0 close

LINE = "#2a78d6"
P_COLOR = "#52514e"
INK = "#0b0b0b"
MUTED = "#52514e"


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#bdbcb8",
                         "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    panels = []
    for ev in EVENTS:
        data, meta = load_das_event(DATA_DIR / f"{ev}.h5")
        fs = meta["fs"]
        t, env = envelope_stack(data, fs)
        info = p_from_metadata(meta)
        p = info["p_s"]
        # background = the 10 s just before P (an earlier event can sit at the start)
        base = np.median(env[(t >= p - 12) & (t < p - 2)])
        y = env / base
        # zero-filled data gap (ci39494008 has no data after ~70 s): hide it,
        # plus 1 s before it where the filter rings on the step to zero
        zero = (data == 0).mean(axis=0) > 0.9
        zero[: int(fs)] = False                    # ignore the first second
        gap_start = np.argmax(zero) / fs if zero.any() else None
        if gap_start is not None:
            y = np.where(t >= gap_start - 1.0, np.nan, y)
        panels.append((meta["magnitude"], info["dist_km"], p, t, y, gap_start))
    panels.sort(key=lambda q: -q[0])

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for ax, (mag, km, p, t, y, gap_start) in zip(axs.ravel(), panels):
        ax.semilogy(t, y, color=LINE, lw=1.0)
        if gap_start is not None:
            ax.axvspan(gap_start, 120, color="#ecebe7", lw=0)
            ax.text((gap_start + 120) / 2, 0.5, "no data", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", color=MUTED, fontsize=9)
        ax.axvline(p, color=P_COLOR, ls="--", lw=1.0)
        ax.set_title(f"M{mag:.1f}, {km:.0f} km", color=INK, fontsize=11, loc="left")
        ax.grid(True, which="major", color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, 120)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
    axs[0, 0].text(axs[0, 0].get_lines()[1].get_xdata()[0] + 2, 0.92, "predicted P",
                   transform=axs[0, 0].get_xaxis_transform(), color=P_COLOR, fontsize=9,
                   va="top")
    for ax in axs[1]:
        ax.set_xlabel("Time from window start (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Energy / pre-P background")
    fig.suptitle("Ridgecrest DAS, stacked 2 to 40 Hz energy for six earthquakes",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "assets" / "fig_ridgecrest_detection.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
