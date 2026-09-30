#!/usr/bin/env python3
"""
CAR-bound in mBOU polar interaction distance profiles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os, sys, warnings
warnings.filterwarnings("ignore")

# DIRECTORIES

CAR_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]

CUTOFF     = 0.35   # nm
DPI        = 1000
FONT_TITLE = 18
FONT_LABEL = 16
FONT_TICK  = 13
FONT_LEG   = 13

CAR_COLOR = "#1A1A1A"
CAR_SD    = "#909090"

# INTERACTION DEFINITIONS  (row, col position in 2×3 grid)

INTERACTIONS = [
    # Row 0 — main
    {
        "xvg":    "mindist_ARG183_NH_CAR_O.xvg",
        "title":  "ARG183",
        "legend": "CAR@O3↔R183@NH",
        "type":   "Salt bridge",
        "resid":  183,
        "row": 0, "col": 0,
    },
    {
        "xvg":    "mindist_GLN18_NE_CAR_O.xvg",
        "title":  "GLN18",
        "legend": "CAR@O1↔Q18@NE",
        "type":   "H-bond",
        "resid":  18,
        "row": 0, "col": 1,
    },
    {
        "xvg":    "mindist_ASN285_OD_ND_CAR_O.xvg",
        "title":  "ASN285",
        "legend": "CAR@O1↔N285@ND",
        "type":   "H-bond",
        "resid":  285,
        "row": 0, "col": 2,
    },
    {
        "xvg":    "mindist_TRP228_NE_CAR_O.xvg",
        "title":  "TRP228",
        "legend": "CAR@O1/2↔W228@NE",
        "type":   "H-bond",
        "resid":  228,
        "row": 1, "col": 0,
    },
]

# PARSERS

def parse_xvg(path, stride=10):
    t, d = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ('@','#','&'): continue
            p = s.split()
            if len(p) < 2: continue
            try:
                t.append(float(p[0]))
                d.append(float(p[1]))
            except ValueError: continue
    t = np.array(t)[::stride] / 1000.0
    d = np.array(d)[::stride]
    return t, d


def load_interaction(dirs, xvg_name, stride=10):
    traces, t_ref = [], None
    for d in dirs:
        fp = os.path.join(d, xvg_name)
        if not os.path.exists(fp):
            print(f"  MISSING: {fp}"); continue
        t, v = parse_xvg(fp, stride)
        if t_ref is None: t_ref = t
        if len(v) != len(t_ref):
            v = np.interp(t_ref, t, v)
        traces.append(v)
    if not traces: return None, None, None
    arr = np.array(traces)
    return t_ref, arr.mean(axis=0), arr.std(axis=0, ddof=1) if len(traces)>1 else np.zeros_like(arr[0])


def occupancy(mean_trace, cutoff=CUTOFF):
    return 100.0 * np.mean(mean_trace < cutoff) if mean_trace is not None else 0.0


# LOAD DATA

print("="*65)
print("mCAR INTERACTOME — POLAR INTERACTIONS")
print("="*65)
data = {}
for inter in INTERACTIONS:
    print(f"\n{inter['title']}:")
    t, m, s = load_interaction(CAR_DIRS, inter["xvg"])
    if m is not None:
        data[inter["xvg"]] = {"t":t, "mean":m, "sd":s}
        print(f"  occ={occupancy(m):.1f}%  mean={m.mean():.3f} nm")


# FIGURE — 2 rows × 3 columns

fig = plt.figure(figsize=(15, 9.0))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         hspace=0.55, wspace=0.32,
                         top=0.88, bottom=0.06)

for i, inter in enumerate(INTERACTIONS):
    row, col = i // 2, i % 2
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor("white")
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

    if inter["xvg"] not in data:
        ax.text(0.5, 0.5, "No data", ha="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(inter["title"], fontsize=FONT_TITLE+2.5,
                     fontweight="bold")
        continue

    d = data[inter["xvg"]]
    t, m, s = d["t"], d["mean"], d["sd"]

    ax.fill_between(t, m-s, m+s, color=CAR_SD, alpha=0.38, zorder=1)
    occ = occupancy(m)
    line, = ax.plot(t, m, color=CAR_COLOR, lw=1.4,
                    label=f"{inter['legend']} ({occ:.0f}%)", zorder=3)

    # Cutoff
    ax.axhline(CUTOFF, color="gray", lw=0.8, ls="--", alpha=0.55)

    # Y limits
    all_v = (m-s).tolist() + (m+s).tolist()
    ax.set_ylim(max(0.0, np.min(all_v)-0.02), np.max(all_v)+0.04)

    # Interaction type tag
    type_col = "black" if inter["type"]=="Salt bridge" else "black"
    ax.text(0.03, 0.97, inter["type"],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_TICK+1.5, color=type_col,
            fontweight="bold", alpha=0.55, style="italic")

    ax.set_xlim(0, 1000)
    ax.tick_params(labelsize=FONT_TICK)
    if row == 1:
        ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Distance (nm)", fontsize=FONT_LABEL, fontweight="bold")

    ax.set_title(inter["title"], fontsize=FONT_TITLE+1,
                 fontweight="bold", pad=6)
    ax.legend(handles=[line], fontsize=13, frameon=False, loc="upper right")
# Shared figure legend at top
fig_handles = [
    Line2D([0],[0], color=CAR_COLOR, lw=2.0, label="mCAR-bound (mean)"),
    mpatches.Patch(facecolor=CAR_SD, edgecolor="none",
                   alpha=0.5, label="mCAR ±SD"),
    Line2D([0],[0], color="gray", lw=0.9, ls="--",
           label=f"Cutoff {CUTOFF} nm"),
]
fig.legend(handles=fig_handles, fontsize=FONT_LEG+2.0,
           loc="upper center", ncol=5,
           bbox_to_anchor=(0.5, 0.97), frameon=False, columnspacing=1.2)

plt.suptitle(
    "mCAR-bound polar interaction profiles",
    fontsize=FONT_TITLE+1, fontweight="bold", y=1.01)

plt.savefig("mCAR_interactome_timeseries.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("mCAR_interactome_timeseries.pdf",
            bbox_inches="tight", facecolor="white")
print("\nSaved: mCAR_interactome_timeseries.png / .pdf")
plt.show()
