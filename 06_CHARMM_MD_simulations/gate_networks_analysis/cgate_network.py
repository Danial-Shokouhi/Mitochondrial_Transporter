#!/usr/bin/env python3
"""
C-gate salt bridge and brace interaction distances.
Carnitine m-state (black) vs Apo m-state (#2166AC).
2 rows x 3 columns: row 1 = main salt bridges, row 2 = side interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, warnings
warnings.filterwarnings("ignore")

# Directories
CAR_DIRS = [".../rep1",
            ".../rep2",
            ".../rep3"]
APO_DIRS = [".../rep1",
            ".../rep2",
            ".../rep3"]

# Row 1: main c-gate salt bridges | Row 2: side interactions
# (xvg_filename, panel_title, type, row)
INTERACTIONS = [
    # Row 0 -- main salt bridges
    ("dist_ARG87_GLU196.xvg",  "R87@NH ↔ E196@OE",  "saltbridge",   0),
    ("dist_LYS199_GLU293.xvg", "K199@NZ ↔ E293@OE", "saltbridge",   0),
    ("dist_ARG296_ASP7.xvg",   "R296@NH ↔ D7@OD",   "saltbridge",   0),
    # Row 1 -- side interactions
    ("dist_LYS199_GLU196.xvg", "K199@NZ ↔ E196@OE", "saltbridge",   1),
    ("dist_GLU293_TYR195.xvg", "E293@OE ↔ Y195@OH", "hbond(brace)", 1),
    ("dist_ASP7_TYR292.xvg",   "D7@OD ↔ Y292@OH",   "hbond(brace)", 1),
]

CAR_COLOR  = "#1A1A1A"   
APO_COLOR  = "#2166AC"   
SD_ALPHA   = 0.18
LW         = 1.0
SB_CUT     = 0.35   # nm -- salt bridge formation threshold
STRIDE     = 10
FONT_TITLE = 15
FONT_LABEL = 13
FONT_TICK  = 13.5
FONT_LEG   = 13.5
DPI        = 1000


# Parsers
def parse_xvg(path, stride=1):
    times, vals = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ('#', '@'):
                continue
            p = s.split()
            if len(p) < 2:
                continue
            try:
                times.append(float(p[0]))
                vals.append(float(p[1]))
            except ValueError:
                continue
    t = np.array(times)[::stride] / 1000.0   # ps -> ns
    v = np.array(vals)[::stride]
    return t, v


def load_replicas(dirs, xvg, stride=STRIDE):
    traces, t_ref = [], None
    for d in dirs:
        fp = os.path.join(d, xvg)
        if not os.path.exists(fp):
            print(f"  WARNING missing: {fp}")
            continue
        t, v = parse_xvg(fp, stride)
        if t_ref is None:
            t_ref = t
        if len(v) != len(t_ref):
            v = np.interp(t_ref, t, v)
        traces.append(v)
        print(f"  OK {fp} ({len(v)} frames)")
    if not traces:
        return None, None, None
    arr  = np.array(traces)
    mean = arr.mean(axis=0)
    sd   = arr.std(axis=0, ddof=1) if len(traces) > 1 else np.zeros_like(mean)
    return t_ref, mean, sd


def pct_formed(mean, cut=SB_CUT):
    return 100.0 * np.mean(mean < cut)


# Load data
print("Loading c-gate xvg files...")
data = []
for (xvg, title, type, row) in INTERACTIONS:
    print(f"\n{title.split(chr(10))[0]} -- CAR:")
    t_c, m_c, s_c = load_replicas(CAR_DIRS, xvg)
    print(f"{title.split(chr(10))[0]} -- APO:")
    t_a, m_a, s_a = load_replicas(APO_DIRS, xvg)
    data.append((xvg, title, type, row, t_c, m_c, s_c, t_a, m_a, s_a))


# Figure -- 2 rows x 3 columns
fig = plt.figure(figsize=(15, 9.0))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.50)

axes = []
for p_idx, (xvg, title, type, row, t_c, m_c, s_c, t_a, m_a, s_a) in enumerate(data):
    col = p_idx % 3
    ax  = fig.add_subplot(gs[row, col])
    axes.append(ax)

    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

    # Apo (bottom layer)
    if t_a is not None:
        ax.fill_between(t_a, m_a - s_a, m_a + s_a,
                        color=APO_COLOR, alpha=SD_ALPHA, zorder=1)
        ax.plot(t_a, m_a, color=APO_COLOR, lw=LW,
                alpha=0.75, zorder=3, label="Apo-mBOU")

    # Carnitine (top layer)
    if t_c is not None:
        ax.fill_between(t_c, m_c - s_c, m_c + s_c,
                        color=CAR_COLOR, alpha=SD_ALPHA, zorder=2)
        ax.plot(t_c, m_c, color=CAR_COLOR, lw=LW,
                alpha=0.85, zorder=4, label="mCAR-bound")

    # Subtle background tint by interaction type
    bg = "#F8F0FF" if type == "hbond(brace)" else "white"
    ax.set_facecolor(bg)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

    # Salt bridge cutoff line
    ax.axhline(SB_CUT, color="dimgray", lw=0.8,
               ls="--", alpha=0.6,
               label=f"Cutoff ({SB_CUT} nm)")

    # Y limits -- tight around data (no zero baseline)
    all_v = []
    for m, s in [(m_c, s_c), (m_a, s_a)]:
        if m is not None:
            all_v.extend((m - s).tolist())
            all_v.extend((m + s).tolist())
    if all_v:
        ymin = max(0.0, np.min(all_v) - 0.03)
        ymax = np.max(all_v) + 0.06
        ax.set_ylim(ymin, ymax)

    # Occupancy annotations
    if m_c is not None:
        ax.text(0.97, 0.97, f"mCAR ({pct_formed(m_c):.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_TICK - 0.5, color=CAR_COLOR,
                fontweight="bold")
    if m_a is not None:
        ax.text(0.97, 0.87, f"Apo ({pct_formed(m_a):.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_TICK - 0.5, color=APO_COLOR,
                fontweight="bold")

    # Interaction type tag
    tag = "H-bond/n(brace)" if type == "hbond(brace)" else "Salt bridge"
    ax.text(0.03, 0.97, tag,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_TICK - 1, color="dimgray", style="italic")

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Distance (nm)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8)
    ax.tick_params(labelsize=FONT_TICK)

# Shared legend above all panels
leg_handles = [
    Line2D([0],[0], color=CAR_COLOR, lw=1.4, label="mCAR-bound (mean)"),
    Patch(facecolor=CAR_COLOR, alpha=0.35, edgecolor="none", label="mCAR ±SD"),
    Line2D([0],[0], color=APO_COLOR, lw=1.4, label="Apo-mBOU (mean)"),
    Patch(facecolor=APO_COLOR, alpha=0.18, edgecolor="none", label="Apo ±SD"),
    Line2D([0],[0], color="dimgray", lw=0.8, ls="--",
           label=f"Cutoff ({SB_CUT} nm)"),
]
fig.legend(handles=leg_handles, fontsize=FONT_LEG,
           loc="upper center", ncol=5,
           bbox_to_anchor=(0.5, 1.04), frameon=False)

plt.suptitle("C-gate Interaction Network — mCAR-bound vs Apo m-state",
             fontsize=FONT_TITLE + 0.5, fontweight="bold", y=1.08)

plt.savefig("cgate_interactions.png", dpi=DPI, bbox_inches="tight",
            facecolor="white")
plt.savefig("cgate_interactions.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved: cgate_interactions.png / .pdf")
plt.show()
