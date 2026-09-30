#!/usr/bin/env python3
"""
Figure 7 Panel A: M-gate salt bridge distances, FAD vs PCAR vs Apo, 3 replicates each
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, warnings
warnings.filterwarnings("ignore")

# Directories
FAD_DIRS = [".../rep1",
            ".../rep2",
            ".../rep3"]
APO_DIRS = [".../rep1",
            ".../rep2",
            ".../rep3"]
PCAR_DIRS = [".../rep1",
             ".../rep2",
             ".../rep3"]

# (xvg_filename, panel_title, type, highlight_as_key_panel)
SALT_BRIDGES = [
    ("dist_ASP26_LYS129.xvg",  "D26@OD ↔ K129@NZ",  "saltbridge", True),
    ("dist_GLU126_LYS238.xvg", "E126@OE ↔ K238@NZ", "saltbridge", False),
    ("dist_ASP235_LYS29.xvg",  "D235@OD ↔ K29@NZ",  "saltbridge", False),
]

FAD_COLOR  = "#C03830"
PCAR_COLOR = "black"
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
print("Loading m-gate xvg files...")
data = []
for (xvg, title, type, hi) in SALT_BRIDGES:
    print(f"\n{title} -- FAD:")
    t_f, m_f, s_f = load_replicas(FAD_DIRS, xvg)
    print(f"{title} -- PCAR:")
    t_p, m_p, s_p = load_replicas(PCAR_DIRS, xvg)
    print(f"{title} -- APO:")
    t_a, m_a, s_a = load_replicas(APO_DIRS, xvg)
    data.append((xvg, title, type, hi, t_f, m_f, s_f, t_p, m_p, s_p, t_a, m_a, s_a))


# Figure
fig = plt.figure(figsize=(15, 4.5))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

axes = []

for p_idx, (xvg, title, type, hi, t_f, m_f, s_f, t_p, m_p, s_p, t_a, m_a, s_a) in enumerate(data):
    ax = fig.add_subplot(gs[p_idx])
    axes.append(ax)

    ax.set_facecolor("#FFF8F8" if hi else "white")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

    # Apo (bottom layer)
    if t_a is not None:
        ax.fill_between(t_a, m_a - s_a, m_a + s_a,
                        color=APO_COLOR, alpha=SD_ALPHA, zorder=1)
        ax.plot(t_a, m_a, color=APO_COLOR, lw=LW,
                alpha=0.75, zorder=3, label="Apo-cBOU")

    # P-carnitine (middle layer)
    if t_p is not None:
        ax.fill_between(t_p, m_p - s_p, m_p + s_p,
                        color=PCAR_COLOR, alpha=SD_ALPHA, zorder=2)
        ax.plot(t_p, m_p, color=PCAR_COLOR, lw=LW,
                alpha=0.80, zorder=4, label="cPCar-bound")

    # FAD (top layer)
    if t_f is not None:
        ax.fill_between(t_f, m_f - s_f, m_f + s_f,
                        color=FAD_COLOR, alpha=SD_ALPHA, zorder=2)
        ax.plot(t_f, m_f, color=FAD_COLOR, lw=LW,
                alpha=0.85, zorder=4, label="cFAD-bound")

    # Salt bridge cutoff line
    ax.axhline(SB_CUT, color="dimgray", lw=0.8,
               ls="--", alpha=0.6,
               label=f"Cutoff ({SB_CUT} nm)")

    # Interaction type tag
    tag = "Salt bridge" if type == "saltbridge" else ""
    ax.text(0.03, 0.97, tag,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_TICK - 1, color="dimgray", style="italic")

    # Y limits -- tight around data (no zero baseline)
    all_v = []
    for m, s in [(m_f, s_f), (m_p, s_p), (m_a, s_a)]:
        if m is not None:
            all_v.extend((m - s).tolist())
            all_v.extend((m + s).tolist())
    if all_v:
        ymin = max(0.0, np.min(all_v) - 0.03)
        ymax = np.max(all_v) + 0.06
        ax.set_ylim(ymin, ymax)

    # Occupancy annotations
    if m_f is not None:
        ax.text(0.97, 0.97, f"cFAD ({pct_formed(m_f):.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_TICK - 0.5, color=FAD_COLOR,
                fontweight="bold")
    if m_p is not None:
        ax.text(0.97, 0.87, f"cPCar ({pct_formed(m_p):.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_TICK - 0.5, color=PCAR_COLOR,
                fontweight="bold")
    if m_a is not None:
        ax.text(0.97, 0.77, f"Apo ({pct_formed(m_a):.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_TICK - 0.5, color=APO_COLOR,
                fontweight="bold")

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL, fontweight="bold")
    if p_idx == 0:
        ax.set_ylabel("Distance (nm)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8)
    ax.tick_params(labelsize=FONT_TICK)

# Shared legend above all panels
leg_handles = [
    Line2D([0],[0], color=FAD_COLOR,  lw=1.4, label="cFAD-bound (mean)"),
    Patch(facecolor=FAD_COLOR,  alpha=0.35, edgecolor="none", label="cFAD ±SD"),
    Line2D([0],[0], color=PCAR_COLOR, lw=1.4, label="cPCar-bound (mean)"),
    Patch(facecolor=PCAR_COLOR, alpha=0.35, edgecolor="none", label="cPCar ±SD"),
    Line2D([0],[0], color=APO_COLOR,  lw=1.4, label="Apo-cBOU (mean)"),
    Patch(facecolor=APO_COLOR,  alpha=0.18, edgecolor="none", label="Apo ±SD"),
    Line2D([0],[0], color="dimgray",  lw=0.8, ls="--",
           label=f"Salt bridge cutoff ({SB_CUT} nm)"),
]
fig.legend(handles=leg_handles, fontsize=FONT_LEG,
           loc="upper center", ncol=7,
           bbox_to_anchor=(0.5, 1.06), frameon=False)

plt.suptitle("M-gate Salt Bridge Network — cFAD-bound vs cPCar-bound vs Apo c-state",
             fontsize=FONT_TITLE + 0.5, fontweight="bold", y=1.14)

plt.savefig("Fig7A_mgate.png", dpi=DPI, bbox_inches="tight",
            facecolor="white")
plt.savefig("Fig7A_mgate.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved: Fig7A_mgate.png / .pdf")
plt.show()
