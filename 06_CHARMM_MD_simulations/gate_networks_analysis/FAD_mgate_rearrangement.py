#!/usr/bin/env python3
"""
Figure 7 Panel B -- protein-FAD-protein bridge (m-gate rearrangement)
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

# Panel definitions
# (xvg, title, itype, use_apo, row, col)
# itype: "hbond" or "saltbridge"
# use_apo: False = FAD only (new bridge), True = compare FAD vs Apo
PANELS = [
    (
        "dist_LYS129_FADO5.xvg",
        "LYS129@NZ ↔ FAD@O5",
        "hbond", False, 0, 0,
    ),
    (
        "dist_GLU126_FADN3.xvg",
        "GLU126@OE ↔ FAD@N3",
        "hbond", False, 0, 1,
    ),
    (
        "dist_LYS129_GLU126.xvg",
        "LYS129@NZ ↔ GLU126@OE",
        "saltbridge", True, 1, 0,
    ),
    (
        "dist_ASP26_ARG280.xvg",
        "ASP26@OD ↔ ARG280@NH",
        "saltbridge", True, 1, 1,
    ),
]

FAD_COLOR  = "#C03830"
APO_COLOR  = "#2166AC"
SD_ALPHA   = 0.18
LW         = 1.0
SB_CUT     = 0.35   # nm
HB_CUT     = 0.35   # nm
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


def pct_formed(mean, cut):
    return 100.0 * np.mean(mean < cut)


# Load all data
print("Loading FAD m-gate rearrangement data...")
loaded = []
for (xvg, title, itype, use_apo, row, col) in PANELS:
    label_short = title.split('\n')[0]
    print(f"\n{label_short} -- FAD:")
    t_f, m_f, s_f = load_replicas(FAD_DIRS, xvg)
    if use_apo:
        print(f"{label_short} -- APO:")
        t_a, m_a, s_a = load_replicas(APO_DIRS, xvg)
    else:
        t_a, m_a, s_a = None, None, None
    loaded.append((xvg, title, itype, use_apo,
                   row, col, t_f, m_f, s_f, t_a, m_a, s_a))


# Figure -- 2 rows x 2 columns
fig = plt.figure(figsize=(11, 8))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

for (xvg, title, itype, use_apo,
     row, col, t_f, m_f, s_f, t_a, m_a, s_a) in loaded:

    ax = fig.add_subplot(gs[row, col])

    # Background tint -- FAD-exclusive panels get subtle gold tint
    bg = "#F8F0FF" if itype == "hbond" else "white"
    ax.set_facecolor(bg)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

    cut = HB_CUT if itype == "hbond" else SB_CUT

    # Apo (only for panels where use_apo=True)
    if use_apo and t_a is not None:
        ax.fill_between(t_a, m_a - s_a, m_a + s_a,
                        color=APO_COLOR, alpha=SD_ALPHA, zorder=1)
        ax.plot(t_a, m_a, color=APO_COLOR, lw=LW,
                alpha=0.75, zorder=3, label="Apo-cBOU")

    # FAD
    if t_f is not None:
        ax.fill_between(t_f, m_f - s_f, m_f + s_f,
                        color=FAD_COLOR, alpha=SD_ALPHA, zorder=2)
        ax.plot(t_f, m_f, color=FAD_COLOR, lw=LW,
                alpha=0.85, zorder=4, label="cFAD-bound")

    # Cutoff line
    cut_lbl = "H-bond" if itype == "hbond" else "Salt bridge"
    ax.axhline(cut, color="dimgray", lw=0.8, ls="--", alpha=0.6,
               label=f"{cut_lbl} cutoff ({cut} nm)")

    # Y limits -- tight around data
    all_v = []
    for m, s in [(m_f, s_f), (m_a, s_a)]:
        if m is not None:
            all_v.extend((m - s).tolist())
            all_v.extend((m + s).tolist())
    if all_v:
        ymin = max(0.0, np.min(all_v) - 0.03)
        ymax = np.max(all_v) + 0.06
        ax.set_ylim(ymin, ymax)

    # Occupancy annotations
    if m_f is not None:
        ax.text(0.97, 0.97,
                f"cFAD ({pct_formed(m_f, cut):.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_TICK - 0.5, color=FAD_COLOR,
                fontweight="bold")
    if use_apo and m_a is not None:
        ax.text(0.97, 0.87,
                f"Apo ({pct_formed(m_a, cut):.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_TICK - 0.5, color=APO_COLOR,
                fontweight="bold")

    # Interaction type tag
    ax.text(0.03, 0.88 if not use_apo else 0.97,
            cut_lbl,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_TICK - 1.5, color="dimgray", style="italic")

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Distance (nm)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=7)
    ax.tick_params(labelsize=FONT_TICK)

# Shared legend
leg_handles = [
    Line2D([0],[0], color=FAD_COLOR, lw=1.4, label="cFAD-bound (mean)"),
    Patch(facecolor=FAD_COLOR, alpha=0.35, edgecolor="none",
          label="cFAD ±SD"),
    Line2D([0],[0], color=APO_COLOR, lw=1.4, label="Apo-cBOU (mean)"),
    Patch(facecolor=APO_COLOR, alpha=0.18, edgecolor="none",
          label="Apo ±SD"),
    Line2D([0],[0], color="dimgray", lw=0.8, ls="--",
           label="Interaction cutoff (0.35 nm)"),
]
fig.legend(handles=leg_handles, fontsize=FONT_LEG,
           loc="upper center", ncol=5,
           bbox_to_anchor=(0.5, 1.03), frameon=False)

plt.suptitle(
    "FAD M-gate rearrangement",
    fontsize=FONT_TITLE + 0.5, fontweight="bold", y=1.07)

plt.savefig("Fig7B_FAD_mgate_rearrangement.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("Fig7B_FAD_mgate_rearrangement.pdf",
            bbox_inches="tight", facecolor="white")
print("\nSaved: Fig7B_FAD_mgate_rearrangement.png / .pdf")
plt.show()
