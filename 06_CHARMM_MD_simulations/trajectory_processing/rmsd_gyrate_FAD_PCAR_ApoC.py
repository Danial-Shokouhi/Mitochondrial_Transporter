#!/usr/bin/env python3
"""
Protein backbone RMSD | Ligand-vs-backbone RMSD | Radius of gyration
FAD (#C03830) | PCAR (black) | Apo (#2166AC)
3 replicas x 1us each -- mean +/- SD + KDE distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from scipy.ndimage import uniform_filter1d
import os, warnings
warnings.filterwarnings("ignore")

# Directories
FAD_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]
PCAR_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]
APO_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]

XVG_BACKBONE = "rmsd_backbone.xvg"
XVG_LIGAND   = "rmsd_LIG_vs_Protein.xvg"
XVG_GYRATE   = "gyrate.xvg"

CONDITIONS = {
    "FAD":  {"color": "#C03830", "ls": "-",  "label": "cFAD-bound",
             "shade": "#F0A09D", "dirs": FAD_DIRS},
    "PCAR": {"color": "#1A1A1A", "ls": "-", "label": "cPCar-bound",
             "shade": "#A0A0A0", "dirs": PCAR_DIRS},
    "APO":  {"color": "#2166AC", "ls": "--",  "label": "Apo-cBOU",
             "shade": "#9EC8F0", "dirs": APO_DIRS},
}
REP_COLORS = {
    1: {"FAD":"#E8534A", "PCAR":"#555555", "APO":"#5B9BD5"},
    2: {"FAD":"#C03830", "PCAR":"#333333", "APO":"#2166AC"},
    3: {"FAD":"#8B1A13", "PCAR":"#111111", "APO":"#0D3F7A"},
}

DPI        = 1000
SMOOTH_WIN = 50
STRIDE     = 1

plt.rcParams.update({
    "font.size":      19,
    "axes.linewidth": 1.5,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
    "font.family":    "sans-serif",
})


# Parsers
def parse_xvg(path, col=1, stride=STRIDE):
    """Read GROMACS XVG file. Returns (time_ns, values)."""
    t, v = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ('@','#','&'): continue
            p = s.split()
            if len(p) <= col: continue
            try:
                t.append(float(p[0]))
                v.append(float(p[col]))
            except ValueError: continue
    t = np.array(t)[::stride] / 1000.0   # ps -> ns
    v = np.array(v)[::stride]
    return t, v


def load_condition(dirs, xvg_file, col=1):
    """Load all replicas for one condition. Returns (t_ref, traces list)."""
    traces, t_ref = [], None
    for i, d in enumerate(dirs, 1):
        fp = os.path.join(d, xvg_file)
        if not os.path.exists(fp):
            print(f"  MISSING: {fp}")
            continue
        t, v = parse_xvg(fp, col)
        if t_ref is None: t_ref = t
        if len(v) != len(t_ref):
            v = np.interp(t_ref, t, v)
        traces.append(v)
        print(f"  rep{i}: {len(v)} frames  mean={v.mean():.4f}  "
              f"std={v.std():.4f}  [{fp.split('/')[-2]}]")
    return t_ref, traces


def stats(traces):
    arr  = np.array(traces)
    mean = arr.mean(axis=0)
    sd   = arr.std(axis=0, ddof=1) if len(traces) > 1 else np.zeros_like(mean)
    return mean, sd


def smooth(arr, win=SMOOTH_WIN):
    return uniform_filter1d(arr, size=win, mode="nearest")


# Shared panel drawing
def draw_timeseries(ax, t, mean, sd, color, shade, ls, label, win=SMOOTH_WIN):
    """Draw mean +/- SD with a smoothed mean line."""
    ax.fill_between(t, mean-sd, mean+sd, color=shade, alpha=0.35, zorder=1)
    ax.plot(t, mean, color=shade, lw=0.6, alpha=0.5, zorder=2)
    sm = smooth(mean, win)
    ax.plot(t, sm, color=color, lw=2.0, ls=ls, label=label, zorder=4)


def draw_distribution(ax_hist, traces, color, shade, ls):
    """Draw KDE + histogram sideways."""
    all_vals = np.concatenate(traces)
    ax_hist.hist(all_vals, bins=50, orientation="horizontal",
                 density=True, color=shade, alpha=0.50,
                 edgecolor="white", linewidth=0.3, zorder=1)
    kde = gaussian_kde(all_vals, bw_method=0.15)
    y_range = np.linspace(all_vals.min() - 0.01,
                           all_vals.max() + 0.01, 300)
    ax_hist.plot(kde(y_range), y_range,
                 color=color, lw=2.0, ls=ls, zorder=3)
    ax_hist.axhline(all_vals.mean(), color=color,
                    lw=1.2, ls="--", alpha=0.80, zorder=4)


def style_ax(ax, xlabel="", ylabel="", xlim=None, ylim=None):
    ax.set_xlabel(xlabel, fontsize=19, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=19, fontweight="bold")
    ax.tick_params(direction="in", width=1.5, length=6, labelsize=16)
    for sp in ax.spines.values(): sp.set_visible(True)
    ax.set_facecolor("white"); ax.grid(False)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)


def style_hist(ax_hist):
    ax_hist.tick_params(left=False, labelleft=False,
                         bottom=False, labelbottom=False)
    for sp in ax_hist.spines.values(): sp.set_visible(True)
    ax_hist.set_facecolor("#FAFAFA"); ax_hist.grid(False)


# Load all data
print("="*65)
print("Loading RMSD backbone data...")
bb_data = {}
for cond, cfg in CONDITIONS.items():
    print(f"\n-- {cond} --")
    t, traces = load_condition(cfg["dirs"], XVG_BACKBONE)
    if traces: bb_data[cond] = {"t": t, "traces": traces,
                                  "mean": stats(traces)[0],
                                  "sd":   stats(traces)[1]}

print("\nLoading ligand RMSD data...")
lig_data = {}
for cond in ["FAD", "PCAR"]:
    print(f"\n-- {cond} --")
    t, traces = load_condition(CONDITIONS[cond]["dirs"], XVG_LIGAND)
    if traces: lig_data[cond] = {"t": t, "traces": traces,
                                   "mean": stats(traces)[0],
                                   "sd":   stats(traces)[1]}

print("\nLoading gyration radius data...")
gyr_data = {}
for cond, cfg in CONDITIONS.items():
    print(f"\n-- {cond} --")
    t, traces = load_condition(cfg["dirs"], XVG_GYRATE)
    if traces: gyr_data[cond] = {"t": t, "traces": traces,
                                   "mean": stats(traces)[0],
                                   "sd":   stats(traces)[1]}


# Figure factory (used for Fig1, Fig2, Fig3)
def make_figure(data, ylabel, title, fname, conds, show_legend=True):
    """Generic time-series + distribution figure for any metric."""
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                             width_ratios=[4, 1.2], wspace=0.06)
    ax      = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharey=ax)

    all_vals = np.concatenate([
        np.concatenate(data[c]["traces"])
        for c in conds if c in data])
    y_min = max(0, all_vals.min() - 0.02)
    y_max = all_vals.max() + 0.03

    for cond in conds:
        if cond not in data: continue
        cfg = CONDITIONS[cond]
        d   = data[cond]
        draw_timeseries(ax, d["t"], d["mean"], d["sd"],
                        cfg["color"], cfg["shade"], cfg["ls"], cfg["label"])
        draw_distribution(ax_hist, d["traces"],
                          cfg["color"], cfg["shade"], cfg["ls"])

    t_all = data[conds[0]]["t"] if conds[0] in data else np.array([0,1000])
    style_ax(ax, xlabel="Time (ns)", ylabel=ylabel,
             xlim=(t_all.min(), t_all.max()),
             ylim=(y_min, y_max))
    style_hist(ax_hist)

    if show_legend:
        leg_h = [Line2D([0],[0], color=CONDITIONS[c]["color"],
                         lw=2.0, ls=CONDITIONS[c]["ls"],
                         label=CONDITIONS[c]["label"])
                 for c in conds if c in data]
        leg_h += [Patch(facecolor="#CCCCCC", alpha=0.5,
                        edgecolor="none", label="+/-SD")]
        ax.legend(handles=leg_h, frameon=False,
                  fontsize=14, loc="upper left")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.savefig(f"{fname}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {fname}.png / .pdf")
    plt.show()


print("\nGenerating Figure 1 -- Backbone RMSD...")
make_figure(
    bb_data,
    ylabel="Backbone RMSD (nm)",
    title="Protein backbone RMSD",
    fname="Fig1_RMSD_Backbone",
    conds=["FAD","PCAR","APO"],
)

print("\nGenerating Figure 2 -- Ligand RMSD...")
make_figure(
    lig_data,
    ylabel="Ligand RMSD vs backbone (nm)",
    title="Ligand RMSD relative to protein backbone",
    fname="Fig2_RMSD_Ligand",
    conds=["FAD","PCAR"],
)

print("\nGenerating Figure 3 -- Radius of Gyration...")
make_figure(
    gyr_data,
    ylabel="Radius of gyration (nm)",
    title="Protein radius of gyration",
    fname="Fig3_Gyration",
    conds=["FAD","PCAR","APO"],
)


# Supplementary -- per-replica panels (3x3 grid)
print("\nGenerating Supplementary -- per-replica plots...")

def make_replica_figure(bb, lig, gyr):
    """3-row x 3-column: FAD | PCAR | APO per replica."""
    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.30)

    metrics = [
        (bb,  "Backbone RMSD (nm)",         ["FAD","PCAR","APO"]),
        (gyr, "Radius of gyration (nm)",     ["FAD","PCAR","APO"]),
        (lig, "Ligand RMSD vs backbone (nm)",["FAD","PCAR",None]),
    ]
    row_labels = ["Backbone RMSD",
                  "Gyration radius",
                  "Ligand RMSD"]

    for row, (data, ylabel, cond_order) in enumerate(metrics):
        for col, cond in enumerate(cond_order):
            ax = fig.add_subplot(gs[row, col])
            if cond is None or cond not in data:
                ax.text(0.5, 0.5, "",
                        ha="center", va="center",
                        fontsize=14, color="gray",
                        transform=ax.transAxes)
                ax.axis("off"); continue

            cfg    = CONDITIONS[cond]
            traces = data[cond]["traces"]
            t      = data[cond]["t"]

            for ri, trace in enumerate(traces):
                rep_col = REP_COLORS[ri+1][cond]
                sm      = smooth(trace, win=30)
                ax.plot(t, trace, color=rep_col, lw=0.5, alpha=0.35)
                ax.plot(t, sm,    color=rep_col, lw=1.6,
                        label=f"Rep {ri+1}")

            style_ax(ax, xlabel="Time (ns)", ylabel=ylabel if col==0 else "",
                     xlim=(t.min(), t.max()))

            if row == 0:
                ax.set_title(cfg["label"], fontsize=16,
                             fontweight="bold", color=cfg["color"])
            ax.legend(frameon=False, fontsize=11, loc="upper left")
            ax.text(0.02, 0.97, row_labels[row],
                    transform=ax.transAxes, fontsize=15,
                    fontweight="bold", va="top", color="#555555")

    plt.suptitle(
        "Per-replica trajectories RMSD & gyration radius",
        fontsize=13, fontweight="bold")
    plt.savefig("FigS_RMSD_Replicates.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("FigS_RMSD_Replicates.pdf",
                bbox_inches="tight", facecolor="white")
    print("Saved: FigS_RMSD_Replicates.png / .pdf")
    plt.show()


make_replica_figure(bb_data, lig_data, gyr_data)


# Combined overview figure (all 3 metrics stacked)
print("\nGenerating combined overview figure...")

fig_all = plt.figure(figsize=(15, 13))
fig_all.patch.set_facecolor("white")
gs_all = gridspec.GridSpec(
    3, 2, figure=fig_all,
    width_ratios=[4, 1.2], wspace=0.06, hspace=0.46)

panel_cfg = [
    (bb_data,  ["FAD","PCAR","APO"], "Backbone RMSD (nm)"),
    (lig_data, ["FAD","PCAR"],       "Ligand RMSD vs backbone (nm)"),
    (gyr_data, ["FAD","PCAR","APO"], "Radius of gyration (nm)"),
]

for row, (data, conds, ylabel) in enumerate(panel_cfg):
    ax      = fig_all.add_subplot(gs_all[row, 0])
    ax_hist = fig_all.add_subplot(gs_all[row, 1], sharey=ax)

    all_v = np.concatenate([np.concatenate(data[c]["traces"])
                             for c in conds if c in data])
    y_min = max(0, all_v.min() - 0.02)
    y_max = all_v.max() + 0.03

    for cond in conds:
        if cond not in data: continue
        cfg = CONDITIONS[cond]
        d   = data[cond]
        draw_timeseries(ax, d["t"], d["mean"], d["sd"],
                        cfg["color"], cfg["shade"], cfg["ls"], cfg["label"])
        draw_distribution(ax_hist, d["traces"],
                          cfg["color"], cfg["shade"], cfg["ls"])

    t_ref = data[conds[0]]["t"] if conds[0] in data else np.array([0,1000])
    style_ax(ax, xlabel="Time (ns)" if row==2 else "",
             ylabel=ylabel,
             xlim=(t_ref.min(), t_ref.max()),
             ylim=(y_min, y_max))
    style_hist(ax_hist)

    if row == 0:
        leg_h = [Line2D([0],[0], color=CONDITIONS[c]["color"],
                         lw=2.0, ls=CONDITIONS[c]["ls"],
                         label=CONDITIONS[c]["label"])
                 for c in ["FAD","PCAR","APO"]]
        leg_h += [Patch(facecolor="#CCCCCC", alpha=0.5,
                        edgecolor="none", label="+/-SD (all 3 replicas)")]
        ax.legend(handles=leg_h, frameon=False,
                  fontsize=13, loc="upper left", ncol=2)

    if row == 0:
        ax_hist.set_title("Distribution\n+ KDE",
                          fontsize=11, color="dimgray", pad=4)

plt.suptitle(
    "Simulation stability\n"
    "Backbone RMSD | Ligand RMSD | Radius of Gyration",
    fontsize=15, fontweight="bold", y=1.01)

plt.savefig("Fig_RMSD_Gyration_Combined.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("Fig_RMSD_Gyration_Combined.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: Fig_RMSD_Gyration_Combined.png / .pdf")
plt.show()


# Summary CSV
print("\nWriting summary statistics...")
with open("Summary_statistics.csv", "w") as f:
    f.write("Metric,Condition,Mean,SD,Min,Max,N_replicas\n")
    for metric_name, data, conds in [
            ("Backbone_RMSD_nm", bb_data,  ["FAD","PCAR","APO"]),
            ("Ligand_RMSD_nm",   lig_data, ["FAD","PCAR"]),
            ("Gyration_nm",      gyr_data, ["FAD","PCAR","APO"]),
    ]:
        for cond in conds:
            if cond not in data: continue
            all_v = np.concatenate(data[cond]["traces"])
            f.write(f"{metric_name},{cond},"
                    f"{all_v.mean():.4f},{all_v.std():.4f},"
                    f"{all_v.min():.4f},{all_v.max():.4f},"
                    f"{len(data[cond]['traces'])}\n")

print("Saved: Summary_statistics.csv")

print("\n" + "="*65)
print("ALL DONE")
for fn in ["Fig1_RMSD_Backbone.png", "Fig2_RMSD_Ligand.png",
           "Fig3_Gyration.png", "FigS_RMSD_Replicates.png",
           "Fig_RMSD_Gyration_Combined.png", "Summary_statistics.csv"]:
    status = "OK  " if os.path.exists(fn) else "MISS"
    print(f"  [{status}] {fn}")
