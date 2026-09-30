#!/usr/bin/env python3
"""
M-state simulation stability and dynamics analysis
Carnitine-bound (black) vs Apo m-state (#2166AC)
3 replicas × 1µs each

Metrics:
  • Backbone RMSD (rmsd_backbone.xvg)
  • Ligand-vs-Backbone RMSD (rmsd_LIG_vs_Protein.xvg) 
  • Radius of Gyration (gyrate.xvg)
  • Per-residue Cα RMSF (rmsf_residue.xvg)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import gaussian_kde
from scipy.ndimage import uniform_filter1d
import os, warnings
warnings.filterwarnings("ignore")

# DIRECTORIES

CAR_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]
APO_M_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]

XVG_BACKBONE = "rmsd_backbone.xvg"
XVG_LIGAND   = "rmsd_LIG_vs_Protein.xvg"
XVG_GYRATE   = "gyrate.xvg"
XVG_RMSF     = "rmsf_residue.xvg"

# Colors
CAR_COLOR   = "#1A1A1A"
CAR_SHADE   = "#909090"
APO_COLOR   = "#2166AC"
APO_SHADE   = "#9EC8F0"

CAR_REP_COLS = ["#555555", "#333333", "#111111"]
APO_REP_COLS = ["#5B9BD5", "#2166AC", "#0D3F7A"]

CONDITIONS = {
    "CAR": {"color": CAR_COLOR, "shade": CAR_SHADE, "ls": "-",
            "label": "mCAR-bound", "dirs": CAR_DIRS,
            "rep_cols": CAR_REP_COLS},
    "APO": {"color": APO_COLOR, "shade": APO_SHADE, "ls": "--",
            "label": "Apo-mBOU",   "dirs": APO_M_DIRS,
            "rep_cols": APO_REP_COLS},
}

# Structural regions for RMSF annotation
REGIONS = [
    ("TMH1",    2,  33), ("ml12",    34,  48),
    ("mh12",    49,  69), ("TMH2",   70,  94), ("imsl23",  95, 103),
    ("TMH3",   104, 138), ("ml34",  139, 155), ("mh34",   156, 176),
    ("TMH4",   177, 203), ("imsl45",204, 212), ("TMH5",   213, 242),
    ("ml56",   243, 253), ("mh56",  254, 273), ("TMH6",   274, 299),
]

def region_color(name):
    if name.startswith("TMH"): return "#D4E6F1"
    if name in ("NTER","CTER"): return "#F2F3F4"
    return "#FAD7A0"

REGION_COLORS = [region_color(n) for n,_,_ in REGIONS]

# Key residues to annotate on RMSF
KEY_RESIDS_RMSF = {
    26: "ASP26", 29: "LYS29", 126: "GLU126", 129: "LYS129",
    183: "ARG183", 235: "ASP235", 238: "LYS238",   # m-gate
    87: "ARG87",  196: "GLU196", 199: "LYS199",
    293: "GLU293", 296: "ARG296", 7: "ASP7",         # c-gate
    18: "GLN18",  79: "PHE79",  228: "TRP228",
    280: "ARG280", 184: "GLU184",                    # binding site
}

DPI        = 1000
SMOOTH_WIN = 50

plt.rcParams.update({
    "font.size":      19,
    "axes.linewidth": 1.5,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
    "font.family":    "sans-serif",
})


# PARSERS

def parse_xvg(path, col=1):
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
    return np.array(t), np.array(v)


def load_condition(dirs, xvg_file, col=1, time_unit="ps"):
    """Load replicas. Returns (t_ns, traces)."""
    traces, t_ref = [], None
    for i, d in enumerate(dirs, 1):
        fp = os.path.join(d, xvg_file)
        if not os.path.exists(fp):
            print(f"  MISSING rep{i}: {fp}"); continue
        t, v = parse_xvg(fp, col)
        divisor = 1000.0 if time_unit == "ps" else 1.0
        t_ns = t / divisor
        if t_ref is None: t_ref = t_ns
        if len(v) != len(t_ref):
            v = np.interp(t_ref, t_ns, v)
        traces.append(v)
        print(f"  rep{i}: {len(v)} frames  "
              f"mean={v.mean():.4f} ± {v.std():.4f}")
    return t_ref, traces


def load_rmsf(dirs):
    """Load RMSF per residue. Returns (residues, traces)."""
    traces, res_ref = [], None
    for i, d in enumerate(dirs, 1):
        fp = os.path.join(d, XVG_RMSF)
        if not os.path.exists(fp):
            print(f"  MISSING rep{i}: {fp}"); continue
        r, v = parse_xvg(fp, col=1)
        if res_ref is None: res_ref = r.astype(int)
        traces.append(v)
        print(f"  rep{i}: {len(v)} residues  "
              f"mean RMSF={v.mean():.4f} nm")
    return res_ref, traces


def smooth(arr, win=SMOOTH_WIN):
    return uniform_filter1d(arr, size=win, mode="nearest")


def mean_sd(traces):
    arr = np.array(traces)
    m   = arr.mean(axis=0)
    s   = arr.std(axis=0, ddof=1) if len(traces)>1 else np.zeros_like(m)
    return m, s


# DRAWING HELPERS

def draw_ts(ax, t, mean, sd, color, shade, ls, label):
    ax.fill_between(t, mean-sd, mean+sd,
                    color=shade, alpha=0.35, zorder=1)
    ax.plot(t, mean, color=shade, lw=0.5, alpha=0.50, zorder=2)
    sm = smooth(mean)
    ax.plot(t, sm, color=color, lw=2.2, ls=ls,
            label=label, zorder=4)


def draw_kde(ax_hist, traces, color, shade, ls):
    all_v = np.concatenate(traces)
    ax_hist.hist(all_v, bins=50, orientation="horizontal",
                 density=True, color=shade, alpha=0.45,
                 edgecolor="white", linewidth=0.3, zorder=1)
    kde   = gaussian_kde(all_v, bw_method=0.15)
    yr    = np.linspace(all_v.min()-0.005, all_v.max()+0.005, 400)
    ax_hist.plot(kde(yr), yr, color=color, lw=2.2, ls=ls, zorder=3)
    ax_hist.axhline(all_v.mean(), color=color, lw=1.2,
                    ls="--", alpha=0.80, zorder=4)


def style_main(ax, xlabel="", ylabel="", xlim=None, ylim=None,
               bottom_frame=True):
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


def cond_legend(conds=("CAR","APO")):
    handles = [
        Line2D([0],[0], color=CONDITIONS[c]["color"],
               lw=2.2, ls=CONDITIONS[c]["ls"],
               label=CONDITIONS[c]["label"])
        for c in conds
    ]
    handles += [Patch(facecolor="#CCCCCC", alpha=0.5,
                      edgecolor="none", label="±SD")]
    return handles


def add_region_bands(ax, alpha=0.22, orientation="x"):
    for (name, s, e), col in zip(REGIONS, REGION_COLORS):
        if orientation == "x":
            ax.axvspan(s-0.5, e+0.5, color=col, alpha=alpha, zorder=0)
        else:
            ax.axhspan(s-0.5, e+0.5, color=col, alpha=alpha, zorder=0)


# LOAD DATA

print("="*65)
print("LOADING DATA — M-STATE SIMULATIONS")
print("="*65)

bb_data, lig_data, gyr_data, rmsf_data = {}, {}, {}, {}

for cond, cfg in CONDITIONS.items():
    print(f"\n-- {cond}: Backbone RMSD --")
    t, tr = load_condition(cfg["dirs"], XVG_BACKBONE)
    if tr: bb_data[cond] = {"t":t,"traces":tr,"mean":mean_sd(tr)[0],"sd":mean_sd(tr)[1]}

print()
for cond in ["CAR"]:   # ligand RMSD only for CAR
    print(f"-- {cond}: Ligand RMSD --")
    t, tr = load_condition(CONDITIONS[cond]["dirs"], XVG_LIGAND)
    if tr: lig_data[cond] = {"t":t,"traces":tr,"mean":mean_sd(tr)[0],"sd":mean_sd(tr)[1]}

print()
for cond, cfg in CONDITIONS.items():
    print(f"-- {cond}: Gyration --")
    t, tr = load_condition(cfg["dirs"], XVG_GYRATE)
    if tr: gyr_data[cond] = {"t":t,"traces":tr,"mean":mean_sd(tr)[0],"sd":mean_sd(tr)[1]}

print()
for cond, cfg in CONDITIONS.items():
    print(f"-- {cond}: RMSF --")
    res, tr = load_rmsf(cfg["dirs"])
    if tr: rmsf_data[cond] = {"res":res,"traces":tr,
                               "mean":mean_sd(tr)[0],"sd":mean_sd(tr)[1]}


# GENERIC TIME-SERIES FIGURE FACTORY

def ts_figure(data, conds, ylabel, title, fname):
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                             width_ratios=[4,1.2], wspace=0.06)
    ax      = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharey=ax)

    all_v = np.concatenate([
        np.concatenate(data[c]["traces"]) for c in conds if c in data])
    y_min = max(0, all_v.min()-0.02)
    y_max = all_v.max()+0.03

    for cond in conds:
        if cond not in data: continue
        cfg = CONDITIONS[cond]; d = data[cond]
        draw_ts(ax, d["t"], d["mean"], d["sd"],
                cfg["color"], cfg["shade"], cfg["ls"], cfg["label"])
        draw_kde(ax_hist, d["traces"],
                 cfg["color"], cfg["shade"], cfg["ls"])

    t0 = data[conds[0]]["t"] if conds[0] in data else np.array([0,1000])
    style_main(ax, xlabel="Time (ns)", ylabel=ylabel,
               xlim=(t0.min(),t0.max()), ylim=(y_min,y_max))
    style_hist(ax_hist)
    ax.legend(handles=cond_legend(conds), frameon=False,
              fontsize=14, loc="upper left")
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(f"{fname}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {fname}.png / .pdf")
    plt.show()


# FIGURE 1 — Backbone RMSD

print("\nFigure 1 — Backbone RMSD...")
ts_figure(
    bb_data, ["CAR","APO"],
    "Backbone RMSD (nm)",
    "Protein backbone RMSD",
    "Fig1_mstate_RMSD_Backbone"
)

# FIGURE 2 — Ligand RMSD

print("Figure 2 — Ligand RMSD...")
ts_figure(
    lig_data, ["CAR"],
    "Ligand RMSD vs backbone\n(nm)",
    "Ligand RMSD relative to protein backbone",
    "Fig2_mstate_RMSD_Ligand"
)

# FIGURE 3 — Gyration radius

print("Figure 3 — Gyration radius...")
ts_figure(
    gyr_data, ["CAR","APO"],
    "Radius of gyration (nm)",
    "Protein radius of gyration",
    "Fig3_mstate_Gyration"
)

# FIGURE 4 — RMSF (Cα per residue)

print("Figure 4 — RMSF...")

def rmsf_figure():
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor("white")

    res_range = np.arange(1, 301)

    for ax_idx, (ax, cond) in enumerate(zip(axes, ["CAR","APO"])):
        if cond not in rmsf_data:
            ax.text(0.5, 0.5, f"No data for {cond}",
                    ha="center", transform=ax.transAxes)
            continue
        cfg = CONDITIONS[cond]
        d   = rmsf_data[cond]
        res = d["res"]
        m   = d["mean"]
        s   = d["sd"]

        # Region background bands
        add_region_bands(ax, alpha=0.30)

        # RMSF per replica (thin, faded)
        for ri, tr in enumerate(d["traces"]):
            ax.plot(res, tr, color=cfg["rep_cols"][ri],
                    lw=0.6, alpha=0.35, zorder=2)

        # Mean ± SD
        ax.fill_between(res, m-s, m+s,
                        color=cfg["shade"], alpha=0.40, zorder=3)
        ax.plot(res, m, color=cfg["color"], lw=2.0,
                label=cfg["label"], zorder=5)

        # Annotate key residues
        for resid, lbl in KEY_RESIDS_RMSF.items():
            idx = np.where(res == resid)[0]
            if len(idx) == 0: continue
            i   = idx[0]
            val = m[i]
            col_ann = "#E74C3C" if resid in {87,196,199,293,296,7} else \
                      "#2980B9" if resid in {26,29,126,129,183,235,238} else \
                      "#F39C12"
            ax.scatter(resid, val, color=col_ann,
                       s=55, zorder=6, edgecolors="white",
                       linewidth=0.7)
            va = "bottom" if val > m.mean() else "top"
            offset = 0.004 if va == "bottom" else -0.004
            ax.text(resid, val+offset, lbl,
                    ha="center", va=va,
                    fontsize=6.5, fontweight="bold",
                    color=col_ann, rotation=65, zorder=7)

        # Region labels at bottom
        y_lo = ax.get_ylim()[0] if ax.get_ylim()[0] != 0.0 else 0
        for (name, s_r, e_r), col in zip(REGIONS, REGION_COLORS):
            mid = (s_r + e_r) / 2
            ax.text(mid, 0.90, name, ha="center", va="top",
                    fontsize=8.5, color="black",
                    fontweight="bold",
                    clip_on=False)

        ax.set_ylabel("Cα RMSF (nm)", fontsize=13, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", width=1.5, length=6, labelsize=11)
        for sp in ax.spines.values(): sp.set_visible(True)
        ax.set_facecolor("white"); ax.grid(False)

        panel = "A)" if ax_idx == 0 else "B)"
        ax.text(0.01, 0.98, f"{panel}  {cfg['label']}",
                transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top",
                color=cfg["color"])

        # Legend: condition + replica lines
        leg_h = [Line2D([0],[0], color=cfg["color"], lw=2.0,
                         label=f"{cfg['label']} (mean)"),
                 Patch(facecolor=cfg["shade"], alpha=0.4,
                       edgecolor="none", label="±SD")]
        ax.legend(handles=leg_h, frameon=False,
                  fontsize=8.5, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Residue Number", fontsize=13, fontweight="bold")
    axes[-1].set_xlim(1, 300)

    # Annotation legend
    ann_handles = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#E74C3C", markersize=7,
               label="C-gate residues"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#2980B9", markersize=7,
               label="M-gate residues"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#F39C12", markersize=7,
               label="Binding-site residues"),
        Patch(facecolor="#D4E6F1", alpha=0.8,
              edgecolor="none", label="TM helices"),
        Patch(facecolor="#FAD7A0", alpha=0.8,
              edgecolor="none", label="Loop / matrix helix"),
    ]
    fig.legend(handles=ann_handles, fontsize=9, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.01),
               frameon=False)

    plt.suptitle(
        "Cα RMSF",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("Fig4_mstate_RMSF.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("Fig4_mstate_RMSF.pdf",
                bbox_inches="tight", facecolor="white")
    print("Saved: Fig4_mstate_RMSF.png / .pdf")
    plt.show()

rmsf_figure()

# SUPPLEMENTARY — Per-replica grid (backbone + ligand + gyrate)

print("Supplementary — per-replica figure...")

def supp_replica_figure():
    # Rows: backbone | gyration | ligand
    # Cols: CAR rep1 | CAR rep2 | CAR rep3 | APO rep1 | APO rep2 | APO rep3
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 6, figure=fig,
                             hspace=0.45, wspace=0.28)

    metric_cfg = [
        (bb_data,  "Backbone RMSD (nm)",          ""),
        (gyr_data, "Gyration radius (nm)",         ""),
        (lig_data, "Ligand RMSD vs backbone (nm)", ""),
    ]

    for row, (data, ylabel, panel_lbl) in enumerate(metric_cfg):
        for col_block, (cond, rep_cols) in enumerate([
                ("CAR", CAR_REP_COLS),
                ("APO", APO_REP_COLS)]):
            for rep_idx in range(3):
                col = col_block*3 + rep_idx
                ax  = fig.add_subplot(gs[row, col])
                ax.set_facecolor("white"); ax.grid(False)
                ax.spines[["top","right"]].set_visible(False)
                ax.tick_params(direction="in", width=1.2, length=5,
                               labelsize=9)

                if cond not in data or \
                   rep_idx >= len(data[cond]["traces"]):
                    ax.text(0.5, 0.5, "", ha="center",
                            va="center", color="gray",
                            transform=ax.transAxes, fontsize=13)
                    ax.axis("off"); continue

                t   = data[cond]["t"]
                tr  = data[cond]["traces"][rep_idx]
                sm  = smooth(tr, win=30)
                col_c = rep_cols[rep_idx]

                ax.plot(t, tr, color=col_c, lw=0.5, alpha=0.35)
                ax.plot(t, sm, color=col_c, lw=1.8)

                if row == 0:
                    ax.set_title(
                        f"{CONDITIONS[cond]['label']}\nRep {rep_idx+1}",
                        fontsize=9, fontweight="bold",
                        color=CONDITIONS[cond]["color"])
                if col == 0 or col == 3:
                    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
                if row == 2:
                    ax.set_xlabel("Time (ns)", fontsize=12,
                                  fontweight="bold")

                ax.text(0.03, 0.96,
                        f"{panel_lbl}  mean={tr.mean():.3f} nm",
                        transform=ax.transAxes, fontsize=11,
                        va="top", color=col_c, fontweight="bold")

    plt.suptitle(
        "Per-replica trajectories",
        fontsize=13, fontweight="bold")
    plt.savefig("FigS_mstate_Replicates.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("FigS_mstate_Replicates.pdf",
                bbox_inches="tight", facecolor="white")
    print("Saved: FigS_mstate_Replicates.png / .pdf")
    plt.show()

supp_replica_figure()

# COMBINED OVERVIEW — all 3 stability metrics stacked

print("Combined overview figure...")

def combined_figure():
    fig = plt.figure(figsize=(15, 13))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             width_ratios=[4,1.2],
                             wspace=0.06, hspace=0.42)

    panel_info = [
        (bb_data,  ["CAR","APO"], "Backbone RMSD (nm)"),
        (lig_data, ["CAR"],       "Carnitine RMSD vs backbone (nm)"),
        (gyr_data, ["CAR","APO"], "Radius of gyration (nm)"),
    ]

    for row, (data, conds, ylabel) in enumerate(panel_info):
        ax      = fig.add_subplot(gs[row, 0])
        ax_hist = fig.add_subplot(gs[row, 1], sharey=ax)

        all_v = np.concatenate([
            np.concatenate(data[c]["traces"])
            for c in conds if c in data])
        y_min = max(0, all_v.min()-0.02)
        y_max = all_v.max()+0.03

        for cond in conds:
            if cond not in data: continue
            cfg = CONDITIONS[cond]; d = data[cond]
            draw_ts(ax, d["t"], d["mean"], d["sd"],
                    cfg["color"], cfg["shade"], cfg["ls"], cfg["label"])
            draw_kde(ax_hist, d["traces"],
                     cfg["color"], cfg["shade"], cfg["ls"])

        t0 = data[conds[0]]["t"] if conds[0] in data else np.array([0,1000])
        style_main(ax,
                   xlabel="Time (ns)" if row==2 else "",
                   ylabel=ylabel,
                   xlim=(t0.min(),t0.max()),
                   ylim=(y_min,y_max))
        style_hist(ax_hist)

        if row == 0:
            ax.legend(handles=cond_legend(["CAR","APO"]),
                      frameon=False, fontsize=10,
                      loc="upper left", ncol=2)
            ax_hist.set_title("Distribution\n+ KDE",
                              fontsize=9, color="dimgray", pad=4)

    plt.suptitle(
        "M-state Simulation Stability\n"
        "Backbone RMSD | Carnitine RMSD | Radius of gyration",
        fontsize=13, fontweight="bold", y=1.01)
    plt.savefig("FigC_mstate_Combined.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("FigC_mstate_Combined.pdf",
                bbox_inches="tight", facecolor="white")
    print("Saved: FigC_mstate_Combined.png / .pdf")
    plt.show()

combined_figure()

# SUMMARY CSV

with open("mstate_Summary_statistics.csv","w") as f:
    f.write("Metric,Condition,Mean,SD,Min,Max,N_replicas\n")
    for mn, data, conds in [
            ("Backbone_RMSD_nm", bb_data,  ["CAR","APO"]),
            ("Ligand_RMSD_nm",   lig_data, ["CAR"]),
            ("Gyration_nm",      gyr_data, ["CAR","APO"]),
    ]:
        for cond in conds:
            if cond not in data: continue
            av = np.concatenate(data[cond]["traces"])
            f.write(f"{mn},{cond},{av.mean():.4f},{av.std():.4f},"
                    f"{av.min():.4f},{av.max():.4f},"
                    f"{len(data[cond]['traces'])}\n")
    for cond in ["CAR","APO"]:
        if cond not in rmsf_data: continue
        av = np.concatenate(rmsf_data[cond]["traces"])
        f.write(f"RMSF_CA_nm,{cond},{av.mean():.4f},{av.std():.4f},"
                f"{av.min():.4f},{av.max():.4f},"
                f"{len(rmsf_data[cond]['traces'])}\n")

print("Saved: mstate_Summary_statistics.csv")

print("\n" + "="*65)
print("ALL DONE")
for fn in ["Fig1_mstate_RMSD_Backbone.png","Fig2_mstate_RMSD_Ligand.png",
           "Fig3_mstate_Gyration.png","Fig4_mstate_RMSF.png",
           "FigS_mstate_Replicates.png","FigC_mstate_Combined.png",
           "mstate_Summary_statistics.csv"]:
    print(f"  [{'OK  ' if os.path.exists(fn) else 'MISS'}] {fn}")
