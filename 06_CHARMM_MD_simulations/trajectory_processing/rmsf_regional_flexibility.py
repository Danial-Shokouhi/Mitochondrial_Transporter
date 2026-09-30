#!/usr/bin/env python3
"""
  Panel A: Per-region RMSF distribution (violin + mean marker)
  Panel B: Δ-RMSF (ligand-bound minus Apo) per structural region

Colors: FAD=#C03830 | PCAR=#1A1A1A | Apo-c=#2166AC
        CAR=#1A1A1A | Apo-m=#63B5E5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os, warnings
warnings.filterwarnings("ignore")

# DIRECTORIES

CONDS = {
    # -- c-state
    "FAD": {
        "dirs": [".../rep1",
                 ".../rep2",
                 ".../rep3"],
        "label": "cFAD-bound", "color": "#C03830", "state": "c",
    },
    "PCAR": {
        "dirs": [".../rep1",
                 ".../rep2",
                 ".../rep3"],
        "label": "cPCar-bound", "color": "#1A1A1A", "state": "c",
    },
    "Apo_c": {
        "dirs": [".../rep1",
                 ".../rep2",
                 ".../rep3"],
        "label": "Apo-cBOU", "color": "#2166AC", "state": "c",
    },
    # -- m-state
    "CAR": {
        "dirs": [".../rep1",
                 ".../rep2",
                 ".../rep3"],
        "label": "mCAR-bound", "color": "#1A1A1A", "state": "m",
    },
    "Apo_m": {
        "dirs": [".../rep1",
                 ".../rep2",
                 ".../rep3"],
        "label": "Apo-mBOU", "color": "#63B5E5", "state": "m",
    },
}

XVG_RMSF = "rmsf_residue.xvg"

# -- Structural regions
REGIONS = [
    ("NTER",    1,   1),
    ("TMH1",    2,  33),
    ("ml12",   34,  48),
    ("mh12",   49,  69),
    ("TMH2",   70,  94),
    ("imsl23", 95, 103),
    ("TMH3",  104, 138),
    ("ml34",  139, 155),
    ("mh34",  156, 176),
    ("TMH4",  177, 203),
    ("imsl45",204, 212),
    ("TMH5",  213, 242),
    ("ml56",  243, 253),
    ("mh56",  254, 273),
    ("TMH6",  274, 299),
    ("CTER",  300, 300),
]

def region_type(name):
    if name.startswith("TMH"):   return "tmh"
    if name in ("NTER","CTER"):  return "ter"
    return "loop"

TYPE_COLORS = {
    "tmh":  "#D4E6F1",
    "loop": "#FAD7A0",
    "ter":  "#F2F3F4",
}

DPI        = 1000
FONT_TITLE = 16
FONT_LABEL = 15
FONT_TICK  = 15
NM_TO_A = 10.0


# PARSERS

def parse_rmsf(path):
    """Return (residues_array, rmsf_nm_array)."""
    res, rmsf = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ('@','#','&'): continue
            p = s.split()
            if len(p) < 2: continue
            try:
                res.append(int(float(p[0])))
                rmsf.append(float(p[1]))
            except ValueError: continue
    return np.array(res), np.array(rmsf) * NM_TO_A 


def load_condition(cond_name, cfg):
    """
    Returns:
      per_rep : list of (res_arr, rmsf_arr) per replica
      reg_vals: {region_name: [rmsf values from all residues × all reps]}
      reg_mean: {region_name: mean per rep → shape (n_reps,)}
    """
    per_rep  = []
    reg_vals = {name: [] for name, _, _ in REGIONS}
    reg_means= {name: [] for name, _, _ in REGIONS}

    for d in cfg["dirs"]:
        fp = os.path.join(d, XVG_RMSF)
        if not os.path.exists(fp):
            print(f"  MISSING: {fp}"); continue
        res_arr, rmsf_arr = parse_rmsf(fp)
        per_rep.append((res_arr, rmsf_arr))
        print(f"  {cond_name}: {len(res_arr)} residues "
              f"(mean RMSF={rmsf_arr.mean():.3f} Å) [{d.split('/')[-1]}]")

        for name, r_start, r_end in REGIONS:
            mask = (res_arr >= r_start) & (res_arr <= r_end)
            vals = rmsf_arr[mask]
            reg_vals[name].extend(vals.tolist())
            if vals.size > 0:
                reg_means[name].append(float(vals.mean()))

    return per_rep, reg_vals, reg_means


# LOAD ALL DATA

print("="*65)
print("LOADING RMSF DATA")
print("="*65)

data = {}
for cond_name, cfg in CONDS.items():
    print(f"\n-- {cfg['label']} --")
    per_rep, reg_vals, reg_means = load_condition(cond_name, cfg)
    data[cond_name] = {
        "per_rep":   per_rep,
        "reg_vals":  reg_vals,
        "reg_means": reg_means,
        "cfg":       cfg,
    }


# FIGURE FACTORY

def make_figure(state, lig_conds, apo_cond, fname):
    """
    state      : "c" or "m"
    lig_conds  : list of condition keys for ligand-bound (e.g. ["FAD","PCAR"])
    apo_cond   : condition key for apo reference (e.g. "Apo_c")
    fname      : output filename stem
    """
    all_conds = lig_conds + [apo_cond]
    n_lig     = len(lig_conds)
    reg_names = [name for name, _, _ in REGIONS]
    reg_types = [region_type(name) for name, _, _ in REGIONS]
    n_reg     = len(REGIONS)

    # Y positions for horizontal layout
    y_pos = np.arange(n_reg)

    # -- violin width per condition
    # offset within each region
    n_conds  = len(all_conds)
    v_gap    = 0.18
    offsets  = np.linspace(-(n_conds-1)*v_gap/2,
                            (n_conds-1)*v_gap/2, n_conds)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                             wspace=0.38, width_ratios=[1.6, 1])

    
    # Panel A: violin per region
    
    ax_v = fig.add_subplot(gs[0])
    ax_v.set_facecolor("white"); ax_v.grid(False)
    ax_v.spines[["top","right"]].set_visible(False)

    # Region background stripes
    for yi, (rname, rtype) in enumerate(zip(reg_names, reg_types)):
        ax_v.axhspan(yi-0.45, yi+0.45,
                     color=TYPE_COLORS[rtype], alpha=0.45, zorder=0)

    for ci, cond_name in enumerate(all_conds):
        d   = data[cond_name]
        cfg = d["cfg"]
        col = cfg["color"]
        off = offsets[ci]

        for yi, rname in enumerate(reg_names):
            vals = np.array(d["reg_vals"][rname])
            if vals.size == 0: continue
            yc = yi + off

            # Violin body
            try:
                vp = ax_v.violinplot(
                    [vals], positions=[yc], vert=False,
                    widths=v_gap*0.85, showmedians=False,
                    showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(col)
                    pc.set_alpha(0.55)
                    pc.set_edgecolor(col)
                    pc.set_linewidth(0.5)
            except Exception:
                pass

            # Mean marker
            ax_v.scatter(vals.mean(), yc, color=col,
                         s=40, zorder=5,
                         edgecolors="white", linewidth=0.7)
    leg_handles = []
    for cond_name in all_conds:
        cfg = data[cond_name]["cfg"]
        leg_handles.append(
            Line2D([0],[0], color=cfg["color"], lw=3.0,
                   alpha=0.75, label=cfg["label"]))
    leg_handles += [
        Patch(facecolor="#D4E6F1", alpha=0.7, edgecolor="none",
              label="TM helices (rigid)"),
        Patch(facecolor="#FAD7A0", alpha=0.7, edgecolor="none",
              label="Loops / matrix helices (flexible)"),
        Line2D([0],[0], color="gray", lw=1.5, ls="--",
               alpha=0.5, label="±0.5 Å reference"),
    ]
    ax_v.legend(handles=leg_handles, fontsize=12, frameon=False,
                loc="upper right")

    ax_v.set_yticks(y_pos)
    ax_v.tick_params(axis="y", labelsize=17)
    ax_v.tick_params(axis="x", labelsize=17)
    ax_v.set_yticklabels(reg_names, fontsize=10, fontweight="bold")
    ax_v.set_xlabel("Cα RMSF (Å)", fontsize=FONT_LABEL, fontweight="bold")
    ax_v.tick_params(labelsize=17, direction="in")
    ax_v.set_xlim(left=0)
    ax_v.set_ylim(-0.6, n_reg-0.4)

    state_lbl = "c-state" if state=="c" else "m-state"
    ax_v.set_title(
        f"A)  Per-region Cα RMSF distribution ({state_lbl})",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)

    
    # Panel B: Δ-RMSF (ligand − Apo)
    
    ax_d = fig.add_subplot(gs[1])
    ax_d.set_facecolor("white"); ax_d.grid(False)
    ax_d.spines[["top","right"]].set_visible(False)

    # Region background stripes
    for yi, (rname, rtype) in enumerate(zip(reg_names, reg_types)):
        ax_d.axhspan(yi-0.45, yi+0.45,
                     color=TYPE_COLORS[rtype], alpha=0.45, zorder=0)

    bw    = 0.32
    n_lig = len(lig_conds)
    off_d = np.linspace(-(n_lig-1)*bw/2, (n_lig-1)*bw/2, n_lig)

    apo_means = data[apo_cond]["reg_means"]

    for oi, cond_name in zip(off_d, lig_conds):
        cfg      = data[cond_name]["cfg"]
        col      = cfg["color"]
        lig_m    = data[cond_name]["reg_means"]

        for yi, rname in enumerate(reg_names):
            apo_v = np.array(apo_means[rname])
            lig_v = np.array(lig_m[rname])
            if apo_v.size == 0 or lig_v.size == 0: continue

            apo_mean = apo_v.mean()
            lig_mean = lig_v.mean()
            delta    = lig_mean - apo_mean

            # SD of the difference (propagated)
            sd = np.sqrt((apo_v.std(ddof=1) if len(apo_v)>1 else 0)**2 +
                          (lig_v.std(ddof=1) if len(lig_v)>1 else 0)**2)

            yc = yi + oi
            ax_d.barh(yc, delta, bw*0.88,
                      color=col if delta >= 0 else col,
                      alpha=0.80 if delta >= 0 else 0.55,
                      edgecolor="white", linewidth=0.4)
            ax_d.errorbar(delta, yc, xerr=sd,
                          fmt="none", color=col, capsize=3,
                          elinewidth=1.0, capthick=1.0)

    ax_d.axvline(0, color="black", lw=1.0)
    ax_d.axvline( 0.5, color="gray", lw=0.7, ls="--", alpha=0.40)
    ax_d.axvline(-0.5, color="gray", lw=0.7, ls="--", alpha=0.40)
    ax_d.tick_params(axis="y", labelsize=17)
    ax_d.tick_params(axis="x", labelsize=17)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(reg_names, fontsize=10, fontweight="bold")
    ax_d.set_xlabel("Δ RMSF (Å)\n[ligand-bound − Apo]",
                    fontsize=FONT_LABEL, fontweight="bold")
    ax_d.tick_params(labelsize=17, direction="in")
    ax_d.set_ylim(-0.6, n_reg-0.4)

    apo_lbl = data[apo_cond]["cfg"]["label"]
    ax_d.set_title(
        f"B)  Δ RMSF = ligand − {apo_lbl}\n"
        f"[Positive = more flexible in ligand-bound state]",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)

    # -- Suptitle
    lig_labels = " | ".join(
        data[c]["cfg"]["label"] for c in lig_conds)
    plt.suptitle(
        f"Structural domain flexibility\n{lig_labels} vs {apo_lbl}",
        fontsize=FONT_TITLE+0.5, fontweight="bold", y=1.02)

    plt.savefig(f"{fname}.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf",
                bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {fname}.png / .pdf")
    plt.show()


# GENERATE BOTH FIGURES

print("\n" + "="*65)
print("Figure 1: c-state (FAD | PCAR | Apo-c)")
make_figure(
    state     = "c",
    lig_conds = ["FAD", "PCAR"],
    apo_cond  = "Apo_c",
    fname     = "Fig1_RMSF_cstate",
)

print("\n" + "="*65)
print("Figure 2: m-state (CAR | Apo-m)")
make_figure(
    state     = "m",
    lig_conds = ["CAR"],
    apo_cond  = "Apo_m",
    fname     = "Fig2_RMSF_mstate",
)

print("\nDone.")
