#!/usr/bin/env python3
"""
Hydrophobic / van der Waals contact analysis
FAD-bound vs P-carnitine-bound cBOU | 3 × 1µs replicas
Contact criterion: minimum heavy-atom distance < 4.5 Å
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
import os, sys, warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("Install MDAnalysis: conda install -c conda-forge mdanalysis")

# CONFIGURATION

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

GRO    = "step7_production.gro"
XTC    = "step7_production_fit.xtc"
STRIDE = 10   # analyse every Nth frame (~100 ps)

CONTACT_CUTOFF = 0.45   # nm = 4.5 Å

# Target residues (1-based GROMACS residue IDs)
TARGET_RESIDS = [15, 19, 72, 76, 80, 83, 118, 122, 125,
                 191, 192, 227, 228, 288, 289, 292]

# Residue display labels
RESID_LABELS = {
    15:  "GLY15",  19:  "LEU19",  72:  "PRO72",  76:  "VAL76",
    80:  "ASN80",  83:  "LEU83",  118: "VAL118", 122: "ALA122",
    125: "THR125", 191: "MET191", 192: "PHE192", 227: "PHE227",
    228: "TRP228", 288: "CYS288", 289: "PHE289", 292: "TYR292",
}

# Residue type for color-coding in Figure A
RES_TYPE = {
    "aromatic":  [192, 227, 228, 289, 292],  # PHE TRP TYR
    "aliphatic": [19, 72, 76, 83, 118, 122, 15, 191, 288],  # LEU PRO VAL ALA MET CYS GLY
    "polar":     [80, 125],   # ASN THR
}
TYPE_COLORS = {
    "aromatic":  "#9B59B6",
    "aliphatic": "#27AE60",
    "polar":     "#2980B9",
}

FAD_COLOR  = "#C03830"
PCAR_COLOR = "black"
FAD_SD     = "#F0A09D"
PCAR_SD    = "#A0A0A0"
DPI        = 1000
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICK  = 13


# COMPUTATION

def compute_contacts_replica(gro_path, xtc_path, resids, stride=STRIDE):
    """
    For each target residue, compute minimum distance to UNL (ligand)
    heavy atoms every `stride` frames.
    Returns:
      dist_traces: {resid: np.array of min-distances in nm, shape (n_frames,)}
      t_ns:        np.array of times in ns
    """
    u       = mda.Universe(gro_path, xtc_path)
    ligand  = u.select_atoms("resname UNL and not name H*")

    if len(ligand) == 0:
        print(f"    WARNING: no UNL atoms found in {gro_path}")
        return None, None

    # Pre-build protein selections per residue
    res_atoms = {}
    for resid in resids:
        sel = u.select_atoms(
            f"protein and resid {resid} and not name H*")
        if len(sel) > 0:
            res_atoms[resid] = sel
        else:
            print(f"    WARNING: resid {resid} not found")

    dist_lists = {r: [] for r in res_atoms}
    t_list     = []

    for i, ts in enumerate(u.trajectory):
        if i % stride != 0:
            continue
        lig_pos = ligand.positions / 10.0   # Å → nm

        for resid, sel in res_atoms.items():
            prot_pos = sel.positions / 10.0
            dm       = cdist(prot_pos, lig_pos)
            dist_lists[resid].append(float(dm.min()))

        t_list.append(ts.time / 1000.0)   # ps → ns

    return {r: np.array(v) for r, v in dist_lists.items()}, \
           np.array(t_list)


def load_or_compute_condition(dirs, tag, resids):
    """
    Load or compute contact traces for all replicas of one condition.
    Returns:
      rep_traces: list of {resid: array} per replica
      t_ref:      common time grid
    """
    rep_traces = []
    t_ref      = None

    for rep_idx, d in enumerate(dirs, 1):
        cache = os.path.join(d, f"hydrophobic_contacts_{tag}.npy")
        gp    = os.path.join(d, GRO)
        xp    = os.path.join(d, XTC)

        if os.path.exists(cache):
            print(f"    {tag} rep{rep_idx}: loading cache")
            data = np.load(cache, allow_pickle=True).item()
            rep_traces.append(data["traces"])
            if t_ref is None:
                t_ref = data["t"]
            continue

        if not (os.path.exists(gp) and os.path.exists(xp)):
            print(f"    {tag} rep{rep_idx}: files missing — skip")
            continue

        print(f"    {tag} rep{rep_idx}: computing "
              f"({len(resids)} residues × trajectory)...")
        traces, t_ns = compute_contacts_replica(gp, xp, resids)
        if traces is None:
            continue

        np.save(cache, {"traces": traces, "t": t_ns})
        print(f"    {tag} rep{rep_idx}: {len(t_ns)} frames saved")
        rep_traces.append(traces)
        if t_ref is None:
            t_ref = t_ns

    return rep_traces, t_ref


def average_across_replicas(rep_traces, resids, t_ref):
    """
    Interpolate replicas onto common time grid, return mean±SD per residue.
    Also returns occupancy (% frames < CONTACT_CUTOFF).
    """
    mean_dict = {}; sd_dict = {}; occ_dict = {}

    for resid in resids:
        traces_r = []
        for rep in rep_traces:
            if resid in rep:
                v = rep[resid]
                if len(v) != len(t_ref):
                    v = np.interp(t_ref,
                                  np.linspace(t_ref[0], t_ref[-1], len(v)),
                                  v)
                traces_r.append(v)

        if not traces_r:
            continue
        arr          = np.array(traces_r)
        mean_dict[resid] = arr.mean(axis=0)
        sd_dict[resid]   = arr.std(axis=0, ddof=1) \
                            if len(traces_r) > 1 else np.zeros(arr.shape[1])
        # occupancy from concatenated frames
        occ_dict[resid]  = 100.0 * np.mean(
            np.concatenate(traces_r) < CONTACT_CUTOFF)

    return mean_dict, sd_dict, occ_dict


# LOAD / COMPUTE ALL DATA

print("="*65)
print("HYDROPHOBIC CONTACT ANALYSIS")
print(f"Cutoff: {CONTACT_CUTOFF*10:.1f} Å | STRIDE: {STRIDE}")
print("="*65)

print("\n FAD replicas ")
fad_reps, fad_t = load_or_compute_condition(FAD_DIRS,  "FAD",  TARGET_RESIDS)
print("\n PCAR replicas ")
pcar_reps, pcar_t = load_or_compute_condition(PCAR_DIRS, "PCAR", TARGET_RESIDS)

# Use the longer time reference
t_ref = fad_t if fad_t is not None else pcar_t

print("\nAveraging across replicas...")
fad_mean,  fad_sd,  fad_occ  = average_across_replicas(fad_reps,  TARGET_RESIDS, t_ref)
pcar_mean, pcar_sd, pcar_occ = average_across_replicas(pcar_reps, TARGET_RESIDS, t_ref)

#  Save CSVs 
for tag, occ_d in [("FAD", fad_occ), ("PCAR", pcar_occ)]:
    csv_path = f"HydrophobicContacts_{tag}.csv"
    with open(csv_path, "w") as f:
        f.write("Residue,ResID,Occupancy_pct,Type\n")
        for resid in sorted(occ_d, key=lambda r: -occ_d[r]):
            rtype = next((k for k, v in RES_TYPE.items()
                          if resid in v), "other")
            f.write(f"{RESID_LABELS.get(resid,str(resid))},"
                    f"{resid},{occ_d[resid]:.2f},{rtype}\n")
    print(f"Saved: {csv_path}")

# Print summary
print(f"\n{'Residue':10} {'FAD occ%':10} {'PCAR occ%':10}")
print("" * 32)
all_r = sorted(set(fad_occ) | set(pcar_occ))
for r in all_r:
    lbl = RESID_LABELS.get(r, str(r))
    fo  = f"{fad_occ.get(r,0):.1f}%"
    po  = f"{pcar_occ.get(r,0):.1f}%"
    print(f"{lbl:10} {fo:10} {po:10}")


# SUPP FIGURE A — Dual occupancy bar chart

def res_type_color(resid):
    for rtype, ids in RES_TYPE.items():
        if resid in ids:
            return TYPE_COLORS[rtype]
    return "#888888"


def fig_occupancy():
    all_r = sorted(set(fad_occ) | set(pcar_occ),
                   key=lambda r: -(fad_occ.get(r,0) + pcar_occ.get(r,0))/2)
    labels   = [RESID_LABELS.get(r, str(r)) for r in all_r]
    fad_vals = [fad_occ.get(r, 0)  for r in all_r]
    pcar_vals= [pcar_occ.get(r, 0) for r in all_r]
    n        = len(all_r)
    y        = np.arange(n)
    bw       = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(18, max(14, n*0.65)),
                              gridspec_kw={"width_ratios":[2.5,1]})
    fig.patch.set_facecolor("white")

    #  Left: dual bars 
    ax = axes[0]
    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)

    for i, (r, fv, pv, lbl) in enumerate(
            zip(all_r, fad_vals, pcar_vals, labels)):
        rc = res_type_color(r)
        # FAD bar
        ax.barh(i + bw/2, fv, bw,
                color=FAD_COLOR, alpha=1.00, edgecolor="white")
        # PCAR bar
        ax.barh(i - bw/2, pv, bw,
                color=PCAR_COLOR, alpha=1.00, edgecolor="white",
                hatch="///", linewidth=0.0)
        # Type indicator on left
        ax.barh(i, 2.5, 0.85, left=-3, color=rc,
                alpha=0.95, edgecolor="none")

    # Value labels
    for i, (fv, pv) in enumerate(zip(fad_vals, pcar_vals)):
        if fv > 3:
            ax.text(fv+0.5, i+bw/2, f"{fv:.0f}%",
                    va="center", fontsize=16.5, color=FAD_COLOR,
                    fontweight="bold")
        if pv > 3:
            ax.text(pv+0.5, i-bw/2, f"{pv:.0f}%",
                    va="center", fontsize=16.5, color=PCAR_COLOR,
                    fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=19, fontweight="bold")
    ax.set_xlabel("Contact occupancy (%)", fontsize=19,
              fontweight="bold")
    ax.tick_params(axis="x", labelsize=24)
    ax.set_xlim(-4, 108)
    ax.axvline(50, color="gray", lw=0.7, ls="--", alpha=0.45)
    ax.axvline(80, color="gray", lw=0.5, ls="--", alpha=0.30)
    ax.set_title("Hydrophobic contact occupancy\n"
             f"[Cutoff: {CONTACT_CUTOFF*10:.1f} Å]",
             fontsize=FONT_TITLE+1, fontweight="bold")
    # Legends
    leg1 = [
        mpatches.Patch(color=FAD_COLOR,  alpha=1.00, label="cFAD-bound"),
        mpatches.Patch(color=PCAR_COLOR, alpha=1.00, hatch="///", label="cPCar-bound"),
        Line2D([0],[0], color="white", lw=1.0, ls="", label=""),
    ]
    leg2 = [mpatches.Patch(color=c, alpha=1.00, label=k.capitalize())
            for k, c in TYPE_COLORS.items()]
    ax.legend(handles=leg1+leg2, fontsize=16.5, frameon=False,
              loc="upper right", ncol=2)

    #  Right: difference plot 
    ax2 = axes[1]
    ax2.set_facecolor("white"); ax2.grid(False)
    ax2.spines[["top","right"]].set_visible(False)

    diff = [fv - pv for fv, pv in zip(fad_vals, pcar_vals)]
    bar_cols = [FAD_COLOR if d >= 0 else PCAR_COLOR for d in diff]
    ax2.barh(y, diff, 0.65,
             color=bar_cols, alpha=0.80, edgecolor="white")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.tick_params(axis="x", labelsize=24)
    ax2.set_xlabel("Δ Occupancy (%)\n[FAD − PCar]",
                   fontsize=19, fontweight="bold")
    ax2.set_title("Differential occupancy",
                  fontsize=FONT_TITLE, fontweight="bold")
    for i, d in enumerate(diff):
        offset = 0.80 if d >= 0 else -0.80
        col    = FAD_COLOR if d >= 0 else PCAR_COLOR
        ax2.text(d+offset, i, f"{d:+.0f}",
                 va="center",
                 ha="left" if d >= 0 else "right",
                 fontsize=12.5, color=col, fontweight="bold")

    plt.suptitle(
        "Hydrophobic / van der Waals contact profile",
        fontsize=FONT_TITLE+2, fontweight="bold")
    plt.tight_layout()
    plt.savefig("SuppFig_A_contact_occupancy.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("SuppFig_A_contact_occupancy.pdf",
                bbox_inches="tight", facecolor="white")
    print("Saved: SuppFig_A_contact_occupancy.png / .pdf")
    plt.show()


# SUPP FIGURE B — Time series grid (4 × 4)

def fig_timeseries():
    all_r = [r for r in TARGET_RESIDS
             if r in fad_mean or r in pcar_mean]
    ncols = 4
    nrows = int(np.ceil(len(all_r) / ncols))

    fig = plt.figure(figsize=(ncols*4.5, nrows*3.0 + 0.9))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                             hspace=0.58, wspace=0.30,
                             top=0.88, bottom=0.06)

    for idx, resid in enumerate(all_r):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)

        lbl   = RESID_LABELS.get(resid, str(resid))
        rc    = res_type_color(resid)
        lines = []

        has_fad  = resid in fad_mean
        has_pcar = resid in pcar_mean

        if has_fad:
            m, s = fad_mean[resid], fad_sd[resid]
            ax.fill_between(t_ref, m-s, m+s, color=FAD_SD, alpha=0.38)
            occ_f = fad_occ.get(resid, 0)
            l, = ax.plot(t_ref, m, color=FAD_COLOR, lw=1.4,
                         label=f"FAD ({occ_f:.0f}%)")
            lines.append(l)

        if has_pcar:
            m, s = pcar_mean[resid], pcar_sd[resid]
            ax.fill_between(t_ref, m-s, m+s, color=PCAR_SD, alpha=0.28)
            occ_p = pcar_occ.get(resid, 0)
            l, = ax.plot(t_ref, m, color=PCAR_COLOR, lw=1.2,
                          label=f"PCAR ({occ_p:.0f}%)")
            lines.append(l)

        # Cutoff line (no legend entry — shared legend at top)
        ax.axhline(CONTACT_CUTOFF, color="gray",
                   lw=0.8, ls="--", alpha=0.55)

        # Y limits
        all_v = []
        if has_fad:
            m, s = fad_mean[resid], fad_sd[resid]
            all_v.extend((m-s).tolist()); all_v.extend((m+s).tolist())
        if has_pcar:
            m, s = pcar_mean[resid], pcar_sd[resid]
            all_v.extend((m-s).tolist()); all_v.extend((m+s).tolist())
        if all_v:
            ymin = max(0.0, np.min(all_v)-0.02)
            ymax = np.max(all_v)+0.04
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(0, 1000)
        ax.tick_params(labelsize=FONT_TICK)

        if row == nrows-1:
            ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL,
                          fontweight="bold")
        if col == 0:
            ax.set_ylabel("Distance (nm)",
                          fontsize=FONT_LABEL, fontweight="bold")

        # Title: residue name, colored by type
        ax.set_title(lbl, fontsize=FONT_TITLE+1, fontweight="bold",
                     pad=5, color=rc)

        # Per-panel legend (lines only)
        if lines:
            ax.legend(handles=lines, fontsize=9.5,
                      frameon=False, loc="upper right")

    # Empty panels off
    for idx in range(len(all_r), nrows*ncols):
        r, c = divmod(idx, ncols)
        fig.add_subplot(gs[r, c]).axis("off")

    # Shared figure-level legend at TOP
    fig_handles = [
        Line2D([0],[0], color=FAD_COLOR,  lw=2.0,
               label="cFAD-bound (mean)"),
        mpatches.Patch(facecolor=FAD_SD, edgecolor="none",
                       alpha=0.5, label="cFAD ±SD"),
        Line2D([0],[0], color=PCAR_COLOR, lw=1.6,
               label="cPCar-bound (mean)"),
        mpatches.Patch(facecolor=PCAR_SD, edgecolor="none",
                       alpha=0.5, label="cPCar ±SD"),
        Line2D([0],[0], color="gray", lw=0.9, ls="--",
               label=f"Cutoff {CONTACT_CUTOFF*10:.1f} Å"),
        # Type legend
        mpatches.Patch(color=TYPE_COLORS["aromatic"],  alpha=1.00,
                       label="Aromatic"),
        mpatches.Patch(color=TYPE_COLORS["aliphatic"], alpha=1.00,
                       label="Aliphatic"),
        mpatches.Patch(color=TYPE_COLORS["polar"],     alpha=1.00,
                       label="Polar"),
    ]
    fig.legend(handles=fig_handles, fontsize=FONT_TICK+0.5,
               loc="upper center", ncol=8,
               bbox_to_anchor=(0.5, 0.97),
               frameon=False, columnspacing=1.0)

    plt.suptitle(
        "Hydrophobic / vdW contact distance profiles",
        fontsize=FONT_TITLE+1, fontweight="bold", y=1.01)

    plt.savefig("SuppFig_B_contact_timeseries.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("SuppFig_B_contact_timeseries.pdf",
                bbox_inches="tight", facecolor="white")
    print("Saved: SuppFig_B_contact_timeseries.png / .pdf")
    plt.show()


# MAIN

print("\nGenerating figures...")
fig_occupancy()
fig_timeseries()

print("\n" + "="*65)
print("COMPLETE")
for f in ["SuppFig_A_contact_occupancy.png",
          "SuppFig_B_contact_timeseries.png",
          "HydrophobicContacts_FAD.csv",
          "HydrophobicContacts_PCAR.csv"]:
    print(f"  [{'OK' if os.path.exists(f) else 'miss'}] {f}")
