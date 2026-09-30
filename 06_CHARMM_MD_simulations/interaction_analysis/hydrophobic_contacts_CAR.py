#!/usr/bin/env python3
"""
Supplementary: Hydrophobic / van der Waals contact analysis
CAR-bound m-state | 3 replicas × 1µs
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
    sys.exit("conda install -c conda-forge mdanalysis")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

CAR_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]

GRO    = "step7_production.gro"
XTC    = "step7_production_fit.xtc"
STRIDE = 10

CONTACT_CUTOFF = 0.45   # nm = 4.5 Å

# Target residues — those with notable occupancy across replicas
TARGET_RESIDS = [76, 80, 83, 184, 188, 191, 281, 288]

RESID_LABELS = {
    76:  "VAL76", 80:  "ASN80",  83:  "LEU83", 184: "GLU184", 
    188: "ASN188", 191: "MET191", 281: "SER281", 288: "CYS288",
}

RES_TYPE = {
    "aliphatic": [76, 83, 191, 288],
    "polar":     [80, 184, 188, 281],
}
TYPE_COLORS = {
    "aliphatic": "#27AE60",
    "polar":     "#2980B9",
}

CAR_COLOR = "black"
CAR_SD    = "#909090"
DPI        = 1000
FONT_TITLE = 18
FONT_LABEL = 15
FONT_TICK  = 14


# ═══════════════════════════════════════════════════════════════════════
# COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_contacts_replica(gro_path, xtc_path, resids, stride=STRIDE):
    u      = mda.Universe(gro_path, xtc_path)
    ligand = u.select_atoms("resname UNL and not name H*")
    if len(ligand) == 0:
        print(f"  WARNING: no UNL in {gro_path}"); return None, None

    res_atoms = {}
    for resid in resids:
        sel = u.select_atoms(f"protein and resid {resid} and not name H*")
        if len(sel) > 0: res_atoms[resid] = sel

    dist_lists = {r: [] for r in res_atoms}
    t_list     = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        lig_pos = ligand.positions / 10.0
        for resid, sel in res_atoms.items():
            dm = cdist(sel.positions/10.0, lig_pos)
            dist_lists[resid].append(float(dm.min()))
        t_list.append(ts.time/1000.0)
    return {r: np.array(v) for r, v in dist_lists.items()}, np.array(t_list)


def load_or_compute(dirs, tag, resids):
    rep_traces, t_ref = [], None
    for rep_idx, d in enumerate(dirs, 1):
        cache = os.path.join(d, f"hydrophobic_contacts_{tag}.npy")
        gp = os.path.join(d, GRO); xp = os.path.join(d, XTC)
        if os.path.exists(cache):
            print(f"  {tag} rep{rep_idx}: loading cache")
            data = np.load(cache, allow_pickle=True).item()
            rep_traces.append(data["traces"])
            if t_ref is None: t_ref = data["t"]
            continue
        if not (os.path.exists(gp) and os.path.exists(xp)):
            print(f"  {tag} rep{rep_idx}: files missing"); continue
        print(f"  {tag} rep{rep_idx}: computing...")
        traces, t_ns = compute_contacts_replica(gp, xp, resids)
        if traces is None: continue
        np.save(cache, {"traces": traces, "t": t_ns})
        rep_traces.append(traces)
        if t_ref is None: t_ref = t_ns
    return rep_traces, t_ref


def average_replicas(rep_traces, resids, t_ref):
    mean_d, sd_d, occ_d = {}, {}, {}
    for resid in resids:
        traces_r = []
        for rep in rep_traces:
            if resid in rep:
                v = rep[resid]
                if len(v) != len(t_ref):
                    v = np.interp(t_ref,
                                  np.linspace(t_ref[0], t_ref[-1], len(v)), v)
                traces_r.append(v)
        if not traces_r: continue
        arr = np.array(traces_r)
        mean_d[resid] = arr.mean(axis=0)
        sd_d[resid]   = arr.std(axis=0, ddof=1) if len(traces_r)>1 \
                        else np.zeros(arr.shape[1])
        occ_d[resid]  = 100.0*np.mean(np.concatenate(traces_r) < CONTACT_CUTOFF)
    return mean_d, sd_d, occ_d


def res_type_color(resid):
    for rtype, ids in RES_TYPE.items():
        if resid in ids: return TYPE_COLORS[rtype]
    return "#888888"


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

print("="*65)
print("mCAR HYDROPHOBIC CONTACT ANALYSIS")
print(f"Cutoff: {CONTACT_CUTOFF*10:.1f} Å | STRIDE: {STRIDE}")
print("="*65)

car_reps, car_t = load_or_compute(CAR_DIRS, "CAR", TARGET_RESIDS)
t_ref = car_t

print("\nAveraging across replicas...")
car_mean, car_sd, car_occ = average_replicas(car_reps, TARGET_RESIDS, t_ref)

# CSV
csv_path = "HydrophobicContacts_CAR.csv"
with open(csv_path, "w") as f:
    f.write("Residue,ResID,Occupancy_pct,Type\n")
    for resid in sorted(car_occ, key=lambda r: -car_occ[r]):
        rtype = next((k for k, v in RES_TYPE.items() if resid in v), "other")
        f.write(f"{RESID_LABELS.get(resid,str(resid))},{resid},"
                f"{car_occ[resid]:.2f},{rtype}\n")
print(f"Saved: {csv_path}")

print(f"\n{'Residue':10} {'CAR occ%':10}")
print("─"*22)
for r in sorted(car_occ, key=lambda x: -car_occ[x]):
    print(f"{RESID_LABELS.get(r,str(r)):10} {car_occ[r]:5.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE A — Occupancy bar chart + difference from expected
# ═══════════════════════════════════════════════════════════════════════

all_r    = sorted(car_occ, key=lambda r: -car_occ[r])
labels   = [RESID_LABELS.get(r, str(r)) for r in all_r]
occ_vals = [car_occ[r] for r in all_r]
n        = len(all_r)
y        = np.arange(n)

fig1, ax = plt.subplots(figsize=(10, max(7, n*0.42)))
fig1.patch.set_facecolor("white")
ax.set_facecolor("white"); ax.grid(False)
ax.spines[["top","right"]].set_visible(False)

for i, (r, ov, lbl) in enumerate(zip(all_r, occ_vals, labels)):
    rc = res_type_color(r)
    ax.barh(i, ov, 0.65, color=CAR_COLOR, alpha=1.00, edgecolor="white")
    ax.barh(i, 2.5, 0.75, left=-3, color=rc, alpha=0.95, edgecolor="none")
    if ov > 3:
        ax.text(ov+0.5, i, f"{ov:.0f}%",
                va="center", fontsize=11.5, color=CAR_COLOR,
                fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
ax.set_xlabel("Contact occupancy (%)", fontsize=FONT_LABEL+1,
              fontweight="bold")
ax.set_xlim(-4, 108)
ax.axvline(50, color="gray", lw=0.7, ls="--", alpha=0.45)
ax.axvline(80, color="gray", lw=0.5, ls="--", alpha=0.30)
ax.set_title(f"mCAR-bound hydrophobic / vdW contact occupancy\n"
             f"Cutoff: {CONTACT_CUTOFF*10:.1f} Å",
             fontsize=FONT_TITLE+1, fontweight="bold")

leg = [
    mpatches.Patch(color=CAR_COLOR, alpha=1.00, label="mCAR-bound"),
    Line2D([0],[0], color="gray", lw=1.0, ls="--",
           label="50% / 80% thresholds"),
]
leg += [mpatches.Patch(color=c, alpha=0.95, label=k.capitalize())
        for k, c in TYPE_COLORS.items()]
ax.legend(handles=leg, fontsize=12, frameon=False,
          loc="upper right", ncol=2)

plt.tight_layout()
plt.savefig("SuppFig_mCAR_contact_occupancy.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("SuppFig_mCAR_contact_occupancy.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: SuppFig_mCAR_contact_occupancy.png / .pdf")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE B — Time series grid
# ═══════════════════════════════════════════════════════════════════════

def fig_timeseries():
    all_r2 = [r for r in TARGET_RESIDS if r in car_mean]
    ncols  = 4
    nrows  = int(np.ceil(len(all_r2) / ncols))

    fig = plt.figure(figsize=(ncols*4.5, nrows*3.0+0.9))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                             hspace=0.58, wspace=0.30,
                             top=0.88, bottom=0.06)

    for idx, resid in enumerate(all_r2):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)

        lbl = RESID_LABELS.get(resid, str(resid))
        rc  = res_type_color(resid)
        m   = car_mean[resid]; s = car_sd[resid]

        ax.fill_between(t_ref, m-s, m+s, color=CAR_SD, alpha=0.38)
        occ = car_occ.get(resid, 0)
        l,  = ax.plot(t_ref, m, color=CAR_COLOR, lw=1.4, alpha=1.00,
                      label=f"mCAR ({occ:.0f}%)")

        ax.axhline(CONTACT_CUTOFF, color="gray",
                   lw=0.8, ls="--", alpha=0.85)

        all_v = (m-s).tolist()+(m+s).tolist()
        ax.set_ylim(max(0.0, np.min(all_v)-0.02), np.max(all_v)+0.04)
        ax.set_xlim(0, 1000)
        ax.tick_params(labelsize=FONT_TICK+2.5)
        if row == nrows-1:
            ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL+2.5, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Distance (nm)", fontsize=FONT_LABEL+2.5, fontweight="bold")

        ax.set_title(lbl, fontsize=FONT_TITLE+3.0, fontweight="bold",
                     pad=5, color=rc)
        ax.legend(handles=[l], fontsize=15.5, frameon=False, loc="upper right")

    for idx in range(len(all_r2), nrows*ncols):
        r, c = divmod(idx, ncols)
        fig.add_subplot(gs[r, c]).axis("off")

    fig_handles = [
        Line2D([0],[0], color=CAR_COLOR, lw=2.0, label="mCAR-bound (mean)"),
        mpatches.Patch(facecolor=CAR_SD, edgecolor="none",
                       alpha=1.0, label="mCAR ±SD"),
        Line2D([0],[0], color="gray", lw=0.9, ls="--",
               label=f"Cutoff {CONTACT_CUTOFF*10:.1f} Å"),
        mpatches.Patch(color=TYPE_COLORS["aliphatic"], alpha=1.00,
                       label="Aliphatic"),
        mpatches.Patch(color=TYPE_COLORS["polar"],     alpha=1.00,
                       label="Polar"),
    ]
    fig.legend(handles=fig_handles, fontsize=FONT_TICK+3,
               loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.03), frameon=False, columnspacing=1.0)

    plt.suptitle(
        "mCAR-bound hydrophobic",
        fontsize=FONT_TITLE+1, fontweight="bold", y=1.2)
    plt.savefig("SuppFig_mCAR_contact_timeseries.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("SuppFig_mCAR_contact_timeseries.pdf",
                bbox_inches="tight", facecolor="white")
    print("Saved: SuppFig_mCAR_contact_timeseries.png / .pdf")
    plt.show()

fig_timeseries()

print("\n" + "="*65)
print("COMPLETE")
for fn in ["SuppFig_mCAR_contact_occupancy.png",
           "SuppFig_mCAR_contact_timeseries.png",
           "HydrophobicContacts_CAR.csv"]:
    print(f"  [{'OK  ' if os.path.exists(fn) else 'MISS'}] {fn}")
