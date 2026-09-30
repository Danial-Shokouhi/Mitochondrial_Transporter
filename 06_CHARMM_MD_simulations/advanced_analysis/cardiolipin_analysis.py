#!/usr/bin/env python3
"""
Conditions: cFAD | cPCAR | mCAR | Apo-c | Apo-m
3 replicas × 1µs each
Cardiolipin types: LOCCL (18:2/18:2/18:2/18:1) × 9
                   LNCCL (18:2/18:2/18:2/18:3) × 5

Contact cutoff: 6.0 Å (standard for lipid–protein contacts)
Analysis (from cardiolipin_interaction.py):
  1. Per-residue CL contact occupancy (any CL heavy atom < 6.0 Å)
  2. Headgroup vs acyl-chain decomposition
  3. Leaflet assignment (IMS vs matrix) of contacting CLs
  4. Identification of hotspot residues (occupancy > 30%)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist
import os, sys, warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("conda install -c conda-forge mdanalysis")

# ═══════════════════════════════════════════════════════════════════════
# DIRECTORIES / CONDITIONS
# ═══════════════════════════════════════════════════════════════════════

CONDITIONS = {
    "FAD_c": {
        "dirs":  [".../rep1",
                  ".../rep2",
                  ".../rep3"],
        "label": "cFAD-bound",
        "color": "#C03830",
        "state": "c",
    },
    "PCAR_c": {
        "dirs":  [".../rep1",
                  ".../rep2",
                  ".../rep3"],
        "label": "cPCar-bound",
        "color": "#1A1A1A",
        "state": "c",
    },
    "Apo_c": {
        "dirs":  [".../rep1",
                  ".../rep2",
                  ".../rep3"],
        "label": "Apo-cBOU",
        "color": "#2166AC",
        "state": "c",
    },
    "CAR_m": {
        "dirs":  [".../rep1",
                  ".../rep2",
                  ".../rep3"],
        "label": "mCAR-bound",
        "color": "#6B4E2F",
        "state": "m",
    },
    "Apo_m": {
        "dirs":  [".../rep1",
                  ".../rep2",
                  ".../rep3"],
        "label": "Apo-mBOU",
        "color": "#63B5E5",
        "state": "m",
    },
}

GRO    = "step7_production.gro"
XTC    = "step7_production_fit.xtc"
STRIDE = 20    # every 2 ns (heavy computation)
N_RES  = 300
CL_CUTOFF_A = 6.0    # Å — lipid-protein contact threshold

# Cardiolipin residue names
CL_RESNAMES = ["LOCCL", "LNCCL"]

# Headgroup atom name prefixes (phosphate + glycerol backbone)
HEADGROUP_NAMES = {
    "P1", "P3",
    "OP11","OP12","OP13","OP14",
    "OP31","OP32","OP33","OP34",
    "OG11","OG12","OG21","OG22",
    "HG11","HG12","HG21","HG22","HG31","HG32",
    "HO12","HO22",
    "C1","C2","C3",
    "C11","C12","C13","C21","C22","C23","C31","C32","C33",
    "O12","O13","O32","O33",
}
# Everything else (CA*, CB*, CC*, CD* series) = acyl chain

# Structural regions for background shading
REGIONS = [
    ("NTER",  1,  1), ("TMH1",  2, 33), ("ml12", 34, 48),
    ("mh12", 49, 69), ("TMH2", 70, 94), ("imsl23",95,103),
    ("TMH3",104,138), ("ml34",139,155), ("mh34",156,176),
    ("TMH4",177,203), ("imsl45",204,212),("TMH5",213,242),
    ("ml56",243,253), ("mh56",254,273), ("TMH6",274,299),
    ("CTER",300,300),
]
def region_color(n):
    if n.startswith("TMH"): return "#D4E6F1"
    if n in ("NTER","CTER"): return "#F2F3F4"
    return "#FAD7A0"

# Key basic residues likely to interact with headgroup (kept from the
# analysis script for reference; the figure uses GRO-derived res_label()
# instead of these).
BASIC_RESIDS = {7,18,26,29,87,126,129,183,196,199,238,280,293,296}
BASIC_LABELS = {
    7:"D7",18:"Q18",26:"D26",29:"K29",87:"R87",
    126:"E126",129:"K129",183:"R183",196:"E196",199:"K199",
    238:"K238",280:"R280",293:"E293",296:"R296",
}

DPI        = 1000
FONT_TITLE = 12
FONT_LABEL = 11
FONT_TICK  = 10


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════

def is_headgroup(atom_name):
    return atom_name.strip() in HEADGROUP_NAMES


def analyse_replica(gro_path, xtc_path, n_res=N_RES, stride=STRIDE,
                    cutoff_a=CL_CUTOFF_A):
    """
    Returns:
      occ_total (n_res,)     — % frames with any CL contact
      occ_head  (n_res,)     — % frames with headgroup contact
      occ_acyl  (n_res,)     — % frames with acyl chain contact
      occ_ims   (n_res,)     — % frames with IMS-leaflet CL contact
      occ_mat   (n_res,)     — % frames with matrix-leaflet CL contact
    """
    u      = mda.Universe(gro_path, xtc_path)
    prot   = u.select_atoms("protein and not name H*")
    cl_all = u.select_atoms(
        f"resname {' '.join(CL_RESNAMES)} and not name H*")

    if len(cl_all) == 0:
        print("    WARNING: no cardiolipin atoms found")
        return (np.zeros(n_res),)*5

    # Partition CL atoms into headgroup vs acyl chain
    cl_head_idx = np.array([i for i,a in enumerate(cl_all)
                             if is_headgroup(a.name)])
    cl_acyl_idx = np.array([i for i,a in enumerate(cl_all)
                             if not is_headgroup(a.name)])

    # Residue index for each protein atom
    prot_residx = np.array([a.resindex for a in prot])

    # Find membrane midplane from phosphorus Z-coordinates
    # (use first frame only — stable reference)
    first_ts = next(iter(u.trajectory))
    phos = u.select_atoms(
        f"resname {' '.join(CL_RESNAMES)} and name P1 P3")
    phos_z    = phos.positions[:, 2]
    mid_z     = np.median(phos_z)   # membrane midplane
    print(f"    Membrane midplane Z = {mid_z:.1f} Å")

    # Accumulators
    cnt_tot  = np.zeros(n_res, dtype=np.int32)
    cnt_head = np.zeros(n_res, dtype=np.int32)
    cnt_acyl = np.zeros(n_res, dtype=np.int32)
    cnt_ims  = np.zeros(n_res, dtype=np.int32)
    cnt_mat  = np.zeros(n_res, dtype=np.int32)
    n_frames = 0

    for fi, ts in enumerate(u.trajectory):
        if fi % stride != 0: continue
        n_frames += 1

        prot_pos = prot.positions            # (N_prot_atoms, 3)
        cl_pos   = cl_all.positions          # (N_cl_atoms, 3)

        # Full distance matrix — vectorised
        dm = cdist(prot_pos, cl_pos)         # (N_prot, N_cl)

        # Min distance per protein residue to any CL atom
        for ri in range(n_res):
            mask = prot_residx == ri
            if not mask.any(): continue
            row = dm[mask, :]                # (n_atoms_in_res, N_cl)
            min_d = row.min(axis=0)          # (N_cl,)  per CL atom

            in_contact = (min_d < cutoff_a)
            if not in_contact.any(): continue

            cnt_tot[ri] += 1

            # Headgroup / acyl
            if len(cl_head_idx) > 0 and in_contact[cl_head_idx].any():
                cnt_head[ri] += 1
            if len(cl_acyl_idx) > 0 and in_contact[cl_acyl_idx].any():
                cnt_acyl[ri] += 1

            # Leaflet assignment: which CL residues are in contact?
            cl_resindices = cl_all.resindices
            contact_cl_resix = np.unique(
                cl_resindices[in_contact])

            # Get phosphate Z for each contacting CL molecule
            for clix in contact_cl_resix:
                cl_mol = u.select_atoms(
                    f"resname {' '.join(CL_RESNAMES)} "
                    f"and resindex {clix} and name P1 P3")
                if len(cl_mol) == 0: continue
                mean_z = cl_mol.positions[:,2].mean()
                if mean_z > mid_z:
                    cnt_ims[ri] += 1; break
                else:
                    cnt_mat[ri] += 1; break

    if n_frames == 0:
        return (np.zeros(n_res),)*5

    def pct(arr): return 100.0 * arr / n_frames
    print(f"    {n_frames} frames | "
          f"hotspots (>30%): "
          f"{(pct(cnt_tot)>30).sum()} residues")
    return (pct(cnt_tot), pct(cnt_head),
            pct(cnt_acyl), pct(cnt_ims), pct(cnt_mat))


def load_or_compute_condition(cond_name, cfg, n_res=N_RES):
    """
    Returns 5 arrays each (n_res,) averaged across replicas.
    Loads a cached .npy per replica if present, otherwise computes it
    from the GRO/XTC and caches the result for next time.
    """
    keys   = ["tot","head","acyl","ims","mat"]
    sums   = {k: np.zeros(n_res) for k in keys}
    n_done = 0

    for rep_idx, d in enumerate(cfg["dirs"], 1):
        cache = os.path.join(d, f"cardiolipin_{cond_name}_rep{rep_idx}.npy")
        gp    = os.path.join(d, GRO)
        xp    = os.path.join(d, XTC)

        if os.path.exists(cache):
            print(f"  {cond_name} rep{rep_idx}: loading cache")
            arr = np.load(cache)
        elif os.path.exists(gp) and os.path.exists(xp):
            print(f"  {cond_name} rep{rep_idx}: computing...")
            results = analyse_replica(gp, xp, n_res)
            arr = np.array(results)     # (5, n_res)
            np.save(cache, arr)
        else:
            print(f"  {cond_name} rep{rep_idx}: MISSING files — skip")
            continue

        for ki, k in enumerate(keys):
            sums[k] += arr[ki]
        n_done += 1

    if n_done == 0:
        return {k: np.zeros(n_res) for k in keys}

    return {k: sums[k]/n_done for k in keys}


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — Get real residue names from GRO 
# ═══════════════════════════════════════════════════════════════════════

def get_res_names_from_gro(gro_path, n_res=N_RES):
    """Read GROMACS GRO and return {resnum: resname} for protein residues."""
    names = {}
    with open(gro_path) as f:
        f.readline()                        # title line
        n_atoms = int(f.readline().strip())
        for _ in range(n_atoms):
            line = f.readline()
            try:
                resnum   = int(line[:5].strip())
                resname  = line[5:10].strip()
            except ValueError:
                continue
            if resnum not in names and 1 <= resnum <= n_res:
                names[resnum] = resname
    return names

# Try each condition until one GRO is found
res_names = {}
for cond_name, cfg in CONDITIONS.items():
    for d in cfg["dirs"]:
        gro = os.path.join(d, "step7_production.gro")
        if os.path.exists(gro):
            print(f"Reading residue names from: {gro}")
            res_names = get_res_names_from_gro(gro)
            print(f"  Found {len(res_names)} residues")
            break
    if res_names: break

if not res_names:
    # Fallback: generic labels
    print("WARNING: no GRO found, using generic labels")
    res_names = {i: f"AA{i}" for i in range(1, N_RES+1)}


def res_label(resnum):
    """Return e.g. 'LEU177' for residue 177."""
    name = res_names.get(resnum, "UNK")
    return f"{name}{resnum}"


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Load or compute data  (analysis engine, cached as .npy)
# ═══════════════════════════════════════════════════════════════════════

print("="*65)
print("CARDIOLIPIN – PROTEIN INTERACTION ANALYSIS")
print(f"CL contact cutoff: {CL_CUTOFF_A:.1f} Å | STRIDE: {STRIDE}")
print("="*65)

all_data = {}
for cond_name, cfg in CONDITIONS.items():
    print(f"\n── {cfg['label']} ──")
    all_data[cond_name] = load_or_compute_condition(cond_name, cfg)
    top_val = all_data[cond_name]["tot"].max()
    top_res = all_data[cond_name]["tot"].argmax() + 1
    print(f"  {cfg['label']:15}: max {top_val:.1f}% at {res_label(top_res)}")

# Summary print
print("\n" + "="*65)
print("Summary: residues with >30% CL contact occupancy")
for cond_name, cfg in CONDITIONS.items():
    d    = all_data[cond_name]["tot"]
    hot  = np.where(d > 30)[0] + 1   # 1-based
    print(f"  {cfg['label']:15}: {len(hot)} hotspots  "
          f"(max={d.max():.1f}% at R{d.argmax()+1})")

# Save CSV
with open("Cardiolipin_contacts.csv","w") as f:
    headers = ["Residue"] + [
        f"{cond}_{k}" for cond in CONDITIONS
        for k in ["total","head","acyl","IMS","matrix"]]
    f.write(",".join(headers) + "\n")
    for ri in range(N_RES):
        row = [str(ri+1)]
        for cond_name in CONDITIONS:
            d = all_data[cond_name]
            row += [f"{d[k][ri]:.2f}"
                    for k in ["tot","head","acyl","ims","mat"]]
        f.write(",".join(row) + "\n")
print("Saved: Cardiolipin_contacts.csv")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Derive hotspot lists
# ═══════════════════════════════════════════════════════════════════════

hm_data   = np.array([all_data[c]["tot"] for c in CONDITIONS])  # (5,300)
max_occ   = hm_data.max(axis=0)

# Top 12 residues ranked by max occupancy across all conditions
TOP_N     = 12
ANNO_THR  = 60    # annotate heatmap only above this %

top_idx   = np.argsort(max_occ)[::-1][:TOP_N]
top_res   = top_idx + 1
top_labels = [res_label(r) for r in top_res]

# Hotspot residues for annotation on heatmap (>ANNO_THR in any condition)
anno_mask = max_occ >= ANNO_THR
anno_positions = np.where(anno_mask)[0]  # 0-based indices


# ═══════════════════════════════════════════════════════════════════════
# FIGURE — 3 panels, heatmap removed
#   A) c-state hotspots   (dumbbell)
#   B) m-state hotspots   (dumbbell)
#   C) headgroup vs acyl  (tracked stacked bar)
# ═══════════════════════════════════════════════════════════════════════

cond_order = list(CONDITIONS.keys())
n_conds    = len(cond_order)

fig = plt.figure(figsize=(20, 13.5))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(2, 2, figure=fig,
                        top=0.885, hspace=0.65, wspace=0.30,
                        height_ratios=[1.0, 1.05])


def style_axis(ax):
    """Shared cosmetic treatment for all panels."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="y", length=0)


def row_stripes(ax, n_rows):
    """Alternating background stripes to guide the eye across rows."""
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5,
                       color="#F5F6F7", zorder=0)


def dumbbell_panel(ax, state, panel_letter, panel_title, ytick_pad=None):
    """
    Dumbbell / lollipop comparison of CL occupancy across the
    conditions belonging to one state.
    A thin grey rod spans the min–max range for each residue;
    one filled marker per condition sits on the rod.

    ytick_pad: extra spacing (points) between the residue-name labels
    and the axis — useful when a condition has zero occupancy at some
    residues (marker sits right at x=0), which otherwise crowds the
    labels. Leave as None to keep matplotlib's default spacing.
    """
    state_conds = [c for c in cond_order
                   if CONDITIONS[c]["state"] == state]
    max_v = np.max([all_data[c]["tot"] for c in state_conds], axis=0)
    top   = np.argsort(max_v)[::-1][:TOP_N]
    top   = top[::-1]                       # highest at the top of the axis
    labels = [res_label(r + 1) for r in top]
    y      = np.arange(len(top))

    row_stripes(ax, len(top))

    # Reference guides
    ax.axvline(30, color="#B0B0B0", lw=0.9, ls="--", alpha=0.55, zorder=1)
    ax.axvline(60, color="#D0D0D0", lw=0.9, ls="--", alpha=0.45, zorder=1)

    # Connector rods
    for yi, ri in zip(y, top):
        vals = [all_data[c]["tot"][ri] for c in state_conds]
        ax.plot([min(vals), max(vals)], [yi, yi],
                color="#9AA0A6", lw=1.0, solid_capstyle="round",
                zorder=2, alpha=0.85)

    # Condition markers
    for c in state_conds:
        cfg  = CONDITIONS[c]
        vals = [all_data[c]["tot"][ri] for ri in top]
        ax.scatter(vals, y,
                   s=190, color=cfg["color"],
                   edgecolors="white", linewidth=1.6,
                   zorder=4, label=cfg["label"], clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
    if ytick_pad is not None:
        ax.tick_params(axis="y", pad=ytick_pad)
    ax.set_xlabel("Contact occupancy (%)",
                  fontsize=FONT_LABEL+6.0, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.tick_params(labelsize=FONT_TICK+6.0)
    ax.set_title(f"{panel_letter})  {panel_title}",
                 fontsize=FONT_TITLE+4.0, fontweight="bold", pad=10)
    style_axis(ax)


# ── Panel A: c-state ──────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 0])
dumbbell_panel(ax_c, "c", "A", "Top hotspots — c-state")

# ── Panel B: m-state ──────────────────────────────────────────────────
ax_m = fig.add_subplot(gs[0, 1])
dumbbell_panel(ax_m, "m", "B", "Top hotspots — m-state", ytick_pad=15.5)


# ── Shared condition legend for panels A and B ────────────────────────
cond_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=CONDITIONS[c]["color"],
           markeredgecolor="white", markeredgewidth=1.2,
           markersize=13, label=CONDITIONS[c]["label"])
    for c in cond_order
]
fig.legend(handles=cond_handles, fontsize=15, frameon=False,
           loc="upper center", ncol=len(cond_handles),
           bbox_to_anchor=(0.5, 0.955),
           columnspacing=1.8, handletextpad=0.5)


# ── Panel C: headgroup vs acyl chain decomposition ────────────────────
ax_hg = fig.add_subplot(gs[1, :])

avg_head = np.mean([all_data[c]["head"] for c in cond_order], axis=0)
avg_acyl = np.mean([all_data[c]["acyl"] for c in cond_order], axis=0)
avg_ims  = np.mean([all_data[c]["ims"]  for c in cond_order], axis=0)
avg_mat  = np.mean([all_data[c]["mat"]  for c in cond_order], axis=0)
avg_tot  = np.mean([all_data[c]["tot"]  for c in cond_order], axis=0)

top_d    = np.argsort(avg_tot)[::-1][:TOP_N]
top_d    = top_d[::-1]                      # highest at the top
d_labels = [res_label(i + 1) for i in top_d]

head_v = avg_head[top_d]
acyl_v = avg_acyl[top_d]
ims_v  = avg_ims[top_d]
mat_v  = avg_mat[top_d]
tot_v  = head_v + acyl_v

y_d  = np.arange(len(top_d))
bw_d = 0.52

row_stripes(ax_hg, len(top_d))

x_max = max(tot_v.max() * 1.18, 40)

# Light track behind every bar
ax_hg.barh(y_d, np.full_like(tot_v, x_max), bw_d,
           color="#ECEFF1", edgecolor="none", zorder=1)

# Stacked segments — no white edges, flat and clean
ax_hg.barh(y_d, head_v, bw_d,
           color="#C03830", edgecolor="none", zorder=3,
           label="Headgroup (phosphate + glycerol)")
ax_hg.barh(y_d, acyl_v, bw_d, left=head_v,
           color="#2166AC", edgecolor="none", zorder=3,
           label="Acyl chains")

# Reference guides
ax_hg.axvline(30, color="#B0B0B0", lw=0.9, ls="--", alpha=0.95, zorder=4)
ax_hg.axvline(60, color="#D0D0D0", lw=0.9, ls="--", alpha=0.95, zorder=4)

# Leaflet marker + total, placed clear of the track
for yi, (iv, mv, tv) in enumerate(zip(ims_v, mat_v, tot_v)):
    if tv < 5:
        continue
    ax_hg.scatter(x_max * 1.035, yi,
                  marker="^" if iv >= mv else "v",
                  color="#F39C12" if iv >= mv else "#27AE60",
                  s=120, zorder=6, clip_on=False)
    ax_hg.text(x_max * 1.075, yi, f"{tv:.0f}%",
               ha="left", va="center",
               fontsize=16, fontweight="bold",
               color="black", clip_on=False)

# Segment labels inside their own segment
for yi, (hv, av) in enumerate(zip(head_v, acyl_v)):
    if hv > 8:
        ax_hg.text(hv / 2, yi, f"{hv:.0f}",
                   ha="center", va="center",
                   fontsize=13, color="black", fontweight="bold", zorder=5)
    if av > 8:
        ax_hg.text(hv + av / 2, yi, f"{av:.0f}",
                   ha="center", va="center",
                   fontsize=13, color="black", fontweight="bold", zorder=5)

ax_hg.set_yticks(y_d)
ax_hg.set_yticklabels(d_labels, fontsize=13, fontweight="bold")
ax_hg.set_xlabel("Contact occupancy (%)",
                 fontsize=FONT_LABEL+6.0, fontweight="bold")
ax_hg.tick_params(labelsize=FONT_TICK+6.0)
ax_hg.set_xlim(0, x_max)
ax_hg.set_ylim(-0.7, len(top_d) - 0.3)

leg_handles = [
    Patch(color="#C03830", label="Headgroup (phosphate + glycerol)"),
    Patch(color="#2166AC", label="Acyl chains"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#F39C12",
           markersize=14.5, label="Matrix-leaflet CL dominant"),
]
ax_hg.legend(handles=leg_handles, fontsize=15, frameon=False,
             loc="lower center", bbox_to_anchor=(0.5, 1.005),
             ncol=3, columnspacing=2.0, handletextpad=0.6)
ax_hg.set_title(
    "C)  Contact decomposition: phosphate/glycerol headgroup vs acyl chains",
    fontsize=FONT_TITLE+4.0, fontweight="bold", pad=34)
style_axis(ax_hg)


# ── Super title ───────────────────────────────────────────────────────
plt.suptitle(
    "Cardiolipin–BOU interactions\n"
    "[cutoff 6.0 Å]",
    fontsize=FONT_TITLE + 0.5, fontweight="bold", y=0.995)

plt.savefig("Cardiolipin_interactions.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("Cardiolipin_interactions.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: Cardiolipin_interactions.png / .pdf")
plt.show()
