#!/usr/bin/env python3
"""
Reads FINAL_DECOMP_MMGBSA.dat and FINAL_RESULTS_MMGBSA.dat directly.
No H5 file needed. Works with gmx_MMPBSA v1.5.x without PyQt5.
Averages across all available replicas for FAD, P-carnitine, and
carnitine (m-state).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch as MPatch
import os, sys, csv, re

# File locations
DAT_FILES = {
    "FAD": [
        (".../rep1",
         ".../rep2"),
        (".../rep3",
         ".../rep1"),
        (".../rep2",
         ".../rep3"),
    ],
    "PCAR": [
        (".../rep1",
         ".../rep2"),
        (".../rep3",
         ".../rep1"),
        (".../rep2",
         ".../rep3"),
    ],
    "CAR": [
        (".../rep1",
         ".../rep2"),
        (".../rep3",
         ".../rep1"),
        (".../rep2",
         ".../rep3"),
    ],
}

OUT_DIR   = "MMGBSA_averaged_all"
THRESHOLD = 0.5   # kcal/mol

KEY_RESIDS = {
      10: "SER10", 11: "GLY11", 12: "THR12", 14: "GLY14", 15: "GLY15", 16: "ALA16",
      18: "GLN18", 19: "LEU19", 22: "GLY22", 26: "ASP26", 29: "LYS29", 72: "PRO72",
      73: "LEU73", 75: "THR75", 76: "VAL76", 77: "ALA77", 78: "ALA78", 79: "PHE79",
      80: "ASN80", 81: "ALA81", 83: "LEU83", 115: "GLY115", 118: "VAL118", 119: "SER119",
     122: "ALA122", 123: "CYS123", 125: "THR125", 126: "GLU126", 129: "LYS129", 180: "THR180",
     183: "ARG183", 184: "GLU184", 187: "GLY187", 188: "ASN188", 189: "ALA189", 191: "MET191",
     192: "PHE192", 224: "GLY224", 227: "PHE227", 228: "TRP228", 231: "VAL231", 235: "ASP235",
     238: "LYS238", 280: "ARG280", 284: "ALA284",288: "CYS288", 292: "TYR292", 33: "GLN33",
}
LIG_COLORS = {"FAD": "#C03830", "PCAR": "#707071", "CAR": "#1A1A1A"}
LIG_LABELS = {"FAD": "cFAD-bound", "PCAR": "cPCAR-bound", "CAR": "mCAR-bound"}
LIG_EDGE   = {"FAD": "#4f0501",  "PCAR": "black",        "CAR": "#444444"}

os.makedirs(OUT_DIR, exist_ok=True)


# Parse FINAL_RESULTS_MMGBSA.dat
def parse_results(path):
    energies = {}
    in_delta = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("Delta (Complex - Receptor - Ligand)"):
                in_delta = True
                continue
            if not in_delta:
                continue
            if s.startswith("---") or s == "":
                continue
            # Lines like: "ΔVDWAALS   -58.15   ..."
            # Strip the Δ prefix and parse
            s_clean = s.lstrip("Δ").lstrip("δ")
            parts = s_clean.split()
            if len(parts) >= 2:
                try:
                    key = parts[0]          # e.g. VDWAALS, EEL, EGB, ESURF, TOTAL
                    val = float(parts[1])   # Average column
                    energies[key] = val
                except ValueError:
                    pass
            # Stop after TOTAL
            if "TOTAL" in s:
                break
    return energies

# Parse FINAL_DECOMP_MMGBSA.dat
def parse_decomp(path):
    result = {}
    in_section = False
    header_count = 0

    with open(path, newline="") as f:
        for line in f:
            s = line.strip()
            if "Total Energy Decomposition:" in s:
                in_section   = True
                header_count = 0
                continue
            if in_section and ("Sidechain Energy" in s or
                               "Backbone Energy"  in s):
                break
            if not in_section or not s:
                continue
            # Skip 2 header rows
            if s.startswith("Residue,") or s.startswith(",Avg."):
                header_count += 1
                continue
            if header_count < 2:
                continue

            parts = s.split(",")
            if len(parts) < 17:
                continue
            res_key = parts[0].strip()
            if res_key.startswith("L::"):
                continue
            m = re.match(r"[RL]:(?:[A-Z]*):([A-Z]+):(\d+)", res_key)
            if not m:
                continue
            resname = m.group(1)
            resnum  = int(m.group(2))
            try:
                # col[1]=Internal, col[4]=vdW, col[7]=Elec,
                # col[10]=Polar,   col[13]=SASA, col[16]=TOTAL
                vdw   = float(parts[4])
                elec  = float(parts[7])
                polar = float(parts[10])
                sasa  = float(parts[13])
                total = float(parts[16])
                result[resnum] = {
                    "resname": resname,
                    "total":   total,
                    "vdw":     vdw,
                    "elec":    elec,
                    "polar":   polar,
                    "sasa":    sasa,
                }
            except (ValueError, IndexError):
                continue
    return result

# Average across replicas
def average_decomp(decomp_list):
    all_rn = set()
    for d in decomp_list: all_rn.update(d.keys())
    averaged = {}
    for rn in all_rn:
        vals = [d[rn] for d in decomp_list if rn in d]
        if not vals: continue
        tots = [v["total"] for v in vals]
        averaged[rn] = {
            "resname":   vals[0]["resname"],
            "total":     float(np.mean(tots)),
            "total_std": float(np.std(tots, ddof=1)) if len(tots)>1 else 0.0,
            "vdw":       float(np.mean([v["vdw"]   for v in vals])),
            "elec":      float(np.mean([v["elec"]  for v in vals])),
            "polar":     float(np.mean([v["polar"] for v in vals])),
            "sasa":      float(np.mean([v["sasa"]  for v in vals])),
        }
    return averaged


# Load all replicas
results = {}
for ligand, file_pairs in DAT_FILES.items():
    print(f"\n{'='*55}\n  Loading {ligand}...\n{'='*55}")
    decomp_list   = []
    energy_list   = []

    for i, (res_path, dec_path) in enumerate(file_pairs, 1):
        if not os.path.exists(res_path):
            print(f"  rep{i} RESULTS: MISSING — {res_path}")
        if not os.path.exists(dec_path):
            print(f"  rep{i} DECOMP:  MISSING — {dec_path}")
        if not (os.path.exists(res_path) and os.path.exists(dec_path)):
            continue

        print(f"  rep{i}: loading...")
        try:
            eng = parse_results(res_path)
            dec = parse_decomp(dec_path)
            print(f"    Energy keys: {list(eng.keys())}")
            print(f"    Decomp residues: {len(dec)}")
            if dec:
                energy_list.append(eng)
                decomp_list.append(dec)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if not decomp_list:
        print(f"  No usable data for {ligand}")
        continue

    results[ligand] = {
        "decomp":      average_decomp(decomp_list),
        "energies":    energy_list,
        "n_reps":      len(decomp_list),
    }
    print(f"  {ligand}: {len(results[ligand]['decomp'])} residues, "
          f"{results[ligand]['n_reps']} replicas ✓")

if not results:
    print("\nERROR: No data found. Check file paths above.")
    sys.exit(1)


# Print + save summary
print(f"\n{'='*55}\n  Binding Energy Summary\n{'='*55}")

# Key energy components to report
ENERGY_MAP = {
    "VDWAALS": "vdW",
    "EEL":     "Electrostatic",
    "EGB":     "GB polar",
    "ESURF":   "SASA nonpolar",
    "TOTAL":   "TOTAL ΔG binding",
}

summary_data = {}
with open(f"{OUT_DIR}/BindingEnergy_summary.txt", "w") as fw:
    fw.write("MM-GBSA Binding Free Energy Summary\n\n")
    for ligand, res in results.items():
        print(f"\n  {ligand} ({res['n_reps']} replicas):")
        fw.write(f"{''*40}\n{ligand} ({res['n_reps']} replicas)\n{''*40}\n")
        summary_data[ligand] = {}
        for key, label in ENERGY_MAP.items():
            vals = [e.get(key, np.nan) for e in res["energies"]]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                std  = np.std(vals, ddof=1) if len(vals)>1 else 0.0
                line = f"  {label:20} = {mean:8.2f} ± {std:.2f} kcal/mol"
                print(line); fw.write(line + "\n")
                summary_data[ligand][key] = (mean, std)
        fw.write("\n")

print(f"\nSaved: {OUT_DIR}/BindingEnergy_summary.txt")

# Save CSVs
for ligand, res in results.items():
    csv_path = f"{OUT_DIR}/EnergyDecomp_{ligand}_averaged.csv"
    with open(csv_path, "w") as fw:
        fw.write("ResNum,ResName,Total,Total_std,vdW,Elec,Polar,SASA\n")
        for rn, d in sorted(res["decomp"].items()):
            fw.write(f"{rn},{d['resname']},{d['total']:.4f},"
                     f"{d['total_std']:.4f},{d['vdw']:.4f},"
                     f"{d['elec']:.4f},{d['polar']:.4f},{d['sasa']:.4f}\n")
    print(f"Saved: {csv_path}")


# Figure 1: Overall ΔG components
comp_names = ["vdW", "Electrostatic", "GB polar", "SASA", "TOTAL ΔG"]
comp_keys  = ["VDWAALS", "EEL", "EGB", "ESURF", "TOTAL"]

fig1, ax = plt.subplots(figsize=(10, 5.5))
fig1.patch.set_facecolor("white")
ax.set_facecolor("white"); ax.grid(False)
ax.spines[["top","right"]].set_visible(False)

bw = 0.32; xb = np.arange(len(comp_names))
for gi, (ligand, res) in enumerate(results.items()):
    xp  = xb + (gi - (len(results)-1)/2) * bw
    col = LIG_COLORS[ligand]
    ms  = [summary_data[ligand].get(k, (0,0))[0] for k in comp_keys]
    ss  = [summary_data[ligand].get(k, (0,0))[1] for k in comp_keys]
    ax.bar(xp, ms, bw*0.88, yerr=ss, capsize=4, lw=2.5,
           color=col, alpha=0.95, edgecolor=LIG_EDGE[ligand],
           label=f"{LIG_LABELS[ligand]} (n={res['n_reps']})")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xb)
ax.set_xticklabels(comp_names, fontsize=15, fontweight="bold")
ax.set_ylabel("ΔG (kcal/mol)", fontsize=15, fontweight="bold")
ax.tick_params(labelsize=14)
ax.set_title("MM-GBSA Binding Free Energy Components\n",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/BindingEnergy_comparison.png",
            dpi=1000, bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT_DIR}/BindingEnergy_comparison.pdf",
            bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_DIR}/BindingEnergy_comparison.png / .pdf")
plt.show()


# Figure 2: Per-residue decomposition
def filter_sig(decomp, thr=THRESHOLD):
    return sorted([(k, v) for k, v in decomp.items()
                   if abs(v["total"]) >= thr],
                  key=lambda x: x[1]["total"])

# Colors — defined once, used identically for bars AND legend so they
# can never drift out of sync again.
COLOR_VDW          = "#2166AC"   # van der Waals
COLOR_ELEC         = "#D73027"   # Electrostatic
COLOR_POLAR        = "black"   # Polar solvation
COLOR_STABILIZING  = "#27AE60"   # Total panel: non-key, total < 0
COLOR_DESTABILIZE  = "#9112F3"   # Total panel: non-key, total > 0
COLOR_KEY_RESIDUE  = "#F39C12"   # Total panel + annotation text: key residue

n_lig = len(results)
fig2  = plt.figure(figsize=(12.5 * n_lig, 14))
fig2.patch.set_facecolor("white")
og = gridspec.GridSpec(1, n_lig, figure=fig2, wspace=0.55)

for li, (ligand, res) in enumerate(results.items()):
    ss = filter_sig(res["decomp"])
    if not ss:
        print(f"  No residues above threshold for {ligand}")
        continue

    resnums = [r for r, _ in ss]
    labels  = [f"{d['resname']}{r}" for r, d in ss]
    totals  = [d["total"]     for _, d in ss]
    stds_r  = [d["total_std"] for _, d in ss]
    vdws    = [d["vdw"]       for _, d in ss]
    elecs   = [d["elec"]      for _, d in ss]
    polars  = [d["polar"]     for _, d in ss]

    # Total-panel bar colors: key residue takes priority over sign,
    # otherwise green = stabilizing (total < 0), purple = destabilizing.
    bcolors = [
        COLOR_KEY_RESIDUE if r in KEY_RESIDS
        else COLOR_STABILIZING if t < 0
        else COLOR_DESTABILIZE
        for r, t in zip(resnums, totals)
    ]

    # Left (component) panel a bit wider than the Total panel so long
    # bars (ARG183/ARG280) don't run flush against the axis.
    ig = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=og[li],
        wspace=0.10, width_ratios=[2.0, 1.9])

    x = np.arange(len(ss))
    w = 0.65

    #  Left panel: stacked components 
    axL = fig2.add_subplot(ig[0])
    axL.set_facecolor("white")
    axL.grid(False)
    axL.spines[["top", "right"]].set_visible(False)

    axL.barh(x, vdws,  w, color=COLOR_VDW,  alpha=0.85, label="van der Waals")
    axL.barh(x, elecs, w, left=vdws,
             color=COLOR_ELEC, alpha=0.85, label="Electrostatic")
    cum = [v + e for v, e in zip(vdws, elecs)]
    axL.barh(x, polars, w, left=cum,
             color=COLOR_POLAR, alpha=0.85, label="Polar solvation")

    axL.axvline(0, color="black", lw=0.8)
    axL.set_yticks(x)
    axL.set_yticklabels(labels, fontsize=8, fontweight="bold")
    # Equal headroom top & bottom so no row (GLY15, TRP228, ARG280, ...)
    # sits flush against the axes edge and gets clipped.
    axL.set_ylim(-0.7, len(ss) + 0.5)

    axL.set_xlabel("ΔG contribution (kcal/mol)",
                   fontsize=11, fontweight="bold")
    axL.set_title(
        f"{LIG_LABELS[ligand]}\n"
        f"Per-residue Components (n={res['n_reps']} replicas)",
        fontsize=12, fontweight="bold")
    axL.grid(alpha=0.12, axis="x")

    # xlim must cover BOTH the stacked component sum AND the annotation
    # anchor point (each residue's *total*, which can diverge from the
    # component sum — this is what was clipping the TRP228 label).
    comp_left  = min(vdws[i] + elecs[i] for i in range(len(ss)))
    comp_right = max(v + e + p for v, e, p in zip(vdws, elecs, polars))
    left_edge  = min(0, comp_left, min(totals))
    right_edge = max(comp_right, max(totals))
    xr = right_edge - left_edge
    # Generous, asymmetric-safe padding: proportional term plus a fixed
    # floor big enough to fit the longest annotation text ("◄ RESNAME###").
    pad_left  = max(0.30 * xr, 30.0)
    pad_right = max(0.30 * xr, 30.0)
    axL.set_xlim(left_edge - pad_left, right_edge + pad_right)

    for i, (rn, d) in enumerate(ss):
        if rn in KEY_RESIDS:
            t = d["total"]
            axL.annotate(
                f"◄ {KEY_RESIDS[rn]}",
                xy=(t, i),
                xytext=(t - 1.2 if t < 0 else t + 0.2, i),
                fontsize=7, color=COLOR_KEY_RESIDUE,
                fontweight="bold", va="center",
                clip_on=True)

    #  Right panel: total ± SD 
    axR = fig2.add_subplot(ig[1])
    axR.set_facecolor("white")
    axR.grid(False)
    axR.spines[["top", "right"]].set_visible(False)

    axR.barh(x, totals, w,
             xerr=stds_r,
             color=bcolors, alpha=0.90,
             edgecolor="white", linewidth=0.3,
             error_kw={"elinewidth": 0.8,
                       "ecolor": "gray", "capsize": 2})
    axR.axvline(0, color="black", lw=0.8)

    # Wide padding so long value labels like "-244.7" land fully inside
    # the axis instead of sitting on the spine.
    x_range = max(totals) - min(totals)
    pad = max(24.0, 0.24 * x_range)
    axR.set_xlim(min(totals) - pad, max(totals) + pad)

    axR.set_yticks(x)
    axR.set_yticklabels([])
    axR.set_ylim(-0.7, len(ss) + 0.5)   # match axL row spacing exactly
    axR.set_xlabel("Total ΔG ± rep SD (kcal/mol)",
                   fontsize=11, fontweight="bold")
    axR.set_title("Total", fontsize=12, fontweight="bold")
    axR.grid(alpha=0.12, axis="x")

    # Value labels always sit just outside the bar tip (in the direction
    # the bar points), so they never crowd inside a bar or vanish.
    for i, (t, s) in enumerate(zip(totals, stds_r)):
        gap = 0.02 * (axR.get_xlim()[1] - axR.get_xlim()[0])
        if t < 0:
            xpos, ha = t - s - gap, "right"
        else:
            xpos, ha = t + s + gap, "left"
        axR.text(xpos, i, f"{t:.1f}",
                 ha=ha, va="center",
                 color="black", fontsize=7, fontweight="bold",
                 clip_on=True)

# Shared legend
component_handles = [
    MPatch(color=COLOR_VDW,   label="van der Waals"),
    MPatch(color=COLOR_ELEC,  label="Electrostatic"),
    MPatch(color=COLOR_POLAR, label="Polar solvation"),
]
class_handles = [
    MPatch(color=COLOR_STABILIZING, label="Stabilizing"),
    MPatch(color=COLOR_DESTABILIZE, label="Destabilizing"),
    MPatch(color=COLOR_KEY_RESIDUE, label="Key residue"),
]

# Title on top, legend directly underneath it
plt.suptitle(
    f"MM-GBSA Per-Residue Energy Decomposition\n"
    f"(|ΔG| > {THRESHOLD} kcal/mol)",
    fontsize=14, fontweight="bold", y=0.985)

fig2.legend(
    handles=component_handles + class_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=6, fontsize=10,
    frameon=False,
    columnspacing=1.4, handlelength=1.8)

plt.subplots_adjust(top=0.86)
plt.savefig(f"{OUT_DIR}/EnergyDecomp_FAD_PCAR_CAR.png",
            dpi=1000, bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT_DIR}/EnergyDecomp_FAD_PCAR_CAR.pdf",
            bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_DIR}/EnergyDecomp_FAD_PCAR_CAR.png / .pdf")
plt.show()
