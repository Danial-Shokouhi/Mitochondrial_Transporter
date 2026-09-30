#!/usr/bin/env python3
"""
Automated basin fingerprint comparison for both ligand-bound states of
the c-state transporter -- combined PCA across 3 replicas each.

Basin coordinates (combined FEL, 3 replicas each):

  cFAD-bound:
    GMin   PC1= -1.04 PC2= -2.38
    LMin1  PC1= 3.86  PC2= 1.24
    LMin2  PC1= -0.94 PC2= -0.57
    LMin3  PC1= -2.14 PC2= 1.72
    LMin4  PC1= -1.74 PC2= 3.30
    LMin5  PC1= -0.84 PC2= 0.52
    LMin6  PC1= 3.66  PC2= 0.15
    LMin7  PC1= -0.24 PC2= -3.71

  cPCar-bound:
    GMin   PC1= -8.42 PC2= 0.83
    LMin1  PC1= 4.08  PC2= 1.15
    LMin2  PC1= 2.87  PC2= 5.06
    LMin3  PC1= -2.77 PC2= -10.88
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances as mda_dist
except ImportError:
    sys.exit("Install MDAnalysis: conda install -c conda-forge mdanalysis")

np.random.seed(42)   # fixed seed for reproducible frame sampling in compute_fingerprint()

# SHARED CONFIGURATION

GRO_FILENAME = "step7_production.gro"
XTC_FILENAME = "step7_production_fit.xtc"

OUTPUT_INTERVAL_PS   = 100      # ps per frame (nstxout-compressed * dt)
MAX_FRAMES_PER_BASIN = 300      # max frames to sample per basin per replica
BASIN_RADIUS         = 0.60     # nm — radius around basin center


def sel(atoms):
    """Build MDAnalysis bynum selection from GROMACS 1-based indices."""
    return "bynum " + " ".join(str(a) for a in atoms)


# FEATURE DEFINITIONS

FAD_PROTEIN = [
    # FAD <-> ARG183 (strong charge-assisted H-bond over 1us)
    ("fad_ARG183_NH_O4",
     sel([4429]),              # FAD O4
     sel([2656, 2659])),       # ARG183 NH

    # FAD <-> ARG183 (occasional H-bond)
    ("fad_ARG183_NE_O14",
     sel([4439]),              # FAD O14
     sel([2653])),             # ARG183 NE

    # FAD <-> ARG280 phosphate salt bridge (strong over 1us)
    ("fad_ARG280_NH_PO4",
     sel([4426, 4435, 4436]),  # FAD PO4 (O1, O10, O11)
     sel([4097, 4100])),       # ARG280 NH

    # FAD <-> GLU184 (strong charge-assisted H-bond over 1us)
    ("fad_GLU184_OE_O14O15",
     sel([4439, 4440]),        # FAD O14, O15
     sel([2675, 2676])),       # GLU184 OE1, OE2

    # FAD <-> GLU126 / N3H (strong charge-assisted H-bond over 1us)
    ("fad_GLU126_OE_N3",
     sel([4392]),              # FAD N3
     sel([1814, 1815])),       # GLU126 OE1, OE2

    # FAD <-> LYS129 / O5 (strong charge-assisted H-bond over 1us)
    ("fad_LYS129_NZ_O5",
     sel([4430]),              # FAD O5
     sel([1872])),             # LYS129 NZ

    # FAD <-> TRP228 / NE1 (strong H-bond)
    ("fad_TRP228_NE_O10",
     sel([4435, 4436, 4426]),  # FAD O10, O11, O1
     sel([3269, 3270])),       # TRP228 NE1, HE1

    # FAD <-> GLN18 (strong H-bond over 1us)
    ("fad_GLN18_NE2_O8O11N9",
     sel([4433, 4436, 4398]),  # FAD O8, O11, N9
     sel([232, 233, 234, 235])),  # GLN18 OE1, NE2 region

    # FAD <-> GLY15 (almost strong H-bond over 1us)
    ("fad_GLY15_O_O6",
     sel([4431]),              # FAD O6
     sel([200])),              # GLY15 O
]

PCAR_PROTEIN = [
    # PCAR <-> ARG183             (occasional h-bond, mostly van der waals or CH-hydrogen)
    ("pcar_ARG183_NH_O",
     sel([4393, 4394, 4400, 4402]),  # PCAR O
     sel([2653, 2656, 2659])),       # ARG183 NH

    # PCAR <-> ARG280 salt bridge (strong salt bridge over 1us)
    ("pcar_ARG280_NH_O",
     sel([4393, 4394, 4400, 4402]),  # PCAR O
     sel([4094, 4097, 4100])),       # ARG280 NH

    # PCAR <-> GLU184             (strong salt bridge/CH-hydrogen over 1us)
    ("pcar_GLU184_OE_N",
     sel([4396, 4397, 4398, 4399]),  # PCAR N (CH3)3
     sel([2675, 2676])),             # GLU184 OE1, OE2

    # PCAR <-> GLU126             (occasional salt bridge/CH-hydrogen over 1us)
    ("pcar_GLU126_OE_N",
     sel([4396, 4397, 4398, 4399]),  # PCAR N (CH3)3
     sel([1814, 1815])),             # GLU126 OE1, OE2

    # PCAR <-> LYS129             (occasional/semi strong salt bridge over 1us)
    ("pcar_LYS129_NZ_O",
     sel([4393, 4394, 4400, 4402]),  # PCAR O
     sel([1872])),                   # LYS129 NZ

    # PCAR <-> TRP228             (short hbond)
    ("pcar_TRP228_NE_O",
     sel([4393, 4394, 4400, 4402]),  # PCAR O
     sel([3269, 3270])),             # TRP228 NE1, HE1

    # PCAR <-> GLN18              (strong charge-assisted hbond over 1us)
    ("pcar_GLN18_NE2_O",
     sel([4393, 4394, 4400, 4402]),  # PCAR O
     sel([232, 233, 234, 235])),     # GLN18 OE1, NE2 region

    # PCAR <-> LYS29              (short salt bridge)
    ("pcar_LYS29_NZ_O",
     sel([4393, 4394, 4400, 4402]),  # PCAR O
     sel([408])),                    # LYS29 NZ
]

# Shared between both conditions — these are gate residues on the
# transporter itself, not ligand-specific.
M_GATE = [
    # Main salt bridges
    ("mg_LYS29_ASP235",
     sel([408]),               # LYS29 NZ
     sel([3382, 3383])),       # ASP235 OD1, OD2

    ("mg_LYS238_GLU126",
     sel([3434]),              # LYS238 NZ
     sel([1814, 1815])),       # GLU126 OE1, OE2

    ("mg_LYS129_ASP26",
     sel([1872]),              # LYS129 NZ
     sel([355, 356])),         # ASP26 OD1, OD2

    # Occasional
    ("mg_LYS29_ASP26",
     sel([408]),               # LYS29 NZ
     sel([355, 356])),         # ASP26 OD1, OD2

    ("mg_LYS238_ASP235",
     sel([3434]),              # LYS238 NZ
     sel([3382, 3383])),       # ASP235 OD1, OD2

    ("mg_ARG183_ASP235",
     sel([2656, 2659]),        # ARG183 NH
     sel([3382, 3383])),       # ASP235 OD1, OD2

    ("mg_ARG183_GLU126",
     sel([2656, 2659]),        # ARG183 NH
     sel([1814, 1815])),       # GLU126 OE1, OE2
]

C_GATE = [
    # Salt bridges
    ("cg_ARG87_GLU196",
     sel([1268, 1271, 1265]),  # ARG87 NH1, NH2, NE
     sel([2843, 2844])),       # GLU196 OE1, OE2

    ("cg_LYS199_GLU293",
     sel([2893]),              # LYS199 NZ
     sel([4282, 4283])),       # GLU293 OE1, OE2

    ("cg_ARG296_ASP7",
     sel([4333, 4336, 4330]),  # ARG296 NH1, NH2, NE
     sel([106, 107])),         # ASP7 OD1, OD2

    ("cg_LYS199_GLU196",
     sel([2893]),              # LYS199 NZ
     sel([2843, 2844])),       # GLU196 OE1, OE2

    # H-bond braces
    ("cg_ARG87_GLN107",
     sel([1265, 1266]),        # ARG87 NE
     sel([1565])),             # GLN107 OE1

    ("cg_GLU293_TYR195",
     sel([4282, 4283]),        # GLU293 OE
     sel([2824, 2825])),       # TYR195 OH

    ("cg_ASP7_TYR292",
     sel([106, 107]),          # ASP7 OD
     sel([4263, 4264])),       # TYR292 OH
]


# PER-CONDITION CONFIGURATION

FAD_REP_DIRS = [
            ".../rep1",
            ".../rep2",
            ".../rep3",
]
PCAR_REP_DIRS = [
            ".../rep1",
            ".../rep2",
            ".../rep3",
]

CONDITIONS = {
    "fad": {
        "label":        "cFAD-bound",
        "out_prefix":   "FAD",
        "rep_dirs":     FAD_REP_DIRS,
        "rep_labels":   ["Rep1", "Rep2", "Rep3"],
        "pca_files":    [".../rep1",
                          ".../rep2",
                          ".../rep3"],
        "protein_feats":       FAD_PROTEIN,
        "protein_group_label": "FAD–Protein",
        "basins": [
            {"label": "GMin",  "pc1": -1.04, "pc2": -2.38},
            {"label": "LMin1", "pc1":  3.86, "pc2":  1.24},
            {"label": "LMin2", "pc1": -0.94, "pc2": -0.57},
            {"label": "LMin3", "pc1": -2.14, "pc2":  1.72},
            {"label": "LMin4", "pc1": -1.74, "pc2":  3.30},
            {"label": "LMin5", "pc1": -0.84, "pc2":  0.52},
            {"label": "LMin6", "pc1":  3.66, "pc2":  0.15},
            {"label": "LMin7", "pc1": -0.24, "pc2": -3.71},
        ],
        "heatmap_title": "cFAD Structural Fingerprint Heatmap",
    },
    "pcar": {
        "label":        "cPCar-bound",
        "out_prefix":   "PCAR",
        "rep_dirs":     PCAR_REP_DIRS,
        "rep_labels":   ["Rep1", "Rep2", "Rep3"],
        "pca_files":    [".../rep1",
                          ".../rep2",
                          ".../rep3"],
        "protein_feats":       PCAR_PROTEIN,
        "protein_group_label": "PCAR–Protein",
        "basins": [
            {"label": "GMin",  "pc1": -8.42, "pc2":   0.83},
            {"label": "LMin1", "pc1":  4.08, "pc2":   1.15},
            {"label": "LMin2", "pc1":  2.87, "pc2":   5.06},
            {"label": "LMin3", "pc1": -2.77, "pc2": -10.88},
        ],
        "heatmap_title": "cPCar Structural Fingerprint Heatmap",
    },
}


# CORE FUNCTIONS

def read_pca(filepath):
    pc1, pc2 = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith(("@", "#", "&")):
                continue
            s = line.split()
            if len(s) < 2:
                continue
            if len(s) == 2:
                pc1.append(float(s[0])); pc2.append(float(s[1]))
            else:
                pc1.append(float(s[1])); pc2.append(float(s[2]))
    return np.array(pc1), np.array(pc2)


def assign_frames(pc1, pc2, basins, radius):
    result = {b["label"]: [] for b in basins}
    for i in range(len(pc1)):
        for b in basins:
            d = np.sqrt((pc1[i]-b["pc1"])**2 + (pc2[i]-b["pc2"])**2)
            if d < radius:
                result[b["label"]].append(i)
                break
    return {k: np.array(v) for k, v in result.items()}


def min_dist_nm(u, sel1_str, sel2_str):
    """Minimum distance in nm between two atom selections."""
    try:
        ag1 = u.select_atoms(sel1_str)
        ag2 = u.select_atoms(sel2_str)
        if len(ag1) == 0 or len(ag2) == 0:
            return np.nan
        d = mda_dist.distance_array(ag1.positions, ag2.positions)
        return float(np.min(d)) / 10.0  # Angstrom -> nm
    except Exception:
        return np.nan


def compute_fingerprint(gro, xtc, frame_indices, features, dt_ps, max_f):
    if len(frame_indices) == 0:
        return {f[0]: (np.nan, np.nan) for f in features}

    sampled = np.random.choice(frame_indices,
                               min(max_f, len(frame_indices)),
                               replace=False)
    times_ps = set(int(f * dt_ps) for f in sampled)

    try:
        u = mda.Universe(gro, xtc)
    except Exception as e:
        print(f"    ERROR loading: {e}")
        return {f[0]: (np.nan, np.nan) for f in features}

    vals = {f[0]: [] for f in features}
    loaded = 0
    for ts in u.trajectory:
        if int(ts.time) not in times_ps:
            continue
        for fname, sel1, sel2 in features:
            vals[fname].append(min_dist_nm(u, sel1, sel2))
        loaded += 1
        if loaded >= len(times_ps):
            break

    result = {}
    for fname, vlist in vals.items():
        clean = [x for x in vlist if not np.isnan(x)]
        result[fname] = (float(np.mean(clean)), float(np.std(clean))) \
                        if clean else (np.nan, np.nan)
    return result


# PER-CONDITION PIPELINE
# Runs the full fingerprinting pipeline once per condition (FAD, PCAR).

def run_condition(cfg):
    prefix       = cfg["out_prefix"]
    basins       = cfg["basins"]
    basin_labels = [b["label"] for b in basins]
    GMIN         = basin_labels[0]

    ALL_FEATURES = cfg["protein_feats"] + M_GATE + C_GATE
    FEATURE_NAMES = [f[0] for f in ALL_FEATURES]
    GROUP_SIZES  = [len(cfg["protein_feats"]), len(M_GATE), len(C_GATE)]
    GROUP_LABELS = [cfg["protein_group_label"], "M-gate", "C-gate"]

    print("\n" + "=" * 65)
    print(f"STEP 3: BASIN FINGERPRINT COMPARISON — {cfg['label']}")
    print("=" * 65)
    print(f"Basins: {basin_labels}")
    print(f"Global minimum: {GMIN} at "
          f"PC1={basins[0]['pc1']:.3f}, PC2={basins[0]['pc2']:.3f}")

    #  Per-replica fingerprints 
    all_fp = {}   # {rep_label: {basin_label: {feat: (mean, std)}}}

    for rep_dir, rep_label, pca_file in zip(
            cfg["rep_dirs"], cfg["rep_labels"], cfg["pca_files"]):
        print(f"\n{''*65}")
        print(f"[{cfg['label']}] Replica: {rep_label}")

        gro = os.path.join(rep_dir, GRO_FILENAME)
        xtc = os.path.join(rep_dir, XTC_FILENAME)

        for f in [gro, xtc, pca_file]:
            if not os.path.exists(f):
                print(f"  SKIP: missing {f}")
                break
        else:
            pc1, pc2 = read_pca(pca_file)
            basin_frames = assign_frames(pc1, pc2, basins, BASIN_RADIUS)

            print(f"  Frames per basin:")
            for b in basins:
                n = len(basin_frames[b["label"]])
                print(f"    {b['label']:8}: {n:5} frames")

            all_fp[rep_label] = {}
            for b in basins:
                blabel = b["label"]
                frames = basin_frames[blabel]
                if len(frames) == 0:
                    all_fp[rep_label][blabel] = {
                        f[0]: (np.nan, np.nan) for f in ALL_FEATURES}
                    continue
                print(f"  Computing {blabel} ({len(frames)} frames)...",
                      end=" ", flush=True)
                all_fp[rep_label][blabel] = compute_fingerprint(
                    gro, xtc, frames, ALL_FEATURES,
                    OUTPUT_INTERVAL_PS, MAX_FRAMES_PER_BASIN)
                print("done")

    if not all_fp:
        print(f"  No data computed for {cfg['label']} — check file paths. Skipping.")
        return

    #  Save CSV 
    csv = ["Replica,Basin,Feature,Mean_nm,Std_nm"]
    for rep in all_fp:
        for blabel in all_fp[rep]:
            for fname in FEATURE_NAMES:
                m, s = all_fp[rep][blabel].get(fname, (np.nan, np.nan))
                csv.append(f"{rep},{blabel},{fname},{m:.4f},{s:.4f}")
    csv_path = f"{prefix}_Basin_Fingerprint_Summary.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv))
    print(f"\nSaved: {csv_path}")

    #  Statistical comparison 
    print(f"\n{''*65}")
    print(f"FEATURES DISTINGUISHING {GMIN} FROM EACH OTHER BASIN — {cfg['label']}")
    print(f"Criterion: |delta mean| > 0.10 nm  AND  all replicas agree")
    print("" * 65)

    other_basins = [b for b in basin_labels if b != GMIN]
    unique_feats = []

    for other in other_basins:
        print(f"\n  {GMIN} vs {other}:")
        print(f"  {'Feature':35} {'Delta_nm':9} {'Reps_agree':11} Direction")
        print("  " + "" * 65)
        for fname in FEATURE_NAMES:
            deltas = []
            for rep in all_fp:
                gm = all_fp[rep].get(GMIN, {}).get(fname, (np.nan,))[0]
                om = all_fp[rep].get(other, {}).get(fname, (np.nan,))[0]
                if not (np.isnan(gm) or np.isnan(om)):
                    deltas.append(gm - om)
            if not deltas:
                continue
            mean_d  = np.mean(deltas)
            n_agree = sum(1 for d in deltas
                          if np.sign(d) == np.sign(mean_d))
            direction = "SHORTER in GMin" if mean_d < 0 else "LONGER in GMin"
            notable = " ◄" if abs(mean_d) > 0.10 and n_agree == len(deltas) else ""
            print(f"  {fname:35} {mean_d:+9.3f} "
                  f"{n_agree}/{len(deltas):1}        {direction}{notable}")
            if abs(mean_d) > 0.10 and n_agree == len(deltas):
                unique_feats.append({
                    "cmp":  f"{GMIN}_vs_{other}",
                    "feat": fname,
                    "delta": mean_d,
                    "dir":  direction,
                    "reps": f"{n_agree}/{len(deltas)}",
                })

    # Save unique features
    uf_path = f"{prefix}_Basin_UniqueFeatures.txt"
    with open(uf_path, "w") as f:
        f.write(f"FEATURES CONSISTENTLY DIFFERENT IN {GMIN} — {cfg['label']}\n")
        f.write("Criterion: |delta| > 0.10 nm, all replicas agree\n")
        f.write("=" * 65 + "\n")
        if unique_feats:
            for uf in unique_feats:
                interp = (
                    "Contact STRONGER/SHORTER in GMin — likely STABILIZES it."
                    if uf["delta"] < 0 else
                    "Contact WEAKER/LONGER in GMin — its absence enables GMin."
                )
                f.write(f"\n[{uf['cmp']}]\n")
                f.write(f"  Feature:    {uf['feat']}\n")
                f.write(f"  Delta:      {uf['delta']:+.3f} nm\n")
                f.write(f"  Direction:  {uf['dir']}\n")
                f.write(f"  Replicas:   {uf['reps']}\n")
                f.write(f"  Meaning:    {interp}\n")
        else:
            f.write("\nNo features with |delta| > 0.10 nm consistent across "
                    "all replicas.\nTry increasing BASIN_RADIUS or "
                    "MAX_FRAMES_PER_BASIN.\n")
    print(f"\nSaved: {uf_path}")

    #  Heatmap 
    matrix = np.full((len(FEATURE_NAMES), len(basin_labels)), np.nan)
    for j, blabel in enumerate(basin_labels):
        for i, fname in enumerate(FEATURE_NAMES):
            rep_means = [
                all_fp[r][blabel][fname][0]
                for r in all_fp
                if blabel in all_fp[r]
                and not np.isnan(all_fp[r][blabel].get(fname, (np.nan,))[0])
            ]
            if rep_means:
                matrix[i, j] = np.mean(rep_means)

    fig_h = max(8, len(FEATURE_NAMES) * 0.45 + 2)
    fig_w = max(10, len(basin_labels) * 1.5 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin = max(0.0, np.nanmin(matrix))
    vmax = min(1.0, np.nanmax(matrix))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.40, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", norm=norm)

    for i in range(len(FEATURE_NAMES)):
        for j in range(len(basin_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                tc = "white" if (val > 0.70 or val < 0.20) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold")

    # Group separator lines
    cum = 0
    for gs, gl in zip(GROUP_SIZES, GROUP_LABELS):
        if cum > 0:
            ax.axhline(cum - 0.5, color="black", lw=2)
        cum += gs

    # Group labels — vertical, left of y-tick labels
    cum = 0
    for gs, gl in zip(GROUP_SIZES, GROUP_LABELS):
        mid_frac = (cum + gs / 2) / len(FEATURE_NAMES)
        y_frac = 1.0 - mid_frac
        ax.text(-0.28, y_frac, gl,
                ha="center", va="center",
                fontsize=14, color="navy", fontweight="bold",
                rotation=90,
                transform=ax.transAxes)
        cum += gs

    ax.set_xticks(range(len(basin_labels)))
    ax.set_xticklabels(basin_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(FEATURE_NAMES)))
    short = [f.replace("fad_", "").replace("pcar_", "")
              .replace("mg_", "").replace("cg_", "")
             for f in FEATURE_NAMES]
    ax.set_yticklabels(short, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean distance (nm) — averaged across 3 replicates",
                   fontsize=10)
    cbar.ax.axhline(0.40, color="black", lw=1.5, ls="--")

    ax.set_title(cfg["heatmap_title"], fontsize=11, pad=12)

    plt.tight_layout()
    plt.savefig(f"{prefix}_Basin_Heatmap.png", dpi=1000, bbox_inches="tight")
    plt.savefig(f"{prefix}_Basin_Heatmap.pdf", bbox_inches="tight")
    print(f"Saved: {prefix}_Basin_Heatmap.png and .pdf")
    plt.show()

    #  Consistency table 
    cons_path = f"{prefix}_Basin_Consistency.txt"
    with open(cons_path, "w") as f:
        f.write(f"Per-replica {GMIN} fingerprint values (nm) — {cfg['label']}\n")
        f.write("=" * 65 + "\n\n")
        for fname in FEATURE_NAMES:
            vals = [f"{r}: {all_fp[r].get(GMIN, {}).get(fname, (np.nan,))[0]:.3f}"
                    for r in all_fp]
            f.write(f"{fname}: {',  '.join(vals)}\n")
    print(f"Saved: {cons_path}")


# MAIN

def main():
    for cond_key, cfg in CONDITIONS.items():
        run_condition(cfg)

    print(f"\n{'='*65}")
    print("INTERPRETATION KEY")
    print("" * 65)
    print("  SHORTER in GMin  → interaction STRONGER/MORE FORMED in the")
    print("                     most stable state — likely stabilizes it")
    print("  LONGER  in GMin  → interaction WEAKER/BROKEN in the")
    print("                     most stable state — its absence may be")
    print("                     what ENABLES the most stable conformation")
    print("  All 3 reps agree → robust, publishable finding")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
