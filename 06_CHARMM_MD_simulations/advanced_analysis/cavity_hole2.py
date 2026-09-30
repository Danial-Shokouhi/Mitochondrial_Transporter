#!/usr/bin/env python3
"""
Pore/cavity radius analysis using HOLE2
Three conditions: cFAD-bound | cPCar-bound | Apo c-state
3 replicas × 1µs each.
All profiles cached as .npy for fast re-plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, sys, subprocess, tempfile, shutil, warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("Install MDAnalysis: conda install -c conda-forge mdanalysis")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

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

GRO = "step7_production.gro"
XTC = "step7_production_fit.xtc"

HOLE_EXE = shutil.which("hole") or shutil.which("hole2")
if HOLE_EXE is None:
    sys.exit("HOLE2 not found. Activate hole2 conda environment.")

HOLE_RADIUS = ".../hole2/simple.rad"

# Seed point: auto-calculated from CA centroid of binding-site residues
# (overridden at runtime per trajectory — see get_seed_from_pdb())
SEED_X, SEED_Y, SEED_Z = None, None, None   # set at runtime
CVECT_X, CVECT_Y, CVECT_Z = 0.0, 0.0, 1.0
HOLE_ENDRAD = 50.0

# Binding-site residues used to calculate seed (1-based)
SEED_RESIDS = [18, 79, 183, 184, 228, 280]


def get_seed_from_pdb(pdb_path):
    """
    Calculate seed point as centroid of CA atoms from SEED_RESIDS.
    Reads directly from PDB file — no MDAnalysis needed.
    """
    ca_coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            atom_name = line[12:16].strip()
            if atom_name != "CA": continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError: continue
            if resnum not in SEED_RESIDS: continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coords.append((x, y, z))
            except ValueError: continue
    if not ca_coords:
        return None, None, None
    arr = np.array(ca_coords)
    return float(arr[:,0].mean()), float(arr[:,1].mean()), float(arr[:,2].mean())

# Membrane Z range (Å)
MEMBRANE_Z_MIN = 15.0
MEMBRANE_Z_MAX = 65.0
MEMBRANE_Z_CTR = 40.0

STRIDE = 100   # every Nth frame

Z_PLOT_MIN = MEMBRANE_Z_MIN - 5
Z_PLOT_MAX = MEMBRANE_Z_MAX + 5
Z_GRID = np.linspace(Z_PLOT_MIN, Z_PLOT_MAX, 300)

# Condition styling
CONDITIONS = {
    "FAD":  {"color": "#C03830", "ls": "-",  "label": "cFAD-bound",
             "cmap": "Reds",    "dirs": FAD_DIRS},
    "PCAR": {"color": "black", "ls": "-", "label": "cPCar-bound",
             "cmap": "Greys",   "dirs": PCAR_DIRS},
    "APO":  {"color": "#2166AC", "ls": "--",  "label": "Apo-cBOU",
             "cmap": "Blues",   "dirs": APO_DIRS},
}

# Key residue Z positions (Å) and their role
KEY_RESIDUES = [
    # ── Cytosolic gate (c-gate) — IMS side, low Z ─────────────────────
    ("LYS199",  32.0, "c-gate", "#E67E22"),
    ("ARG296",  32.4, "c-gate", "#E67E22"),
    ("GLU196",  33.6, "c-gate", "#E67E22"),
    ("GLU293",  33.7, "c-gate", "#E67E22"),
    ("ASP7",    34.0, "c-gate", "#E67E22"),
    ("ARG87",   36.9, "c-gate", "#E67E22"),
    # ── Substrate binding site — mid-membrane ─────────────────────────
    ("GLU184",  49.3, "site",   "#F1C40F"),
    ("TRP228",  49.9, "site",   "#F1C40F"),
    ("ARG280",  50.8, "site",   "#F1C40F"),
    ("ARG183",  52.2, "site",   "#F1C40F"),
    # ── Matrix gate (m-gate) — matrix side, high Z ────────────────────
    ("ASP26",   56.6, "m-gate", "#FF81C0"),
    ("GLU126",  57.0, "m-gate", "#FF81C0"),
    ("ASP235",  57.4, "m-gate", "#FF81C0"),
    ("LYS129",  59.1, "m-gate", "#FF81C0"),
    ("LYS29",   60.1, "m-gate", "#FF81C0"),
    ("LYS238",  60.3, "m-gate", "#FF81C0"),
]


# ═══════════════════════════════════════════════════════════════════════
# HOLE2 RUNNER (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════

def write_hole_input(pdb_file, work_dir, sx, sy, sz):
    inp = os.path.join(work_dir, "hole.inp")
    sph = os.path.join(work_dir, "hole.sph")
    with open(inp, "w") as f:
        f.write(
            f"RADIUS {HOLE_RADIUS}\n"
            f"SPHPDB {sph}\n"
            f"COORD {pdb_file}\n"
            f"CVECT {CVECT_X} {CVECT_Y} {CVECT_Z}\n"
            f"CPOINT {sx} {sy} {sz}\n"
            f"ENDRAD {HOLE_ENDRAD}\n")
    return inp, sph

def parse_hole_output(out_text):
    z_vals, r_vals = [], []
    for line in out_text.split("\n"):
        if "(sampled)" not in line: continue
        parts = line.split()
        if len(parts) < 2: continue
        try:
            z = float(parts[0]); r = float(parts[1])
            if r > 0: z_vals.append(z); r_vals.append(r)
        except (ValueError, IndexError): continue
    return (np.array(z_vals), np.array(r_vals)) if z_vals else (None, None)

def run_hole2_on_pdb(pdb_file, sx, sy, sz):
    work_dir = tempfile.mkdtemp(prefix="hole_")
    try:
        inp_file, _ = write_hole_input(pdb_file, work_dir, sx, sy, sz)
        result = subprocess.run(
            [HOLE_EXE], stdin=open(inp_file),
            capture_output=True, text=True,
            cwd=work_dir, timeout=60)
        if "sampled" not in result.stdout.lower():
            result = subprocess.run(
                f"{HOLE_EXE} < {inp_file}",
                capture_output=True, text=True,
                cwd=work_dir, timeout=60, shell=True)
        if "sampled" not in result.stdout.lower(): return None, None
        return parse_hole_output(result.stdout)
    except Exception: return None, None
    finally: shutil.rmtree(work_dir, ignore_errors=True)

def interp_profile(z_raw, r_raw, z_grid):
    if z_raw is None or len(z_raw) < 3:
        return np.full(len(z_grid), np.nan)
    order = np.argsort(z_raw)
    return np.interp(z_grid, z_raw[order], r_raw[order],
                     left=np.nan, right=np.nan)

def process_trajectory(gro_path, xtc_path, label, cache_npy):
    """
    Extract protein frames using GROMACS (more reliable PDB format for HOLE2)
    then run HOLE2 on each frame.
    """
    if os.path.exists(cache_npy):
        arr = np.load(cache_npy)
        # Validate cache — reject if all NaN
        if not np.isnan(arr).all():
            print(f"    Cache: {os.path.basename(cache_npy)}")
            return arr
        else:
            print(f"    Cache {os.path.basename(cache_npy)} is all-NaN — recomputing")
            os.remove(cache_npy)

    print(f"    Computing: {label}")
    import shlex

    # Find TPR (same dir as GRO or parent dir)
    tpr_path = gro_path.replace(".gro", ".tpr")
    if not os.path.exists(tpr_path):
        # Try parent directory
        tpr_path = os.path.join(
            os.path.dirname(os.path.dirname(gro_path)),
            "step7_production.tpr")
    if not os.path.exists(tpr_path):
        print(f"    WARNING: TPR not found for {label}, using GRO as topology")
        tpr_path = gro_path

    # Get total frames
    u_check = mda.Universe(gro_path, xtc_path)
    total_frames = len(u_check.trajectory)
    frame_indices = list(range(0, total_frames, STRIDE))
    print(f"    {total_frames} total frames → {len(frame_indices)} to analyse")

    work_dir = tempfile.mkdtemp(prefix="hole_traj_")
    profiles  = []

    try:
        for frame_num, frame_idx in enumerate(frame_indices):
            time_ps = frame_idx * 100   # 100 ps per frame
            tmp_pdb = os.path.join(work_dir, f"frame_{frame_num}.pdb")

            # Extract single protein frame — exactly as manual test:
            # echo "1" | gmx trjconv -s TPR -f XTC -o OUT -dump TIME
            result = subprocess.run(
                ["gmx", "trjconv",
                 "-s", tpr_path,
                 "-f", xtc_path,
                 "-o", tmp_pdb,
                 "-dump", str(time_ps)],
                input="1\n",           # select group 1 = Protein
                capture_output=True,
                text=True,
                timeout=60)

            if not os.path.exists(tmp_pdb) or os.path.getsize(tmp_pdb) == 0:
                if frame_num == 0:
                    print(f"    WARNING: GROMACS trjconv failed")
                    print(f"      TPR: {tpr_path}")
                    print(f"      stderr: {result.stderr[-300:]}")
                continue

            # Auto-calculate seed from first frame CA centroid
            if frame_num == 0:
                sx, sy, sz = get_seed_from_pdb(tmp_pdb)
                if sx is None:
                    print(f"    WARNING: could not calculate seed — "
                          f"using fallback (53,48.8,44.9)")
                    sx, sy, sz = 53.0, 48.8, 44.9
                else:
                    print(f"    Seed auto-calculated: "
                          f"({sx:.1f}, {sy:.1f}, {sz:.1f}) Å "
                          f"from resids {SEED_RESIDS}")

            z_raw, r_raw = run_hole2_on_pdb(tmp_pdb, sx, sy, sz)

            # Debug first frame
            if frame_num == 0:
                if z_raw is not None and len(z_raw) > 0:
                    print(f"    Frame 0: HOLE2 OK — "
                          f"{len(z_raw)} points, "
                          f"Z=[{z_raw.min():.1f}, {z_raw.max():.1f}] Å")
                else:
                    print(f"    Frame 0: HOLE2 NO points — "
                          f"seed=({sx:.1f},{sy:.1f},{sz:.1f})")
                    with open(tmp_pdb) as pf:
                        lines = [l for l in pf.readlines()
                                 if l.startswith("ATOM")][:3]
                    for l in lines: print(f"      {l.rstrip()}")

            profiles.append(interp_profile(z_raw, r_raw, Z_GRID))
            if os.path.exists(tmp_pdb): os.remove(tmp_pdb)

            if (frame_num+1) % 10 == 0:
                valid = sum(1 for p in profiles
                            if not np.isnan(p).all())
                print(f"    {label}: {frame_num+1}/{len(frame_indices)} "
                      f"frames ({valid} valid)")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if not profiles:
        return None

    arr     = np.array(profiles)
    n_valid = np.sum(~np.isnan(arr).all(axis=1))
    print(f"    {label}: {n_valid}/{len(profiles)} frames had valid HOLE2 output")

    if n_valid == 0:
        print(f"    ERROR: All HOLE2 runs failed for {label}")
        print(f"    Check: seed point ({SEED_X},{SEED_Y},{SEED_Z}) Å inside protein?")
        print(f"    Check: HOLE2 path correct? {HOLE_EXE}")
        return None

    np.save(cache_npy, arr)
    return arr

def stack_and_average(arrays, label):
    valid = [a for a in arrays if a is not None]
    if not valid: sys.exit(f"No data for {label}")
    combined = np.concatenate(valid, axis=0)
    print(f"  {label}: {combined.shape[0]} frames from {len(valid)} replicas")
    return (np.nanmean(combined, axis=0),
            np.nanstd(combined, axis=0),
            combined)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main radius profile (3-condition + 2 difference panels)
# ═══════════════════════════════════════════════════════════════════════

def fig_radius_profile(profiles, z_grid):
    """
    5-panel figure:
    Left (tall): all 3 mean profiles overlaid
    Middle-top: ΔDCCM(FAD−APO) profile
    Middle-bot: Δ(PCAR−APO) profile
    Right-top:  Δ(FAD−PCAR) profile
    Right-bot:  Occupancy (% frames with r > 1.4 Å) within membrane
    """
    means = {c: profiles[c]["mean"] for c in CONDITIONS}
    stds  = {c: profiles[c]["std"]  for c in CONDITIONS}

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[1, 1],
                            hspace=0.38, wspace=0.32)

    # ── Left column: main profile (spans both rows) ──────────────────
    ax_main = fig.add_subplot(gs[:, 0])

    # Membrane shading
    ax_main.axhspan(MEMBRANE_Z_MIN, MEMBRANE_Z_MAX,
                    color="#FAD7A0", alpha=0.22, zorder=0,
                    label="Lipid bilayer")
    ax_main.axhline(MEMBRANE_Z_CTR, color="gray",
                    lw=0.8, ls=":", alpha=0.5)

    # Water radius threshold
    ax_main.axvline(1.4, color="black", lw=0.8, ls="--", alpha=0.45,
                    label="Water radius\n(1.4 Å)")

    for cond, cfg in CONDITIONS.items():
        m, s = means[cond], stds[cond]
        ax_main.fill_betweenx(z_grid, m-s, m+s,
                               color=cfg["color"], alpha=0.18)
        ax_main.plot(m, z_grid, color=cfg["color"],
                     lw=2.2, ls=cfg["ls"],
                     label=cfg["label"], zorder=4)

    # Residue Z annotations
    xlim_r = ax_main.get_xlim()[1]
    sorted_res = sorted(KEY_RESIDUES, key=lambda x: x[1])
    prev_z, stagger = -999, 0
    for name, z_pos, role, col in sorted_res:
        ax_main.axhline(z_pos, color=col, lw=0.7, ls="--", alpha=0.55)
        stagger = (stagger + 1) % 2 if abs(z_pos - prev_z) < 2.2 else 0
        ax_main.text(xlim_r + 0.12 + stagger * 1.05, z_pos, f" {name}",
                     va="center", fontsize=9.5,
                     color=col, fontweight="bold")
        prev_z = z_pos

    ax_main.set_xlabel("Pore radius (Å)", fontsize=16, fontweight="bold")
    ax_main.set_ylabel("Z position (Å)", fontsize=16, fontweight="bold")
    ax_main.set_title("Cavity radius profile",
                      fontsize=16, fontweight="bold")
    ax_main.set_ylim(Z_PLOT_MIN, Z_PLOT_MAX)
    ax_main.set_xlim(left=0)
    ax_main.tick_params(axis="y", labelsize=16)
    ax_main.tick_params(axis="x", labelsize=16)
    ax_main.text(-0.105, 0.97, "Matrix", transform=ax_main.transAxes,
                 fontsize=16, va="top", ha="center", color="navy", fontweight="bold")
    ax_main.text(-0.105, 0.03, "IMS", transform=ax_main.transAxes,
                 fontsize=16, va="bottom", ha="center", color="navy", fontweight="bold")
    # Legend moved to a figure-level horizontal strip at the top
    ax_main.set_facecolor("white"); ax_main.grid(False)
    ax_main.spines[["top","right"]].set_visible(False)

    # ── Difference panels ────────────────────────────────────────────
    diff_pairs = [
        ("FAD","APO",  "Δ Radius\n(FAD − Apo)",  0, 1, "#C03830"),
        ("PCAR","APO", "Δ Radius\n(PCar − Apo)", 1, 1, "black"),
        ("FAD","PCAR", "Δ Radius\n(FAD − PCar)", 0, 2, "purple"),
    ]
    for (c1, c2, title, row, col, col_c) in diff_pairs:
        ax = fig.add_subplot(gs[row, col])
        diff     = means[c1] - means[c2]
        diff_std = np.sqrt(stds[c1]**2 + stds[c2]**2)
        ax.fill_betweenx(z_grid, diff-diff_std, diff+diff_std,
                          color=col_c, alpha=0.18)
        ax.plot(diff, z_grid, color=col_c, lw=2.0)
        ax.axvline(0, color="black", lw=0.8)
        ax.axhspan(MEMBRANE_Z_MIN, MEMBRANE_Z_MAX,
                   color="#FAD7A0", alpha=0.22, zorder=0)
        ax.axhline(MEMBRANE_Z_CTR, color="gray",
                   lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel("ΔRadius (Å)", fontsize=16, fontweight="bold")
        ax.set_ylim(Z_PLOT_MIN, Z_PLOT_MAX)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)

    # Annotate direction
       # max_z = z_grid[np.nanargmax(diff)]
       # ax.annotate(f"Wider in\n{CONDITIONS[c1]['label'].split('-')[0]}",
       #             xy=(np.nanmax(diff)*0.6, max_z),
       #             fontsize=7.5, color=col_c, va="center", ha="left")
    # ── Occupancy panel (bottom right) ───────────────────────────────
    ax_occ = fig.add_subplot(gs[1, 2])
    mem_mask = (z_grid >= MEMBRANE_Z_MIN) & (z_grid <= MEMBRANE_Z_MAX)
    WATER_R = 1.4
    for cond, cfg in CONDITIONS.items():
        all_arr = profiles[cond]["all"]
        occ = np.mean(all_arr[:, mem_mask] > WATER_R, axis=0) * 100
        ax_occ.plot(occ, z_grid[mem_mask],
                    color=cfg["color"], lw=2.0, ls=cfg["ls"],
                    label=cfg["label"])
    ax_occ.axhline(MEMBRANE_Z_CTR, color="gray",
                   lw=0.8, ls=":", alpha=0.5)
    ax_occ.set_xlabel("% frames open\n(r > 1.4 Å)", fontsize=16,
                       fontweight="bold")
    ax_occ.set_ylim(MEMBRANE_Z_MIN, MEMBRANE_Z_MAX)
    ax_occ.set_title("Cavity openness\n(% open frames)",
                      fontsize=16, fontweight="bold")
    ax_occ.legend(fontsize=11, frameon=False)
    ax_occ.set_facecolor("white"); ax_occ.grid(False)
    ax_occ.spines[["top","right"]].set_visible(False)
    ax_occ.tick_params(axis="y", labelsize=16)
    ax_occ.tick_params(axis="x", labelsize=16)
    # ── Figure-level horizontal legend across the full width ─────────
    fig_leg_handles = [
        Patch(facecolor="#FAD7A0", alpha=0.45, edgecolor="none",
              label="Lipid bilayer"),
        Line2D([0],[0], color="black", lw=1.2, ls="--", alpha=0.7,
               label="Water radius (1.4 Å)"),
    ]
    for cond, cfg in CONDITIONS.items():
        fig_leg_handles.append(
            Line2D([0],[0], color=cfg["color"], lw=2.4,
                   ls=cfg["ls"], label=cfg["label"]))
    fig_leg_handles += [
        Line2D([0],[0], color="#E67E22", lw=1.4, ls="--",
               label="c-gate residues"),
        Line2D([0],[0], color="#F1C40F", lw=1.4, ls="--",
               label="Binding-site residues"),
        Line2D([0],[0], color="#FF81C0", lw=1.4, ls="--",
               label="m-gate residues"),
    ]
    fig.legend(handles=fig_leg_handles,
               loc="upper center", ncol=len(fig_leg_handles),
               bbox_to_anchor=(0.5, 0.97),
               fontsize=13, frameon=False,
               columnspacing=1.4, handlelength=2.0)

    plt.suptitle(
        "Cavity radius analysis in C-state",
        fontsize=14, fontweight="bold")
    plt.savefig("Cavity_radius_profile.png", dpi=1000,
                bbox_inches="tight", facecolor="white")
    plt.savefig("Cavity_radius_profile.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: Cavity_radius_profile.png / .pdf")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Radius heatmap over Z × time (one per condition)
# ═══════════════════════════════════════════════════════════════════════

def fig_cavity_heatmap(per_rep_arrays, z_grid):
    """
    For each condition show radius as a Z × time heatmap.
    3 rows (conditions) × 3 columns (replicas).
    Instantly shows when and where the cavity opens.
    """
    mem_mask = (z_grid >= MEMBRANE_Z_MIN) & (z_grid <= MEMBRANE_Z_MAX)
    z_mem    = z_grid[mem_mask]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.patch.set_facecolor("white")

    cond_order = ["FAD", "PCAR", "APO"]
    norm = mcolors.Normalize(vmin=0, vmax=8)
    cmap_main = "viridis"

    for row, cond in enumerate(cond_order):
        reps = per_rep_arrays[cond]
        for col in range(3):
            ax = axes[row, col]
            if reps[col] is None:
                ax.text(0.5, 0.5, "No data", ha="center",
                        transform=ax.transAxes)
                continue
            arr = reps[col][:, mem_mask]   # (n_frames, n_z_mem)
            n_frames = arr.shape[0]
            t = np.linspace(0, 1000, n_frames)

            im = ax.pcolormesh(t, z_mem, arr.T,
                                cmap=cmap_main, norm=norm,
                                shading="auto", rasterized=True)

            # Water radius contour
            try:
                ax.contour(t, z_mem, arr.T,
                           levels=[1.4], colors="white",
                           linewidths=1.15, alpha=1.0)
            except Exception:
                pass

            ax.axhline(MEMBRANE_Z_CTR, color="white",
                       lw=0.9, ls=":", alpha=1.0)
            ax.set_xlabel("Time (ns)", fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Z (Å)", fontsize=16, fontweight="bold")
                ax.text(-0.22, 0.5,
                        CONDITIONS[cond]["label"],
                        transform=ax.transAxes, ha="center",
                        va="center", fontsize=16,
                        fontweight="bold",
                        color=CONDITIONS[cond]["color"],
                        rotation=90)
            ax.set_title(f"Rep {col+1}", fontsize=16, fontweight="bold")
            ax.tick_params(labelsize=13)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    sm = plt.cm.ScalarMappable(cmap=cmap_main, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Pore radius (Å)", fontsize=14, fontweight="bold")
    cb.ax.tick_params(labelsize=14)
    cb.ax.axhline(1.4, color="white", lw=1.5, ls="--")
    cb.ax.text(2.2, 1.4, "Water\nradius", fontsize=14, color="white",
               va="center")

    plt.suptitle(
        "Cavity radius heatmap [Z position × Time]\n"
        "(white contour = 1.4 Å water threshold)",
        fontsize=15, fontweight="bold")
    plt.subplots_adjust(left=0.10, right=0.90, hspace=0.38, wspace=0.28)
    plt.savefig("Cavity_heatmap.png", dpi=1000,
                bbox_inches="tight", facecolor="white")
    plt.savefig("Cavity_heatmap.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: Cavity_heatmap.png / .pdf")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Bottleneck radius timeseries
# ═══════════════════════════════════════════════════════════════════════

def fig_bottleneck(per_rep_arrays, z_grid):
    """
    3 panels (one per replica), each with 3 lines (FAD/PCAR/APO).
    Shows minimum pore radius within membrane vs time.
    """
    mem_mask = (z_grid >= MEMBRANE_Z_MIN) & (z_grid <= MEMBRANE_Z_MAX)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.patch.set_facecolor("white")

    for rep_idx in range(3):
        ax = axes[rep_idx]
        for cond, cfg in CONDITIONS.items():
            arr = per_rep_arrays[cond][rep_idx]
            if arr is None: continue
            bot = np.nanmin(arr[:, mem_mask], axis=1)
            n   = len(bot)
            t   = np.linspace(0, 1000, n)
            ax.plot(t, bot, color=cfg["color"],
                    lw=0.8, alpha=0.40, ls=cfg["ls"])
            win = max(1, n // 20)
            sm  = np.convolve(bot, np.ones(win)/win, mode="same")
            ax.plot(t, sm, color=cfg["color"],
                    lw=2.2, ls=cfg["ls"],
                    label=f"{cfg['label']} (mean={np.nanmean(bot):.2f} Å)")
            # Shaded std band using rolling std
            from scipy.ndimage import uniform_filter1d
            roll_std = np.array([
                np.nanstd(bot[max(0,i-win):i+win])
                for i in range(n)])
            ax.fill_between(t, sm-roll_std, sm+roll_std,
                             color=cfg["color"], alpha=0.10)

        ax.axhline(1.4, color="black", lw=0.8,
                   ls="--", alpha=0.45, label="Water radius (1.4 Å)")
        ax.set_ylabel("Min pore radius (Å)",
                      fontsize=14, fontweight="bold")
        ax.set_title(f"Replica {rep_idx+1}",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=12, loc="upper right",
                  frameon=False, ncol=2)
        ax.tick_params(axis="y", labelsize=13)
        ax.tick_params(axis="x", labelsize=13)
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time (ns)", fontsize=15, fontweight="bold")

    plt.suptitle(
        "Cavity bottleneck radius vs time\n"
        "(Minimum pore radius within membrane)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Cavity_bottleneck.png", dpi=1000,
                bbox_inches="tight", facecolor="white")
    plt.savefig("Cavity_bottleneck.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: Cavity_bottleneck.png / .pdf")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("CAVITY VOLUME ANALYSIS — HOLE2")
    print("cFAD | cPCar | Apo-cBOU  —  3 × 1µs each")
    print(f"HOLE2: {HOLE_EXE}")
    print("="*65)

    # ── Process all replicas for all conditions ────────────────────────
    per_rep_arrays = {c: [] for c in CONDITIONS}

    for cond, cfg in CONDITIONS.items():
        print(f"\n── {cfg['label']} ──")
        for rep_idx, d in enumerate(cfg["dirs"], 1):
            gp = os.path.join(d, GRO)
            xp = os.path.join(d, XTC)
            cache = f"hole2_profiles_{cond}_rep{rep_idx}.npy"
            if os.path.exists(gp) and os.path.exists(xp):
                arr = process_trajectory(gp, xp,
                                          f"{cond} rep{rep_idx}", cache)
            else:
                print(f"  SKIP {cond} rep{rep_idx}: files missing")
                arr = None
            per_rep_arrays[cond].append(arr)

    # ── Average across replicas ────────────────────────────────────────
    print("\nAveraging across replicas...")
    profiles = {}
    for cond in CONDITIONS:
        m, s, all_arr = stack_and_average(per_rep_arrays[cond], cond)
        profiles[cond] = {"mean": m, "std": s, "all": all_arr}

    # ── Statistics ────────────────────────────────────────────────────
    mem_mask = (Z_GRID >= MEMBRANE_Z_MIN) & (Z_GRID <= MEMBRANE_Z_MAX)
    print("\nMembrane-region statistics:")
    for cond in CONDITIONS:
        m = profiles[cond]["mean"]
        s = profiles[cond]["std"]
        print(f"  {cond}: mean={np.nanmean(m[mem_mask]):.2f} Å  "
              f"bottleneck={np.nanmin(m[mem_mask]):.2f} Å  "
              f"std={np.nanmean(s[mem_mask]):.2f} Å")

    # ── Save CSV ──────────────────────────────────────────────────────
    with open("Cavity_summary.csv", "w") as f:
        f.write("Z_Ang,R_FAD,SD_FAD,R_PCAR,SD_PCAR,R_APO,SD_APO,"
                "dR_FAD_APO,dR_PCAR_APO,dR_FAD_PCAR\n")
        for i, z in enumerate(Z_GRID):
            rf  = profiles["FAD"]["mean"][i]
            sf  = profiles["FAD"]["std"][i]
            rp  = profiles["PCAR"]["mean"][i]
            sp  = profiles["PCAR"]["std"][i]
            ra  = profiles["APO"]["mean"][i]
            sa  = profiles["APO"]["std"][i]
            f.write(f"{z:.2f},{rf:.4f},{sf:.4f},"
                    f"{rp:.4f},{sp:.4f},"
                    f"{ra:.4f},{sa:.4f},"
                    f"{rf-ra:.4f},{rp-ra:.4f},{rf-rp:.4f}\n")
    print("  Saved: Cavity_summary.csv")

    # ── Figures ───────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_radius_profile(profiles, Z_GRID)
    fig_cavity_heatmap(per_rep_arrays, Z_GRID)
    fig_bottleneck(per_rep_arrays, Z_GRID)

    print("\n" + "="*65)
    print("COMPLETE")
    for fn in ["Cavity_radius_profile.png",
               "Cavity_heatmap.png",
               "Cavity_bottleneck.png",
               "Cavity_summary.csv"]:
        print(f"  [{'OK' if os.path.exists(fn) else 'miss'}] {fn}")


if __name__ == "__main__":
    main()
