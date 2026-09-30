#!/usr/bin/env python3
"""
Computes and plots aromatic interactions from MD trajectories:

FAD simulations:
  PHE79 – FAD Pyrazine ring  (π–π stacking)
metrics: centroid–centroid distance, plane dihedral angle, vertical height, horizontal offset

P-carnitine simulations:
  PHE79  – PCAR (N⁺)  (π–cation)
  PHE227 – PCAR (N⁺)  (π–cation)
  TRP228 – PCAR (N⁺)  (π–cation)
metrics: ring centroid – N⁺ distance, angle between N⁺→centroid vector and ring normal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import os, warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
    from scipy.stats import gaussian_kde
except ImportError:
    raise SystemExit("pip install MDAnalysis scipy")

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
GRO = "step7_production.gro"
XTC = "step7_production_fit.xtc"
STRIDE = 10   # analyse every Nth frame

#  Atom indices (0-based, MDAnalysis convention) 
# PHE79 ring (CG CD1 CE1 CZ CD2 CE2)
PHE79_IDX   = [1129, 1130, 1132, 1134, 1136, 1138]
# FAD isoalloxazine middle ring, Pyrazine ring
FAD_IDX     = [4390, 4391, 4407, 4409, 4410, 4412]
# PHE227 ring (CG CD1 CE1 CZ CD2 CE2)
PHE227_IDX  = [3246, 3247, 3249, 3251, 3253, 3255]
# TRP228 6-membered ring (CE2 CD2 CE3 CZ3 CZ2 CH2)
TRP228_IDX  = [3271, 3272, 3273, 3275, 3277, 3279]
# P-carnitine quaternary N+
PCAR_N_IDX  = 4396

# Thresholds for interaction classification
PIPI_DIST_THR   = 5.5   # Å  centroid–centroid (π–π)
PIPI_ANG_THR    = 30.0  # °  plane dihedral (parallel if < 30°)
PICAT_DIST_THR  = 6.0   # Å  N+–centroid (π–cation)
PICAT_ANG_THR   = 45.0  # °  angle to ring normal (face-on if < 45°)

DPI   = 1000
FAD_COLOR  = "#C03830"
PCAR_COLOR = "black"
FONT_TITLE = 16
FONT_LABEL = 15
FONT_TICK  = 14
FONT_LEG   = 12


# GEOMETRY FUNCTIONS

def centroid(pos_array):
    """Geometric centroid of atom positions."""
    return pos_array.mean(axis=0)

def ring_normal(pos_array):
    """Normal vector to the plane of ring atoms (via SVD)."""
    c   = centroid(pos_array)
    pts = pos_array - c
    _, _, vh = np.linalg.svd(pts)
    normal = vh[-1]                          
    return normal / np.linalg.norm(normal)

def plane_dihedral_deg(n1, n2):
    """Angle between two plane normals (0° = parallel, 90° = perpendicular)."""
    cos_a = np.clip(np.abs(np.dot(n1, n2)), 0, 1)
    return np.degrees(np.arccos(cos_a))

def vec_angle_deg(v1, v2):
    """Angle between two vectors in degrees."""
    cos_a = np.clip(np.dot(v1, v2) /
                    (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    return np.degrees(np.arccos(np.abs(cos_a)))

def pipi_geometry(phe_pos, fad_pos):
    """
    Compute π–π stacking metrics between PHE ring and FAD ring.
    Returns: (centroid_dist, plane_angle, height, offset) in Å / °
    """
    c_phe = centroid(phe_pos)
    c_fad = centroid(fad_pos)
    n_phe = ring_normal(phe_pos)
    n_fad = ring_normal(fad_pos)

    vec        = c_fad - c_phe
    dist       = np.linalg.norm(vec)
    angle      = plane_dihedral_deg(n_phe, n_fad)
    height     = abs(np.dot(vec, n_phe))          # perpendicular distance
    offset_vec = vec - np.dot(vec, n_phe) * n_phe # lateral displacement
    offset     = np.linalg.norm(offset_vec)
    return dist, angle, height, offset

def picat_geometry(ring_pos, n_pos):
    """
    Compute π–cation metrics between aromatic ring and cation N+.
    Returns: (centroid_dist, angle_to_normal) in Å / °
    """
    c_ring  = centroid(ring_pos)
    n_ring  = ring_normal(ring_pos)
    vec     = n_pos - c_ring
    dist    = np.linalg.norm(vec)
    angle   = vec_angle_deg(vec / (dist + 1e-9), n_ring)
    return dist, angle


# TRAJECTORY ANALYSIS

def analyse_pipi_traj(gro, xtc, stride=STRIDE):
    """Analyse one FAD replica for PHE79–FAD π–π stacking."""
    u    = mda.Universe(gro, xtc)
    phe  = u.atoms[PHE79_IDX]
    fad  = u.atoms[FAD_IDX]
    rows = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        d, a, h, o = pipi_geometry(phe.positions, fad.positions)
        rows.append([ts.time / 1000.0, d, a, h, o])  # time in ns
    return np.array(rows)   # (n_frames, 5)


def analyse_picat_traj(gro, xtc, ring_idx, stride=STRIDE):
    """Analyse one PCAR replica for ring–N+ π–cation interaction."""
    u    = mda.Universe(gro, xtc)
    ring = u.atoms[ring_idx]
    ncat = u.atoms[PCAR_N_IDX]
    rows = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        d, a = picat_geometry(ring.positions, ncat.position)
        rows.append([ts.time / 1000.0, d, a])
    return np.array(rows)   # (n_frames, 3)


def load_or_compute(cache_path, compute_fn, *args, **kwargs):
    """Load cached .npy or compute and save."""
    if os.path.exists(cache_path):
        print(f"  Loading cache: {cache_path}")
        return np.load(cache_path)
    print(f"  Computing: {cache_path}")
    data = compute_fn(*args, **kwargs)
    np.save(cache_path, data)
    return data


def load_replicas_pipi(dirs):
    """Load/compute π–π data for all FAD replicas."""
    all_reps = []
    for i, d in enumerate(dirs, 1):
        cache = os.path.join(d, "pipi_phe79_fad.npy")
        gro   = os.path.join(d, GRO)
        xtc   = os.path.join(d, XTC)
        if not (os.path.exists(gro) and os.path.exists(xtc)):
            print(f"  rep{i}: missing files in {d}")
            continue
        data = load_or_compute(cache, analyse_pipi_traj, gro, xtc)
        all_reps.append(data)
        print(f"  rep{i}: {data.shape[0]} frames")
    return all_reps


def load_replicas_picat(dirs, ring_idx, ring_tag):
    """Load/compute π–cation data for all PCAR replicas."""
    all_reps = []
    for i, d in enumerate(dirs, 1):
        cache = os.path.join(d, f"picat_{ring_tag}.npy")
        gro   = os.path.join(d, GRO)
        xtc   = os.path.join(d, XTC)
        if not (os.path.exists(gro) and os.path.exists(xtc)):
            print(f"  rep{i}: missing files in {d}")
            continue
        data = load_or_compute(cache, analyse_picat_traj,
                               gro, xtc, ring_idx)
        all_reps.append(data)
        print(f"  rep{i}: {data.shape[0]} frames")
    return all_reps


def interp_mean_sd(replicas, col):
    """
    Interpolate replicas onto common time grid, return (t, mean, sd).
    col: column index for the metric.
    """
    t_ref = replicas[0][:, 0]
    traces = []
    for r in replicas:
        v = np.interp(t_ref, r[:, 0], r[:, col])
        traces.append(v)
    arr  = np.array(traces)
    return t_ref, arr.mean(axis=0), arr.std(axis=0, ddof=1)


def concat_all(replicas, col):
    """Concatenate a metric column across all replicas."""
    return np.concatenate([r[:, col] for r in replicas])


# LOAD DATA

print("="*60)
print("PHE79 π–π (FAD) and π–cation (P-carnitine) analysis")
print("="*60)

print("\n[1/4] PHE79–FAD π–π stacking (FAD replicas)...")
pipi_reps = load_replicas_pipi(FAD_DIRS)

print("\n[2/4] PHE79–PCAR π–cation (PCAR replicas)...")
phe79_reps  = load_replicas_picat(PCAR_DIRS, PHE79_IDX,  "phe79_pcar")

print("\n[3/4] PHE227–PCAR π–cation (PCAR replicas)...")
phe227_reps = load_replicas_picat(PCAR_DIRS, PHE227_IDX, "phe227_pcar")

print("\n[4/4] TRP228–PCAR π–cation (PCAR replicas)...")
trp228_reps = load_replicas_picat(PCAR_DIRS, TRP228_IDX, "trp228_pcar")


# FIGURE

fig = plt.figure(figsize=(30, 16))
fig.patch.set_facecolor("white")

# 3 rows: (1) distance time series  (2) angle time series  (3) 2D KDE
gs_outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[1, 1, 1.3],
                              hspace=0.48)

#  Row 1: Distance time series 
gs_r1 = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs_outer[0], wspace=0.32)

row1_panels = [
    (pipi_reps,  1, "F79@π ↔ FAD@Pyrazine ring\n(π–π centroid dist.)",
     FAD_COLOR,  PIPI_DIST_THR,  "Centroid dist. (Å)"),
    (phe79_reps,  1, "F79@π ↔ PCAR@N⁺\n(π–cation dist.)",
     PCAR_COLOR, PICAT_DIST_THR, "[N⁺]–centroid dist. (Å)"),
    (phe227_reps, 1, "F227@π ↔ PCAR@N⁺\n(π–cation dist.)",
     PCAR_COLOR, PICAT_DIST_THR, "[N⁺]–centroid dist. (Å)"),
    (trp228_reps, 1, "W228@π ↔ PCAR@N⁺\n(π–cation dist.)",
     PCAR_COLOR, PICAT_DIST_THR, "[N⁺]–centroid dist. (Å)"),
]

row1_axes = []
for p_idx, (reps, col, title, col_c, thr, ylabel) in enumerate(row1_panels):
    ax = fig.add_subplot(gs_r1[p_idx])
    row1_axes.append(ax)
    if not reps: continue
    t, m, s = interp_mean_sd(reps, col)
    ax.fill_between(t, m-s, m+s, color=col_c, alpha=0.18)
    ax.plot(t, m, color=col_c, lw=1.2, alpha=0.9)
    ax.axhline(thr, color="dimgray", lw=0.8, ls="--", alpha=0.6,
               label=f"Cutoff {thr} Å")
    # Occupancy
    all_vals = concat_all(reps, col)
    occ = 100*np.mean(all_vals < thr)
    ax.text(0.97, 0.96, f"{occ:.1f}% formed",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_TICK-0.5, color=col_c, fontweight="bold")
    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ns)", fontsize=19, fontweight="bold")
    if p_idx == 0:
        ax.set_ylabel("Distance (Å)", fontsize=19, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=7)
    ax.tick_params(labelsize=19)

# Panel label row 1

#  Row 2: Angle time series 
gs_r2 = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs_outer[1], wspace=0.32)

row2_panels = [
    (pipi_reps,   2, "F79@π ↔ FAD@Pyrazine ring\n(plane dihedral angle)",
     FAD_COLOR,  PIPI_ANG_THR,  "Plane angle (°)", "Parallel threshold"),
    (phe79_reps,  2, "F79@π ↔ PCAR@N⁺\n(angle to ring normal)",
     PCAR_COLOR, PICAT_ANG_THR, "Angle to normal (°)", "Face-on threshold"),
    (phe227_reps, 2, "F227@π ↔ PCAR@N⁺\n(angle to ring normal)",
     PCAR_COLOR, PICAT_ANG_THR, "Angle to normal (°)", "Face-on threshold"),
    (trp228_reps, 2, "W228@π ↔ PCAR@N⁺\n(angle to ring normal)",
     PCAR_COLOR, PICAT_ANG_THR, "Angle to normal (°)", "Face-on threshold"),
]

for p_idx, (reps, col, title, col_c, thr, ylabel, thr_lbl) in \
        enumerate(row2_panels):
    ax = fig.add_subplot(gs_r2[p_idx])
    if not reps: continue
    t, m, s = interp_mean_sd(reps, col)
    ax.fill_between(t, m-s, m+s, color=col_c, alpha=0.18)
    ax.plot(t, m, color=col_c, lw=1.2, alpha=0.9)
    ax.axhline(thr, color="dimgray", lw=0.8, ls="--", alpha=0.6,
               label=thr_lbl)
    all_vals = concat_all(reps, col)
    occ = 100*np.mean(all_vals < thr)
    ax.text(0.97, 0.96, f"{occ:.1f}% ideal geometry",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_TICK-0.5, color=col_c, fontweight="bold")
    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ns)", fontsize=19, fontweight="bold")
    if p_idx == 0:
        ax.set_ylabel("Angle (°)", fontsize=19, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=7)
    ax.tick_params(labelsize=19)


#  Row 3: 2D KDE density maps 
gs_r3 = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs_outer[2], wspace=0.35)

kde_panels = [
    (pipi_reps,   1, 2, "F79@π ↔ FAD@Pyrazine ring\n(geometry)",
     FAD_COLOR,
     "Centroid dist. (Å)", "Plane angle (°)",
     PIPI_DIST_THR, PIPI_ANG_THR,
     "Parallel\ndisplaced"),
    (phe79_reps,  1, 2, "F79@π ↔ PCAR@N⁺\n(geometry)",
     PCAR_COLOR,
     "N⁺–centroid dist. (Å)", "Angle to normal (°)",
     PICAT_DIST_THR, PICAT_ANG_THR,
     "Face-on\ngeometry"),
    (phe227_reps, 1, 2, "F227@π ↔ PCAR@N⁺\n(geometry)",
     PCAR_COLOR,
     "N⁺–centroid dist. (Å)", "Angle to normal (°)",
     PICAT_DIST_THR, PICAT_ANG_THR,
     "Face-on\ngeometry"),
    (trp228_reps, 1, 2, "W228@π ↔ PCAR@N⁺\n(geometry)",
     PCAR_COLOR,
     "N⁺–centroid dist. (Å)", "Angle to normal (°)",
     PICAT_DIST_THR, PICAT_ANG_THR,
     "Face-on\ngeometry"),
]

for p_idx, (reps, col_x, col_y, title, col_c,
            xlabel, ylabel, thr_x, thr_y, geom_label) in \
        enumerate(kde_panels):
    ax = fig.add_subplot(gs_r3[p_idx])
    if not reps:
        ax.axis("off"); continue

    x = concat_all(reps, col_x)   # distance
    y = concat_all(reps, col_y)   # angle

    # 2D KDE
    xy  = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.12)
    xi  = np.linspace(x.min(), x.max(), 150)
    yi  = np.linspace(y.min(), y.max(), 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi  = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    # Filled contour
    cf = ax.contourf(Xi, Yi, Zi, levels=20,
                     cmap="Reds" if col_c == FAD_COLOR else "Greys",
                     alpha=0.90)
    ax.contour(Xi, Yi, Zi, levels=6,
               colors="white", linewidths=0.4, alpha=0.5)

    # Threshold lines
    ax.axvline(thr_x, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(thr_y, color="black", lw=1.0, ls="--", alpha=0.7)

    # axvline/axhline auto-extend the axis to include the threshold
    # even when it falls outside the actual KDE data range, which would
    # otherwise leave a blank strip of the dark facecolor where no
    # contour was drawn. Pin the limits back to the real data extent
    # so the axes exactly match the density map.
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())

    # Annotate the high-density preferred zone
    peak_idx = np.unravel_index(Zi.argmax(), Zi.shape)
    peak_x   = Xi[peak_idx]
    peak_y   = Yi[peak_idx]
    ax.scatter(peak_x, peak_y, color="white", s=60,
               edgecolors=col_c, zorder=5, linewidth=1.5)
    ax.text(peak_x + 0.15, peak_y + 1.5,
            f"({peak_x:.1f} Å, {peak_y:.0f}°)",
            fontsize=FONT_TICK + 2, color="black", fontweight="bold")

    # Preferred geometry label (bottom-left of threshold zone)
    ax.text(0.97, 0.96, geom_label,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_TICK - 1.5, color="black",
            fontweight="bold", style="italic")

    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability density", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax.set_facecolor("#1a1a2e")   # dark background makes KDE pop
    ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=19, fontweight="bold")
    if p_idx == 0:
        ax.set_ylabel("Angle (°)", fontsize=19, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold",
                 pad=7, color="black")
    ax.tick_params(labelsize=19)


# Shared row labels
fig.text(0.5, 0.99,
         "Aromatic Interactions — FAD (π–π stacking) and "
         "P-carnitine (π–cation)",
         ha="center", va="top",
         fontsize=FONT_TITLE + 1, fontweight="bold")

plt.savefig("Aromatic_interactions.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("Aromatic_interactions.pdf",
            bbox_inches="tight", facecolor="white")
print("\nSaved: Aromatic_interactions.png / .pdf")
plt.show()
