#!/usr/bin/env python3
"""
π–Cation interactions: PHE79 and PHE227 with carnitine N+
mCAR-bound m-state | 3 replicas × 1µs
Computes from scratch (no XVG needed):
  PHE79  ring centroid ↔ CAR N⁺   (π–cation)
  PHE227 ring centroid ↔ CAR N⁺   (π–cation)
Metrics: centroid–N⁺ distance | angle to ring normal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
    from scipy.stats import gaussian_kde
except ImportError:
    raise SystemExit("conda install -c conda-forge mdanalysis scipy")

# CONFIGURATION

CAR_DIRS = [
    ".../rep1",
    ".../rep2",
    ".../rep3",
]
GRO    = "step7_production.gro"
XTC    = "step7_production_fit.xtc"
STRIDE = 10

#  Atom indices (0-based MDAnalysis) 
PHE79_IDX  = [1129, 1130, 1132, 1134, 1136, 1138]  # CG CD1 CE1 CZ CD2 CE2
PHE227_IDX = [3246, 3247, 3249, 3251, 3253, 3255]  # CG CD1 CE1 CZ CD2 CE2
CAR_N_IDX  = 4390   # quaternary N+ (0-based; atom 4391 GROMACS 1-based)

PICAT_DIST_THR = 6.0   # Å
PICAT_ANG_THR  = 45.0  # °

DPI        = 1000
CAR_COLOR  = "black"
FONT_TITLE = 17
FONT_LABEL = 16
FONT_TICK  = 14
FONT_LEG   = 14


# GEOMETRY

def centroid(pos): return pos.mean(axis=0)

def ring_normal(pos):
    c = centroid(pos)
    _, _, vh = np.linalg.svd(pos - c)
    n = vh[-1]
    return n / np.linalg.norm(n)

def vec_angle_deg(v1, v2):
    cos_a = np.clip(np.dot(v1, v2) /
                    (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12), -1, 1)
    return np.degrees(np.arccos(np.abs(cos_a)))

def picat_geometry(ring_pos, n_pos):
    c_ring = centroid(ring_pos)
    n_ring = ring_normal(ring_pos)
    vec    = n_pos - c_ring
    dist   = np.linalg.norm(vec)
    angle  = vec_angle_deg(vec/(dist+1e-12), n_ring)
    return dist, angle


# TRAJECTORY ANALYSIS

def analyse_picat_traj(gro, xtc, ring_idx, stride=STRIDE):
    u    = mda.Universe(gro, xtc)
    ring = u.atoms[ring_idx]
    ncat = u.atoms[CAR_N_IDX]
    rows = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        d, a = picat_geometry(ring.positions, ncat.position)
        rows.append([ts.time/1000.0, d, a])
    return np.array(rows)


def load_or_compute(cache, fn, *args, **kwargs):
    if os.path.exists(cache):
        print(f"  Cache: {os.path.basename(cache)}")
        return np.load(cache)
    print(f"  Computing: {os.path.basename(cache)}")
    data = fn(*args, **kwargs)
    np.save(cache, data)
    return data


def load_replicas(dirs, ring_idx, tag):
    reps = []
    for i, d in enumerate(dirs, 1):
        cache = os.path.join(d, f"picat_{tag}_CAR.npy")
        gro   = os.path.join(d, GRO)
        xtc   = os.path.join(d, XTC)
        if not (os.path.exists(gro) and os.path.exists(xtc)):
            print(f"  rep{i}: missing files"); continue
        data = load_or_compute(cache, analyse_picat_traj, gro, xtc, ring_idx)
        reps.append(data)
        print(f"  rep{i}: {data.shape[0]} frames")
    return reps


def interp_mean_sd(reps, col):
    t_ref = reps[0][:,0]
    traces = [np.interp(t_ref, r[:,0], r[:,col]) for r in reps]
    arr = np.array(traces)
    return t_ref, arr.mean(axis=0), arr.std(axis=0, ddof=1)


def concat_all(reps, col):
    return np.concatenate([r[:,col] for r in reps])


# LOAD DATA

print("="*60)
print("mCAR π–Cation Interactions (PHE79 & PHE227 ↔ CAR N⁺)")
print("="*60)

print("\n[1/2] PHE79 – CAR N⁺ π–cation...")
phe79_reps  = load_replicas(CAR_DIRS, PHE79_IDX,  "phe79")

print("\n[2/2] PHE227 – CAR N⁺ π–cation...")
phe227_reps = load_replicas(CAR_DIRS, PHE227_IDX, "phe227")


# FIGURE — 3 rows × 2 columns

panels = [
    (phe79_reps,  "F79@π ↔ CAR@N⁺\n(π–cation dist.)"),
    (phe227_reps, "F227@π ↔ CAR@N⁺\n(π–cation dist.)"),
]

fig = plt.figure(figsize=(12, 16))
fig.patch.set_facecolor("white")
gs_outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[1, 1, 1.3],
                              hspace=0.48)

#  Row A: Distance time series 
gs_r1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[0], wspace=0.32)

for p_idx, (reps, title) in enumerate(panels):
    ax = fig.add_subplot(gs_r1[p_idx])
    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    if not reps: continue

    t, m, s = interp_mean_sd(reps, 1)
    ax.fill_between(t, m-s, m+s, color=CAR_COLOR, alpha=0.18)
    ax.plot(t, m, color=CAR_COLOR, lw=1.4)
    ax.axhline(PICAT_DIST_THR, color="dimgray",
               lw=0.8, ls="--", alpha=0.6)

    all_v = concat_all(reps, 1)
    occ   = 100*np.mean(all_v < PICAT_DIST_THR)
    ax.text(0.97, 0.97, f"{occ:.1f}% formed",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_TICK-0.5, color=CAR_COLOR, fontweight="bold")

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL, fontweight="bold")
    if p_idx == 0:
        ax.set_ylabel("N⁺–centroid dist. (Å)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title,
                 fontsize=FONT_TITLE, fontweight="bold", pad=7)
    ax.tick_params(labelsize=FONT_TICK)


#  Row B: Angle time series 
gs_r2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[1], wspace=0.32)

angle_titles = [
    "F79@π ↔ CAR@N⁺\n(angle to ring normal)",
    "F227@π ↔ CAR@N⁺\n(angle to ring normal)",
]
for p_idx, (reps, title) in enumerate(zip([phe79_reps, phe227_reps],
                                           angle_titles)):
    ax = fig.add_subplot(gs_r2[p_idx])
    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    if not reps: continue

    t, m, s = interp_mean_sd(reps, 2)
    ax.fill_between(t, m-s, m+s, color=CAR_COLOR, alpha=0.18)
    ax.plot(t, m, color=CAR_COLOR, lw=1.4)
    ax.axhline(PICAT_ANG_THR, color="dimgray",
               lw=0.8, ls="--", alpha=0.6)

    all_v = concat_all(reps, 2)
    occ   = 100*np.mean(all_v < PICAT_ANG_THR)
    ax.text(0.97, 0.97, f"{occ:.1f}% ideal geometry",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_TICK-0.5, color=CAR_COLOR, fontweight="bold")

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL, fontweight="bold")
    if p_idx == 0:
        ax.set_ylabel("Angle to normal (°)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=7)
    ax.tick_params(labelsize=FONT_TICK)


#  Row C: 2D KDE density maps 
gs_r3 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[2], wspace=0.35)

kde_titles = [
    "F79@π ↔ CAR@N⁺\n(geometry)",
    "F227@π ↔ CAR@N⁺\n(geometry)",
]
xlabels = ["N⁺–centroid dist. (Å)", "N⁺–centroid dist. (Å)"]

for p_idx, (reps, title, xlabel) in enumerate(
        zip([phe79_reps, phe227_reps], kde_titles, xlabels)):
    ax = fig.add_subplot(gs_r3[p_idx])
    if not reps: ax.axis("off"); continue

    x = concat_all(reps, 1)
    y = concat_all(reps, 2)

    xy  = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.12)
    xi  = np.linspace(x.min(), x.max(), 150)
    yi  = np.linspace(y.min(), y.max(), 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi  = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    cf = ax.contourf(Xi, Yi, Zi, levels=20,
                     cmap="Greys", alpha=0.92)
    ax.contour(Xi, Yi, Zi, levels=6,
               colors="white", linewidths=0.4, alpha=0.5)

    ax.axvline(PICAT_DIST_THR, color="white", lw=1.0,
               ls="--", alpha=0.7)
    ax.axhline(PICAT_ANG_THR, color="white", lw=1.0,
               ls="--", alpha=0.7)
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())

    peak_idx = np.unravel_index(Zi.argmax(), Zi.shape)
    px, py   = Xi[peak_idx], Yi[peak_idx]
    ax.scatter(px, py, color="white", s=60,
               edgecolors=CAR_COLOR, zorder=5, linewidth=1.5)
    ax.text(px+0.15, py-2.0, f"({px:.1f} Å, {py:.0f}°)",
            fontsize=14, color="black", fontweight="bold")

    ax.text(0.97, 0.97, "Face-on\ngeometry",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_TICK, color="black",
            fontweight="bold", style="italic")

    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability density", fontsize=FONT_TICK)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    ax.set_facecolor("#1a1a2e")
    ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, fontweight="bold")
    if p_idx == 0:
        ax.set_ylabel("Angle (°)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold",
                 pad=7, color="black")
    ax.tick_params(labelsize=FONT_TICK)


fig.text(0.5, 0.99,
         "mCAR π–cation interactions\n[PHE79/227@π ↔ CAR@N⁺]",
         ha="center", va="top",
         fontsize=FONT_TITLE+1, fontweight="bold")
plt.savefig("mCAR_aromatic_picat.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("mCAR_aromatic_picat.pdf",
            bbox_inches="tight", facecolor="white")
print("\nSaved: mCAR_aromatic_picat.png / .pdf")
plt.show()
