#!/usr/bin/env python3
"""
Free Energy Landscape comparison
FAD-bound vs P-carnitine-bound c-state
Reads PCA projection files produced by a separate PCA-generation
script (fad_repN_pca.xvg / pcar_repN_pca.xvg).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.ndimage import gaussian_filter, minimum_filter
import os, sys

# Configuration
CONDITIONS = {
    "fad": {
        "label":      "cFAD-bound",
        "pca_files":  ["fad_rep1_pca.xvg",
                       "fad_rep2_pca.xvg",
                       "fad_rep3_pca.xvg"],
        "color":      "#C03830",
        "cmap":       "YlOrRd",
        "rep_colors": ["#E8534A", "#C03830", "#8B1A13"],
    },
    "pcar": {
        "label":      "cPCar-bound",
        "pca_files":  ["pcar_rep1_pca.xvg",
                       "pcar_rep2_pca.xvg",
                       "pcar_rep3_pca.xvg"],
        "color":      "black",
        "cmap":       "Greys",
        "rep_colors": ["#5B9BD5", "#2166AC", "#0D3F7A"],
    },
}

T     = 303.15        # K
R     = 0.0083144621  # kJ/(mol·K)
BINS  = 100
SIGMA = 1.5
FMAX  = 8.0           # kJ/mol ceiling for display

BASIN_THR = 3.0       # kJ/mol — threshold to call a basin
BASIN_SEP = 0.5       # nm    — min separation between basins

DPI        = 1000
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK  = 14

# Parsers
def read_pca(path):
    pc1, pc2 = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(("@","#","&")): continue
            s = line.split()
            if len(s) == 2:
                pc1.append(float(s[0])); pc2.append(float(s[1]))
            elif len(s) >= 3:
                pc1.append(float(s[1])); pc2.append(float(s[2]))
    return np.array(pc1), np.array(pc2)


def build_fel(pc1_all, pc2_all, bins=BINS, sigma=SIGMA, fmax=FMAX,
              x_range=None, y_range=None):
    """Compute FEL from combined PC1/PC2 data."""
    rng = [[x_range[0], x_range[1]] if x_range else [pc1_all.min(), pc1_all.max()],
           [y_range[0], y_range[1]] if y_range else [pc2_all.min(), pc2_all.max()]]
    H, xe, ye = np.histogram2d(pc1_all, pc2_all, bins=bins,
                                range=rng, density=True)
    H  = gaussian_filter(H, sigma=sigma)
    H[H == 0] = np.nan
    F  = -R * T * np.log(H)
    F  = F - np.nanmin(F)
    F[F > fmax] = fmax
    Xc = (xe[:-1] + xe[1:]) / 2
    Yc = (ye[:-1] + ye[1:]) / 2
    X, Y = np.meshgrid(Xc, Yc)
    return X, Y, F.T


def find_basins(Fplot, X, Y, thr=BASIN_THR, sep=BASIN_SEP):
    mask = (Fplot == minimum_filter(Fplot, size=5)) & (Fplot < thr)
    iy_l, ix_l = np.where(mask)
    cands = sorted(
        [(X[iy,ix], Y[iy,ix], float(Fplot[iy,ix]))
         for iy, ix in zip(iy_l, ix_l) if not np.isnan(Fplot[iy,ix])],
        key=lambda x: x[2])
    filtered = []
    for cx, cy, ce in cands:
        if not any(np.sqrt((cx-fx)**2+(cy-fy)**2) < sep
                   for fx, fy, _ in filtered):
            filtered.append((cx, cy, ce))
    return filtered


# Load data
print("Loading PCA data...")
data = {}
for cond, cfg in CONDITIONS.items():
    reps = []
    all_pc1, all_pc2 = [], []
    for i, f in enumerate(cfg["pca_files"], 1):
        if not os.path.exists(f):
            print(f"  WARNING: {f} not found — skipping rep{i}")
            continue
        pc1, pc2 = read_pca(f)
        reps.append((pc1, pc2))
        all_pc1.extend(pc1.tolist())
        all_pc2.extend(pc2.tolist())
        print(f"  {cond} rep{i}: {len(pc1)} frames")
    if not reps:
        sys.exit(f"No data for {cond}. Run combined_PCA_FAD_PCAR.py first.")
    data[cond] = {
        "reps":    reps,
        "pc1_all": np.array(all_pc1),
        "pc2_all": np.array(all_pc2),
    }

# Shared axis range for fair comparison (used ONLY by the overlay panel E)
pc1_combined = np.concatenate([data[c]["pc1_all"] for c in data])
pc2_combined = np.concatenate([data[c]["pc2_all"] for c in data])
pad1 = (pc1_combined.max() - pc1_combined.min()) * 0.05
pad2 = (pc2_combined.max() - pc2_combined.min()) * 0.05
XLIM = (pc1_combined.min() - pad1, pc1_combined.max() + pad1)
YLIM = (pc2_combined.min() - pad2, pc2_combined.max() + pad2)

# Build FELs on the SHARED grid — needed for panel E, where a common
# basis is the whole point of the comparison.
for cond in data:
    X, Y, F = build_fel(data[cond]["pc1_all"],
                         data[cond]["pc2_all"],
                         x_range=XLIM, y_range=YLIM)
    basins = find_basins(F, X, Y)
    iy, ix = np.unravel_index(np.nanargmin(F), F.shape)
    data[cond]["X"] = X
    data[cond]["Y"] = Y
    data[cond]["F"] = F
    data[cond]["basins"]  = basins
    data[cond]["gmin_xy"] = (float(X[iy,ix]), float(Y[iy,ix]))
    print(f"\n  {cond} global minimum: "
          f"PC1={data[cond]['gmin_xy'][0]:.3f}, "
          f"PC2={data[cond]['gmin_xy'][1]:.3f}")
    print(f"  {cond} basins found: {len(basins)}")

# Build FELs on each condition's OWN grid — used for panels A–D, so
# FAD's much more localized landscape fills its own panel instead of
# sitting shrunk inside PCAR's wider shared range.
for cond in data:
    pc1_own, pc2_own = data[cond]["pc1_all"], data[cond]["pc2_all"]
    pad1_own = (pc1_own.max() - pc1_own.min()) * 0.08
    pad2_own = (pc2_own.max() - pc2_own.min()) * 0.08
    xlim_own = (pc1_own.min() - pad1_own, pc1_own.max() + pad1_own)
    ylim_own = (pc2_own.min() - pad2_own, pc2_own.max() + pad2_own)
    X, Y, F = build_fel(pc1_own, pc2_own,
                         x_range=xlim_own, y_range=ylim_own)
    basins = find_basins(F, X, Y)
    iy, ix = np.unravel_index(np.nanargmin(F), F.shape)
    data[cond]["X_own"]     = X
    data[cond]["Y_own"]     = Y
    data[cond]["F_own"]     = F
    data[cond]["basins_own"] = basins
    data[cond]["gmin_own"]  = (float(X[iy,ix]), float(Y[iy,ix]))
    data[cond]["xlim_own"]  = xlim_own
    data[cond]["ylim_own"]  = ylim_own

levels = np.linspace(0, FMAX, 16)

# FIGURE -- 2 rows x 2 columns
# Layout:
#   Row 0: [FAD 3D surface]    [PCAR 3D surface]
#   Row 1: [FAD 2D contour]    [PCAR 2D contour]

fig = plt.figure(figsize=(16, 12.5))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(2, 2, figure=fig,
                        height_ratios=[1.1, 1.0],
                        hspace=0.38, wspace=0.28)

cond_list = list(CONDITIONS.items())

# Row 0: 3D surfaces
for col_idx, (cond, cfg) in enumerate(cond_list):
    ax3d = fig.add_subplot(gs[0, col_idx], projection="3d")
    d = data[cond]
    surf = ax3d.plot_surface(d["X_own"], d["Y_own"], d["F_own"],
                              cmap=cfg["cmap"],
                              linewidth=0, antialiased=True,
                              shade=True, alpha=0.92)
    # Mark global minimum on surface
    gx, gy = d["gmin_own"]
    gz = 0.0
    ax3d.scatter([gx], [gy], [gz+0.05], color="white",
                  edgecolors="black", s=80, zorder=10)

    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.set_edgecolor("#CCCCCC")
    ax3d.grid(True, alpha=0.15)
    ax3d.set_xlabel("PC1 (nm)", fontsize=FONT_LABEL, labelpad=8)
    ax3d.set_ylabel("PC2 (nm)", fontsize=FONT_LABEL, labelpad=8)
    ax3d.set_zlabel("", fontsize=FONT_LABEL, labelpad=8)
    ax3d.set_title(cfg["label"], fontsize=FONT_TITLE,
                   fontweight="bold", pad=12, color="black")
    ax3d.view_init(elev=28, azim=-55)
    ax3d.tick_params(labelsize=FONT_TICK)

    # Attaching the colorbar directly to ax3d (rather than placing it
    # at absolute figure coordinates) keeps it correctly positioned
    # under the surface regardless of the 3D axes' actual layout.
    cb = fig.colorbar(surf, ax=ax3d, orientation="vertical",
                       shrink=0.55, aspect=25, pad=0.05)
    cb.set_label("ΔG (kJ/mol)", fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK-1)

# Row 1: 2D contour with basins
basin_styles = [
    {"color":"white",   "marker":"*", "s":200, "label":"GMin"},
    {"color":"#FFD700", "marker":"o", "s":100},
    {"color":"#00FF7F", "marker":"o", "s":100},
    {"color":"#FF6347", "marker":"o", "s":100},
    {"color":"#87CEEB", "marker":"o", "s":100},
    {"color":"#DDA0DD", "marker":"o", "s":100},
]

ax2d_list = []
for col_idx, (cond, cfg) in enumerate(cond_list):
    ax2d = fig.add_subplot(gs[1, col_idx])
    ax2d_list.append(ax2d)
    d = data[cond]

    cf = ax2d.contourf(d["X_own"], d["Y_own"], d["F_own"],
                        levels=levels, cmap=cfg["cmap"], alpha=0.92)
    ax2d.contour(d["X_own"], d["Y_own"], d["F_own"],
                  levels=levels, colors="white",
                  linewidths=0.5, alpha=0.4)

    xlim_own = d["xlim_own"]
    for j, (bx, by, be) in enumerate(d["basins_own"]):
        st  = basin_styles[j % len(basin_styles)]
        lbl = "GMin" if j == 0 else f"LMin{j}"
        ax2d.scatter(bx, by, color=st["color"],
                     edgecolors="black" if st["color"]!="white" else "dimgray",
                     s=st["s"], marker=st.get("marker","o"),
                     zorder=10, linewidths=0.8)
        offset_x = 0.15 if bx < np.mean(xlim_own) else -0.15
        ax2d.annotate(f"{lbl}\n({bx:.2f},{by:.2f})",
                       xy=(bx, by),
                       xytext=(bx+offset_x*2, by+0.2),
                       fontsize=8.5, color="white", fontweight="bold",
                       arrowprops=dict(arrowstyle="-",
                                       color="white", lw=0.7))

    # Each panel scales to its own condition's data range (xlim_own/
    # ylim_own) rather than the shared XLIM/YLIM, so a more localized
    # landscape (e.g. FAD) fills its own panel instead of sitting
    # shrunk inside a wider shared range.
    ax2d.set_xlim(d["xlim_own"]); ax2d.set_ylim(d["ylim_own"])
    ax2d.set_xlabel("PC1 (nm)", fontsize=FONT_LABEL, fontweight="bold")
    ax2d.set_ylabel("PC2 (nm)", fontsize=FONT_LABEL, fontweight="bold")
    ax2d.set_title(cfg["label"], fontsize=FONT_TITLE,
                   fontweight="bold", color="black")
    ax2d.tick_params(labelsize=FONT_TICK)
    ax2d.set_aspect("equal")

    cbar_side = fig.colorbar(cf, ax=ax2d, fraction=0.046, pad=0.04)
    cbar_side.set_label("ΔG (kJ/mol)", fontsize=FONT_TICK)
    cbar_side.ax.tick_params(labelsize=FONT_TICK)

# Panel labels
plt.suptitle(
    "Free Energy Landscape in C-state\n"
    "cFAD-bound vs cPCar-bound",
    fontsize=FONT_TITLE+1, fontweight="bold", y=1.005)

plt.savefig("FEL_FAD_PCAR_comparison.png",
            dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig("FEL_FAD_PCAR_comparison.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: FEL_FAD_PCAR_comparison.png / .pdf")
plt.show()
