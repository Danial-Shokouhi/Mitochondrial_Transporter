#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
Dynamic Network Analysis + DCCM for THREE conditions:
  cFAD-bound  (3 replicas × 1µs)
  cPCar-bound (3 replicas × 1µs)
  Apo c-state (3 replicas × 1µs)

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, sys, warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances as mda_dist
except ImportError:
    sys.exit("Install MDAnalysis: conda install -c conda-forge mdanalysis")

try:
    import networkx as nx
except ImportError:
    sys.exit("Install NetworkX: conda install -c conda-forge networkx")

from scipy.spatial.distance import cdist

# ── BOU sequence for one-letter residue labels ─────────────────────────
BOU_SEQ = ("MADAWKDLASGTVGGAAQLVVGHPFDTIKVKLQSQPTPAPGQLPRYTGAIDAVKQTVASE"
            "GTKGLYKGMGAPLATVAAFNAVLFTVRGQMEGLLRSEAGVPLTISQQFVAGAGAGFAVSF"
            "LACPTELIKCRLQAQGALAGASTTSSVVAAVKYGGPMDVARHVLRSEGGARGLFKGLFPT"
            "FAREVPGNATMFAAYEAFKRFLAGGSDTSSLGQGSLIMAGGVAGASFWGIVYPTDVVKSV"
            "LQVDDYKNPRYTGSMDAFRKILKSEGVKGLYKGFGPAMARSVPANAACFLAYEMTRSSLG")

def res_label(res_1based):
    """Return e.g. 'F79' for residue 79."""
    idx = res_1based - 1
    if 0 <= idx < len(BOU_SEQ):
        return f"{BOU_SEQ[idx]}{res_1based}"
    return str(res_1based)



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

N_RESIDUES       = 300
CONTACT_CUTOFF_A = 4.5
STRIDE           = 10

# Source: FAD/PCAR binding-site residues
SOURCE_RESIDS = [280, 183, 184, 228, 79, 18]
SOURCE_LABELS = {280:"ARG280", 183:"ARG183", 184:"GLU184",
                 228:"TRP228",  79:"PHE79",   18:"GLN18"}

# Sink: m-gate residues
SINK_RESIDS = [26, 129, 126, 238, 235, 29]
SINK_LABELS = {26:"ASP26", 129:"LYS129", 126:"GLU126",
               238:"LYS238", 235:"ASP235", 29:"LYS29"}

# Structural regions
REGIONS = [
    ("NTER",  1,   1), ("TMH1",  2,  33), ("ml12", 34,  48),
    ("mh12", 49,  69), ("TMH2", 70,  94), ("imsl23",95,103),
    ("TMH3",104, 138), ("ml34",139, 155), ("mh34",156, 176),
    ("TMH4",177, 203), ("imsl45",204,212), ("TMH5",213,242),
    ("ml56",243, 253), ("mh56",254, 273), ("TMH6",274,299),
    ("CTER",300, 300),
]
def region_color(name):
    if name.startswith("TMH"): return "#D4E6F1"
    if name in ("NTER","CTER"): return "#F2F3F4"
    return "#FAD7A0"
REGION_COLORS = [region_color(n) for n,_,_ in REGIONS]

# Colors per condition
COND_COLORS = {
    "FAD":  "#C03830",
    "PCAR": "black",
    "APO":  "#2166AC",
}
COND_LABELS = {
    "FAD":  "cFAD-bound",
    "PCAR": "cPCar-bound",
    "APO":  "Apo-cBOU",
}


# ═══════════════════════════════════════════════════════════════════════
# COMPUTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_dccm(gro_path, xtc_path, n_res, stride=10):
    u  = mda.Universe(gro_path, xtc_path)
    ca = u.select_atoms("protein and name CA")
    assert len(ca) == n_res, f"Expected {n_res} CA, got {len(ca)}"
    frames = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        frames.append(ca.positions.copy())
    frames = np.array(frames)
    n_frames = frames.shape[0]
    print(f"    {n_frames} frames")
    mean_pos = frames.mean(axis=0)
    delta    = frames - mean_pos[np.newaxis]
    cij = np.zeros((n_res, n_res))
    for i in range(n_res):
        di = delta[:, i, :]
        for j in range(i, n_res):
            dj = delta[:, j, :]
            dot = (di * dj).sum(axis=1).mean()
            cij[i,j] = cij[j,i] = dot
    diag  = np.sqrt(np.diag(cij))
    denom = np.outer(diag, diag)
    denom[denom == 0] = 1.0
    return np.clip(cij / denom, -1.0, 1.0)

def average_dccm(dirs, gro, xtc, n_res, stride, label):
    matrices = []
    for i, d in enumerate(dirs, 1):
        gp, xp = os.path.join(d,gro), os.path.join(d,xtc)
        if not (os.path.exists(gp) and os.path.exists(xp)):
            print(f"  SKIP {d}"); continue
        print(f"  {label} rep{i}: DCCM...")
        matrices.append(compute_dccm(gp, xp, n_res, stride))
    if not matrices: sys.exit(f"No data for {label}")
    return np.mean(matrices,axis=0), np.std(matrices,axis=0)

def compute_rmsf(gro_path, xtc_path, stride=10):
    u  = mda.Universe(gro_path, xtc_path)
    ca = u.select_atoms("protein and name CA")
    frames = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        frames.append(ca.positions.copy())
    frames   = np.array(frames)
    mean_pos = frames.mean(axis=0)
    diff     = frames - mean_pos[np.newaxis]
    return np.sqrt((diff**2).sum(axis=2).mean(axis=0))

def average_rmsf(dirs, gro, xtc, stride, label):
    all_r = []
    for i, d in enumerate(dirs, 1):
        gp, xp = os.path.join(d,gro), os.path.join(d,xtc)
        if not (os.path.exists(gp) and os.path.exists(xp)): continue
        print(f"  {label} rep{i}: RMSF...")
        all_r.append(compute_rmsf(gp, xp, stride))
    return np.mean(all_r,axis=0), np.std(all_r,axis=0)

def compute_contact_matrix(gro_path, xtc_path, n_res, cutoff_a, stride=10):
    u       = mda.Universe(gro_path, xtc_path)
    protein = u.select_atoms("protein and not name H*")
    res_idx = np.array([a.resindex for a in protein])
    csum    = np.zeros((n_res, n_res))
    nf = 0
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        pos = protein.positions; nf += 1
        dm  = cdist(pos, pos)
        for ri in range(n_res):
            mi = res_idx == ri
            for rj in range(ri+3, n_res):
                mj = res_idx == rj
                if dm[np.ix_(mi,mj)].min() < cutoff_a:
                    csum[ri,rj] += 1; csum[rj,ri] += 1
    return csum / max(nf,1)

def average_contact_matrix(dirs, gro, xtc, n_res, cutoff_a, stride, label):
    mats = []
    for i, d in enumerate(dirs, 1):
        gp, xp = os.path.join(d,gro), os.path.join(d,xtc)
        if not (os.path.exists(gp) and os.path.exists(xp)): continue
        print(f"  {label} rep{i}: contacts...")
        mats.append(compute_contact_matrix(gp, xp, n_res, cutoff_a, stride))
    return np.mean(mats, axis=0)

def build_network(contact_matrix, freq_threshold=0.50):
    G = nx.Graph()
    n = contact_matrix.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if contact_matrix[i,j] >= freq_threshold:
                G.add_edge(i, j, weight=contact_matrix[i,j])
    return G

def compute_network_metrics(G, rmsf):
    degree  = nx.degree_centrality(G)
    between = nx.betweenness_centrality(G, weight="weight", normalized=True)
    n = max(G.nodes())+1
    return (np.array([degree.get(i,0.) for i in range(n)]),
            np.array([between.get(i,0.) for i in range(n)]))

def find_optimal_path(G, source_resids, sink_resids, contact_matrix, betweenness=None):
    bet = betweenness or {}
    Gc  = nx.DiGraph()
    for u,v,data in G.edges(data=True):
        w  = max(data.get("weight",0.01),1e-6)
        b  = base = -np.log(w)
        bv = max(bet.get(v,1e-9),1e-9)
        bu = max(bet.get(u,1e-9),1e-9)
        Gc.add_edge(u,v,weight=b/bv)
        Gc.add_edge(v,u,weight=b/bu)
    s0 = [r-1 for r in source_resids if r-1 in Gc]
    t0 = [r-1 for r in sink_resids   if r-1 in Gc]
    for s in s0:
        for t in t0:
            if Gc.has_edge(s,t): Gc.remove_edge(s,t)
            if Gc.has_edge(t,s): Gc.remove_edge(t,s)
    VS,VT = "VSRC","VSINK"
    for s in s0: Gc.add_edge(VS,s,weight=0.)
    for t in t0: Gc.add_edge(t,VT,weight=0.)
    Gw = Gc.copy(); best = None
    for _ in range(30):
        try: raw = nx.shortest_path(Gw,VS,VT,weight="weight")
        except (nx.NetworkXNoPath,nx.NodeNotFound): break
        path = [p for p in raw if p not in (VS,VT)]
        if len(path) >= 8: best = path; break
        for k in range(len(raw)-1):
            u2,v2 = raw[k],raw[k+1]
            if Gw.has_edge(u2,v2): Gw.remove_edge(u2,v2)
        if best is None: best = path
    return best


# ═══════════════════════════════════════════════════════════════════════
# LOAD OR COMPUTE ALL MATRICES
# ═══════════════════════════════════════════════════════════════════════

def load_or_compute_dccm(dirs, tag):
    npy = f"DCCM_{tag}.npy"
    if os.path.exists(npy):
        print(f"  Loading cache: {npy}")
        return np.load(npy)
    m, _ = average_dccm(dirs, GRO, XTC, N_RESIDUES, STRIDE, tag)
    np.save(npy, m); print(f"  Saved: {npy}")
    return m

def load_or_compute_contact(dirs, tag):
    npy = f"ContactMatrix_{tag}.npy"
    if os.path.exists(npy):
        print(f"  Loading cache: {npy}")
        return np.load(npy)
    m = average_contact_matrix(dirs, GRO, XTC, N_RESIDUES,
                                CONTACT_CUTOFF_A, STRIDE, tag)
    np.save(npy, m); print(f"  Saved: {npy}")
    return m


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1
# ═══════════════════════════════════════════════════════════════════════

def fig_dccm(dccm_fad, dccm_pcar, dccm_apo, n_res):
    """
    6-panel figure:
    Row 1: FAD | PCAR | APO  (individual DCCMs)
    Row 2: Δ(FAD−APO) | Δ(PCAR−APO) | Δ(FAD−PCAR)  (difference maps)
    """
    fig = plt.figure(figsize=(26, 14))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.28)

    diff_fa  = dccm_fad  - dccm_apo
    diff_pa  = dccm_pcar - dccm_apo
    diff_fp  = dccm_fad  - dccm_pcar

    norm_corr = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    dlim = 0.55
    norm_diff = TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)

    panels = [
        (dccm_fad,  "cFAD DCCM",        norm_corr, "RdBu_r", False),
        (dccm_pcar, "cPCar DCCM",        norm_corr, "RdBu_r", False),
        (dccm_apo,  "Apo-cBOU DCCM",     norm_corr, "RdBu_r", False),
        (diff_fa,   "ΔDCCM (FAD − Apo)", norm_diff, "PiYG",   True),
        (diff_pa,   "ΔDCCM (PCar − Apo)",norm_diff, "PiYG",   True),
        (diff_fp,   "ΔDCCM (FAD − PCar)",norm_diff, "PRGn",   True),
    ]

    for idx, (mat, title, norm, cmap, is_diff) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(mat, cmap=cmap, norm=norm,
                       origin="upper", aspect="equal",
                       extent=[1, n_res, n_res, 1])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("ΔDCCM" if is_diff else "Correlation",
                     fontsize=17, fontweight="bold")
        cb.ax.tick_params(labelsize=15)
        ax.set_xlabel("Residue", fontsize=17, fontweight="bold")
        ax.set_ylabel("Residue", fontsize=17, fontweight="bold")
        ax.set_title(title, fontsize=20, fontweight="bold", pad=7)
        ax.tick_params(labelsize=15)

        # Source/sink crosshairs
        for r in SOURCE_RESIDS:
            ax.axvline(r, color="#F1C40F", lw=0.7, alpha=0.5)
            ax.axhline(r, color="#F1C40F", lw=0.7, alpha=0.5)
        for r in SINK_RESIDS:
            ax.axvline(r, color="#2ECC71", lw=0.7, alpha=0.5)
            ax.axhline(r, color="#2ECC71", lw=0.7, alpha=0.5)

        # Label key residues on diagonal
        if is_diff:
            for r, lbl in {**SOURCE_LABELS, **SINK_LABELS}.items():
                col_mk = "#F1C40F" if r in SOURCE_RESIDS else "#2ECC71"
                ax.text(r, r, "◆", ha="center", va="center",
                        fontsize=5, color=col_mk, zorder=5)

    # Shared legend
    leg_handles = [
        Line2D([0],[0], color="#F1C40F", lw=2,
               label="Source (binding site)"),
        Line2D([0],[0], color="#2ECC71", lw=2,
               label="Sink (m-gate)"),
    ]
    fig.legend(handles=leg_handles, fontsize=16.5,
               loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    plt.suptitle(
        "Dynamic Cross-Correlation Matrix\n"
        "cFAD-bound | cPCar-bound | Apo-cBOU",
        fontsize=14, fontweight="bold", y=1.01)

    plt.savefig("DCCM_comparison.png", dpi=1000,
                bbox_inches="tight", facecolor="white")
    plt.savefig("DCCM_comparison.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: DCCM_comparison.png / .pdf")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Network metrics (4 panels, 3 conditions)
# ═══════════════════════════════════════════════════════════════════════

def add_region_bands(ax, regions, colors, orientation="x", alpha=0.22):
    for (name, start, end), col in zip(regions, colors):
        s, e = start-1, end
        if orientation == "x":
            ax.axvspan(s, e, color=col, alpha=alpha, zorder=0)
        else:
            ax.axhspan(s, e, color=col, alpha=alpha, zorder=0)


def fig_network_metrics(rmsf_fad, rmsf_pcar, rmsf_apo,
                         deg_fad,  deg_pcar,  deg_apo,
                         bet_fad,  bet_pcar,  bet_apo,
                         path_fad, path_pcar, path_apo, n_res):
    res = np.arange(1, n_res+1)
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    fig.patch.set_facecolor("white")

    triples = [
        (rmsf_fad,rmsf_pcar,rmsf_apo, "RMSF (Å)", ""),
        (deg_fad, deg_pcar, deg_apo,  "Degree centrality", ""),
        (bet_fad, bet_pcar, bet_apo,  "Betweenness centrality", ""),
    ]

    for ax, (vf,vp,va, ylabel, panel_l) in zip(axes[:3], triples):
        add_region_bands(ax, REGIONS, REGION_COLORS)
        ax.plot(res, vf, color=COND_COLORS["FAD"],  lw=1.6,
                label=COND_LABELS["FAD"],  zorder=3)
        ax.plot(res, vp, color=COND_COLORS["PCAR"], lw=1.6,
                label=COND_LABELS["PCAR"], zorder=3)
        ax.plot(res, va, color=COND_COLORS["APO"],  lw=1.4,
                label=COND_LABELS["APO"],  zorder=3, alpha=0.75)
        ax.fill_between(res, vf, va, alpha=0.10,
                         color=COND_COLORS["FAD"])
        ax.fill_between(res, vp, va, alpha=0.10,
                         color=COND_COLORS["PCAR"])
        for r in SOURCE_RESIDS:
            ax.axvline(r, color="#F1C40F", lw=1.0, ls="--",
                       alpha=0.55,
                       label="Source" if r==SOURCE_RESIDS[0] else "")
        for r in SINK_RESIDS:
            ax.axvline(r, color="#2ECC71", lw=1.0, ls="--",
                       alpha=0.55,
                       label="Sink" if r==SINK_RESIDS[0] else "")
        # Mark allosteric paths
        for path, col in [(path_fad,  COND_COLORS["FAD"]),
                           (path_pcar, COND_COLORS["PCAR"])]:
            if path:
                for node in path:
                    ax.axvline(node+1, color=col,
                               lw=0.5, alpha=0.35)
        ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")
        ax.text(0.005, 0.96, panel_l, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top")
        ax.tick_params(axis="y", labelsize=22)
        ax.tick_params(axis="x", labelsize=22)
        ax.legend(fontsize=13.5, loc="upper right",
                  frameon=False, ncol=3)
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)

    # Panel D — ΔBetweenness bars (FAD-Apo and PCAR-Apo side by side)
    ax = axes[3]
    diff_fa = bet_fad  - bet_apo
    diff_pa = bet_pcar - bet_apo
    add_region_bands(ax, REGIONS, REGION_COLORS)
    bw = 0.48
    ax.bar(res - 0.25, diff_fa, bw,
           color=np.where(diff_fa > 0,
                          COND_COLORS["FAD"], "#C03830"),
           alpha=0.8, label="FAD − Apo")
    ax.bar(res + 0.25, diff_pa, bw,
           color=np.where(diff_pa > 0,
                          COND_COLORS["PCAR"], "black"),
           alpha=0.8, label="PCar − Apo")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔBetweenness\n(ligand − Apo)",
                  fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22)
    ax.set_xlabel("Residue Number", fontsize=18, fontweight="bold")
    ax.text(0.005, 0.96, "", transform=ax.transAxes,
            fontsize=19, fontweight="bold", va="top")
    ax.legend(fontsize=13.5, frameon=False)
    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)

    # Annotate top rewired hubs
    top_idx = np.argsort(np.abs(diff_fa))[-8:]
    for idx in top_idx:
        if abs(diff_fa[idx]) > 0.003:
            ax.annotate(res_label(idx+1),
                        xy=(idx+1, diff_fa[idx]),
                        xytext=(idx+1,
                                diff_fa[idx]+np.sign(diff_fa[idx])*0.004),
                        fontsize=10.5, ha="center", color="black")

    # Region labels
    plt.tight_layout()
    ylo, yhi = ax.get_ylim()
    label_y = ylo - (yhi-ylo)*0.18
    for (name,start,end), col in zip(REGIONS, REGION_COLORS):
        mid = (start+end)/2
        ax.annotate(name, xy=(mid,ylo),
                    xytext=(mid,label_y),
                    ha="center", va="top",
                    fontsize=12.5, fontweight="bold",
                    rotation=90, color="dimgray",
                    annotation_clip=False)

    plt.suptitle(
        "Network Metrics — cFAD | cPCar | Apo-cBOU",
        fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Network_metrics.png", dpi=1000,
                bbox_inches="tight", facecolor="white")
    plt.savefig("Network_metrics.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: Network_metrics.png / .pdf")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Allosteric paths (FAD | PCAR | APO)
# ═══════════════════════════════════════════════════════════════════════

def draw_path_subnetwork(ax, path, contact_matrix, color,
                          title, source_resids, sink_resids):
    if not path or len(path) < 2:
        ax.text(0.5, 0.5, "Path not found", ha="center",
                transform=ax.transAxes, fontsize=17)
        ax.axis("off")
        ax.set_title(title, fontsize=17, fontweight="bold")
        return
    G_sub = nx.Graph()
    for p in path: G_sub.add_node(p)
    for k in range(len(path)-1):
        i, j = path[k], path[k+1]
        G_sub.add_edge(i, j, weight=contact_matrix[i,j])
    pos = nx.spring_layout(G_sub, seed=42, k=1.5)
    node_col = []
    node_size = []
    for p in G_sub.nodes():
        if (p+1) in source_resids:
            node_col.append("#F1C40F"); node_size.append(1550)
        elif (p+1) in sink_resids:
            node_col.append("#2ECC71"); node_size.append(1550)
        else:
            node_col.append(color); node_size.append(1300)
    edge_w = [G_sub[u][v]["weight"]*5 for u,v in G_sub.edges()]
    nx.draw_networkx(G_sub, pos=pos, ax=ax,
                     node_color=node_col, node_size=node_size,
                     edge_color=color, width=edge_w,
                     labels={p: res_label(p+1) for p in G_sub.nodes()},
                     font_size=13, font_weight="bold",
                     font_color="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")


def fig_allosteric_path(bet_fad, bet_pcar, bet_apo,
                         path_fad, path_pcar, path_apo,
                         n_res, contact_fad, contact_pcar, contact_apo):
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    # Row 1: betweenness along each path
    paths    = [path_fad,  path_pcar,  path_apo]
    bets     = [bet_fad,   bet_pcar,   bet_apo]
    conds    = ["FAD",     "PCAR",     "APO"]
    titles_r1 = [
        "FAD path betweenness",
        "PCar path betweenness",
        "Apo path betweenness",
    ]
    for col, (path, bet, cond, title) in enumerate(
            zip(paths, bets, conds, titles_r1)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)
        if path:
            path_res = [p+1 for p in path]
            path_bet = [bet[p] for p in path]
            col_pts  = [("#F1C40F" if (r in SOURCE_RESIDS) else
                          "#2ECC71" if (r in SINK_RESIDS) else
                          COND_COLORS[cond])
                         for r in path_res]
            ax.plot(range(len(path_res)), path_bet,
                    "-", color=COND_COLORS[cond], lw=1.8, zorder=2)
            ax.scatter(range(len(path_res)), path_bet,
                       c=col_pts, s=70, zorder=3,
                       edgecolors="white", linewidth=0.7)
            ax.tick_params(axis="y", labelsize=16)
            ax.set_xticklabels([res_label(r) for r in path_res],
                                rotation=45, ha="right", fontsize=15)
        ax.set_ylabel("Betweenness", fontsize=16, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")

    # Row 2: subnetwork graphs
    contacts = [contact_fad, contact_pcar, contact_apo]
    titles_r2 = [
        "FAD allosteric subnetwork",
        "PCar allosteric subnetwork",
        "Apo allosteric subnetwork",
    ]
    for col, (path, cmat, cond, title) in enumerate(
            zip(paths, contacts, conds, titles_r2)):
        ax = fig.add_subplot(gs[1, col])
        draw_path_subnetwork(ax, path, cmat,
                              COND_COLORS[cond], title,
                              SOURCE_RESIDS, SINK_RESIDS)

    # Legend
    leg_handles = [
        Patch(color=COND_COLORS["FAD"],  label=COND_LABELS["FAD"]),
        Patch(color=COND_COLORS["PCAR"], label=COND_LABELS["PCAR"]),
        Patch(color=COND_COLORS["APO"],  label=COND_LABELS["APO"]),
        Patch(color="#F1C40F", label="Source (binding site)"),
        Patch(color="#2ECC71", label="Sink (m-gate)"),
    ]
    fig.legend(handles=leg_handles, fontsize=17,
               loc="lower center", ncol=5,
               bbox_to_anchor=(0.5,-0.02), frameon=False)

    plt.suptitle(
        "Allosteric Communication",
        fontsize=16, fontweight="bold")
    plt.savefig("Allosteric_path.png", dpi=1000,
                bbox_inches="tight", facecolor="white")
    plt.savefig("Allosteric_path.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: Allosteric_path.png / .pdf")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("DYNAMIC NETWORK ANALYSIS + DCCM")
    print("cFAD | cPCar | Apo-cBOU")
    print("="*65)

    # ── DCCM ──────────────────────────────────────────────────────────
    print("\n[1/5] DCCM — FAD")
    dccm_fad  = load_or_compute_dccm(FAD_DIRS,  "FAD")
    print("\n[1/5] DCCM — PCAR")
    dccm_pcar = load_or_compute_dccm(PCAR_DIRS, "PCAR")
    print("\n[1/5] DCCM — APO")
    dccm_apo  = load_or_compute_dccm(APO_DIRS,  "APO")

    # ── RMSF ──────────────────────────────────────────────────────────
    print("\n[2/5] RMSF")
    rmsf_fad,  _ = average_rmsf(FAD_DIRS,  GRO, XTC, STRIDE, "FAD")
    rmsf_pcar, _ = average_rmsf(PCAR_DIRS, GRO, XTC, STRIDE, "PCAR")
    rmsf_apo,  _ = average_rmsf(APO_DIRS,  GRO, XTC, STRIDE, "APO")

    # ── Contact matrices ───────────────────────────────────────────────
    print("\n[3/5] Contact matrices")
    contact_fad  = load_or_compute_contact(FAD_DIRS,  "FAD")
    contact_pcar = load_or_compute_contact(PCAR_DIRS, "PCAR")
    contact_apo  = load_or_compute_contact(APO_DIRS,  "APO")

    # ── Network metrics ────────────────────────────────────────────────
    print("\n[4/5] Network metrics")
    G_fad  = build_network(contact_fad)
    G_pcar = build_network(contact_pcar)
    G_apo  = build_network(contact_apo)
    print(f"  FAD:  {G_fad.number_of_edges()} edges")
    print(f"  PCAR: {G_pcar.number_of_edges()} edges")
    print(f"  APO:  {G_apo.number_of_edges()} edges")

    deg_fad,  bet_fad  = compute_network_metrics(G_fad,  rmsf_fad)
    deg_pcar, bet_pcar = compute_network_metrics(G_pcar, rmsf_pcar)
    deg_apo,  bet_apo  = compute_network_metrics(G_apo,  rmsf_apo)

    path_fad  = find_optimal_path(G_fad,  SOURCE_RESIDS, SINK_RESIDS,
                                   contact_fad,
                                   dict(enumerate(bet_fad)))
    path_pcar = find_optimal_path(G_pcar, SOURCE_RESIDS, SINK_RESIDS,
                                   contact_pcar,
                                   dict(enumerate(bet_pcar)))
    path_apo  = find_optimal_path(G_apo,  SOURCE_RESIDS, SINK_RESIDS,
                                   contact_apo,
                                   dict(enumerate(bet_apo)))

    for tag, path in [("FAD",path_fad),("PCAR",path_pcar),("APO",path_apo)]:
        if path:
            print(f"\n  {tag} path ({len(path)} res): "
                  + " → ".join(res_label(p+1) for p in path))

    # ── CSV ───────────────────────────────────────────────────────────
    with open("Network_summary.csv","w") as f:
        f.write("Residue,RMSF_FAD,RMSF_PCAR,RMSF_APO,"
                "Degree_FAD,Degree_PCAR,Degree_APO,"
                "Betweenness_FAD,Betweenness_PCAR,Betweenness_APO,"
                "On_FAD_path,On_PCAR_path,On_APO_path\n")
        for i in range(N_RESIDUES):
            f.write(f"{i+1},"
                    f"{rmsf_fad[i]:.4f},{rmsf_pcar[i]:.4f},{rmsf_apo[i]:.4f},"
                    f"{deg_fad[i]:.6f},{deg_pcar[i]:.6f},{deg_apo[i]:.6f},"
                    f"{bet_fad[i]:.6f},{bet_pcar[i]:.6f},{bet_apo[i]:.6f},"
                    f"{'yes' if path_fad  and i in path_fad  else 'no'},"
                    f"{'yes' if path_pcar and i in path_pcar else 'no'},"
                    f"{'yes' if path_apo  and i in path_apo  else 'no'}\n")
    print("  Saved: Network_summary.csv")

    # ── Figures ───────────────────────────────────────────────────────
    print("\n[5/5] Figures")
    fig_dccm(dccm_fad, dccm_pcar, dccm_apo, N_RESIDUES)
    fig_network_metrics(rmsf_fad, rmsf_pcar, rmsf_apo,
                         deg_fad,  deg_pcar,  deg_apo,
                         bet_fad,  bet_pcar,  bet_apo,
                         path_fad, path_pcar, path_apo, N_RESIDUES)
    fig_allosteric_path(bet_fad, bet_pcar, bet_apo,
                         path_fad, path_pcar, path_apo,
                         N_RESIDUES, contact_fad, contact_pcar, contact_apo)

    print("\n" + "="*65)
    print("COMPLETE")
    for f in ["DCCM_comparison.png","Network_metrics.png",
              "Allosteric_path.png","Network_summary.csv"]:
        print(f"  [{'OK' if os.path.exists(f) else 'miss'}] {f}")


if __name__ == "__main__":
    main()