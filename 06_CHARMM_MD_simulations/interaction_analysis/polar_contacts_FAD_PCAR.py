#!/usr/bin/env python3
"""
Part 1: Run gmx make_ndx + gmx dist for all FAD and PCAR
Part 2: Parse XVG files, compute mean±SD across replicas, and produce figures
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import subprocess, os, sys, tempfile, shutil, warnings
warnings.filterwarnings("ignore")

# DIRECTORIES

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

TPR = "step7_production.tpr"
XTC = "step7_production_fit.xtc"

FAD_COLOR  = "#C03830"
PCAR_COLOR = "#1A1A1A"
FAD_SD     = "#F0A09D"
PCAR_SD    = "#A0A0A0"

CUTOFF     = 0.35   # nm — interaction threshold
DPI        = 1000
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICK  = 13
FONT_LEG   = 13


# 
# INTERACTION DEFINITIONS
# All atom numbers are GROMACS 1-based
# 

FAD_INTERACTIONS = [
    {
        "xvg":    "mindist_ARG183_NH_FAD_O4.xvg",
        "label":  "ARG183(NH)\n↔ FAD O4",
        "short":  "ARG183–O4",
        "resid":  183,
        "g1_atoms": [4429],
        "g1_name":  "FAD_O4",
        "g2_atoms": [2656, 2659],
        "g2_name":  "ARG183_NH",
        "type":   "H-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_ARG280_NH_FAD_PhosO.xvg",
        "label":  "ARG280(NH)\n↔ FAD PO4",
        "short":  "ARG280–PO4",
        "resid":  280,
        "g1_atoms": [4426,4427,4428,4434,4435,4436,4437],
        "g1_name":  "FAD_PhosO",
        "g2_atoms": [4097,4100,4094],
        "g2_name":  "ARG280_NH",
        "type":   "Salt bridge/\nH-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_GLU184_OE_FAD_O14_15.xvg",
        "label":  "GLU184(OE)\n↔ FAD O14/15",
        "short":  "GLU184–O14/15",
        "resid":  184,
        "g1_atoms": [4439,4440],
        "g1_name":  "FAD_O14_15",
        "g2_atoms": [2675,2676],
        "g2_name":  "GLU184_OE",
        "type":   "H-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_GLU126_OE_FAD_N3.xvg",
        "label":  "GLU126(OE)\n↔ FAD N3",
        "short":  "GLU126–N3",
        "resid":  126,
        "g1_atoms": [4392],
        "g1_name":  "FAD_N3",
        "g2_atoms": [1814,1815],
        "g2_name":  "GLU126_OE",
        "type":   "H-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_LYS129_NZ_FAD_O5.xvg",
        "label":  "LYS129(NZ)\n↔ FAD O5",
        "short":  "LYS129–O5",
        "resid":  129,
        "g1_atoms": [4430],
        "g1_name":  "FAD_O5",
        "g2_atoms": [1872],
        "g2_name":  "LYS129_NZ",
        "type":   "H-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_TRP228_NE_FAD_PhosO.xvg",
        "label":  "TRP228(NE)\n↔ FAD PO4",
        "short":  "TRP228–PO4",
        "resid":  228,
        "g1_atoms": [4426,4435,4436],
        "g1_name":  "FAD_O1_10_11",
        "g2_atoms": [3269,3270],
        "g2_name":  "TRP228_NE",
        "type":   "H-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_GLN18_FAD.xvg",
        "label":  "GLN18(OE/NE)\n↔ FAD O8/N9/O11",
        "short":  "GLN18–FAD",
        "resid":  18,
        "g1_atoms": [4433,4436,4398],
        "g1_name":  "FAD_O8_O11_N9",
        "g2_atoms": [232,233,234,235],
        "g2_name":  "GLN18_OE1_NE2",
        "type":   "H-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_GLY15_O_FAD_O6.xvg",
        "label":  "GLY15(O)\n↔ FAD O6",
        "short":  "GLY15–O6",
        "resid":  15,
        "g1_atoms": [4431],
        "g1_name":  "FAD_O6",
        "g2_atoms": [200],
        "g2_name":  "GLY15_O",
        "type":   "H-bond",
        "strength": "strong",
    },
]

PCAR_INTERACTIONS = [
    {
        "xvg":    "mindist_ARG183_NH_PCAR_O.xvg",
        "label":  "ARG183(NH)\n↔ PCAR O",
        "short":  "ARG183–O",
        "resid":  183,
        "g1_atoms": [4393,4394,4400,4402],
        "g1_name":  "PCAR_O",
        "g2_atoms": [2653,2656,2659],
        "g2_name":  "ARG183_NH",
        "type":   "Salt bridge",
        "strength": "occasional",
    },
    {
        "xvg":    "mindist_ARG280_NH_PCAR_O.xvg",
        "label":  "ARG280(NH)\n↔ PCAR O",
        "short":  "ARG280–O",
        "resid":  280,
        "g1_atoms": [4393,4394,4400,4402],
        "g1_name":  "PCAR_O",
        "g2_atoms": [4094,4097,4100],
        "g2_name":  "ARG280_NH",
        "type":   "Salt bridge/\nH-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_GLU184_OE_PCAR_N.xvg",
        "label":  "GLU184(OE)\n↔ PCAR N⁺/CH",
        "short":  "GLU184–N⁺",
        "resid":  184,
        "g1_atoms": [4396,4397,4398,4399],
        "g1_name":  "PCAR_N_C",
        "g2_atoms": [2675,2676],
        "g2_name":  "GLU184_OE",
        "type":   "Salt bridge",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_GLU126_OE_PCAR.xvg",
        "label":  "GLU126(OE)\n↔ PCAR N⁺/CH",
        "short":  "GLU126–N⁺",
        "resid":  126,
        "g1_atoms": [4396,4397,4398,4399],
        "g1_name":  "PCAR_N_C",
        "g2_atoms": [1814,1815],
        "g2_name":  "GLU126_OE",
        "type":   "Salt bridge",
        "strength": "occasional",
    },
    {
        "xvg":    "mindist_LYS129_NZ_PCAR_O.xvg",
        "label":  "LYS129(NZ)\n↔ PCAR O",
        "short":  "LYS129–O",
        "resid":  129,
        "g1_atoms": [4393,4394,4400,4402],
        "g1_name":  "PCAR_O",
        "g2_atoms": [1872],
        "g2_name":  "LYS129_NZ",
        "type":   "Salt bridge",
        "strength": "occasional",
    },
    {
        "xvg":    "mindist_GLN18_OE_PCAR_O.xvg",
        "label":  "GLN18(OE/NE)\n↔ PCAR O",
        "short":  "GLN18–O",
        "resid":  18,
        "g1_atoms": [4393,4394,4400,4402],
        "g1_name":  "PCAR_O",
        "g2_atoms": [232,233,234,235],
        "g2_name":  "GLN18_OE_NE",
        "type":   "H-bond",
        "strength": "strong",
    },
    {
        "xvg":    "mindist_LYS29_NZ_PCAR_O.xvg",
        "label":  "LYS29(NZ)\n↔ PCAR O",
        "short":  "LYS29–O",
        "resid":  29,
        "g1_atoms": [4393,4394,4400,4402],
        "g1_name":  "PCAR_O",
        "g2_atoms": [408],
        "g2_name":  "LYS29_NZ",
        "type":   "Salt bridge/\nH-bond",
        "strength": "occasional",
    },
]


# PART 1 — RUN GROMACS MINDIST

def make_ndx_input(g1_atoms, g1_name, g2_atoms, g2_name):
    """Build stdin text for gmx make_ndx."""
    lines = []
    lines.append(f"a {' '.join(str(a) for a in g1_atoms)}")
    lines.append(f"name 29 {g1_name}")
    lines.append(f"a {' '.join(str(a) for a in g2_atoms)}")
    lines.append(f"name 30 {g2_name}")
    lines.append("q")
    return "\n".join(lines) + "\n"


def run_mindist(sim_dir, interaction, tag, rep_idx):
    """
    Run gmx make_ndx + gmx mindist for one interaction in one replica.
    XVG is written to sim_dir.
    Returns the XVG path.
    """
    xvg_path = os.path.join(sim_dir, interaction["xvg"])

    if os.path.exists(xvg_path):
        print(f"    [{tag} rep{rep_idx}] skip (exists): "
              f"{interaction['xvg']}")
        return xvg_path

    tpr_path = os.path.join(sim_dir, TPR)
    xtc_path = os.path.join(sim_dir, XTC)

    if not os.path.exists(tpr_path) or not os.path.exists(xtc_path):
        print(f"    [{tag} rep{rep_idx}] SKIP — tpr/xtc missing in {sim_dir}")
        return None

    # Temporary ndx file
    ndx_path = os.path.join(sim_dir, f"_tmp_{tag}_{rep_idx}.ndx")

    # Step 1: make_ndx
    ndx_input = make_ndx_input(
        interaction["g1_atoms"], interaction["g1_name"],
        interaction["g2_atoms"], interaction["g2_name"])

    r1 = subprocess.run(
        ["gmx", "make_ndx", "-f", tpr_path, "-o", ndx_path],
        input=ndx_input, capture_output=True, text=True, timeout=30)

    if not os.path.exists(ndx_path):
        print(f"    [{tag} rep{rep_idx}] make_ndx FAILED: "
              f"{interaction['xvg']}")
        print(f"      {r1.stderr[-200:]}")
        return None

    # Step 2: mindist
    r2 = subprocess.run(
        ["gmx", "mindist",
         "-s", tpr_path,
         "-f", xtc_path,
         "-n", ndx_path,
         "-od", xvg_path],
        input="29\n30\n", capture_output=True, text=True, timeout=300)

    # Clean up temp ndx
    if os.path.exists(ndx_path):
        os.remove(ndx_path)

    if os.path.exists(xvg_path):
        print(f"    [{tag} rep{rep_idx}] OK: {interaction['xvg']}")
    else:
        print(f"    [{tag} rep{rep_idx}] mindist FAILED: "
              f"{interaction['xvg']}")
        print(f"      {r2.stderr[-200:]}")
    return xvg_path


def run_all_mindist():
    print("="*65)
    print("PART 1: RUNNING GROMACS MINDIST")
    print("="*65)

    for tag, dirs, interactions in [
            ("FAD",  FAD_DIRS,  FAD_INTERACTIONS),
            ("PCAR", PCAR_DIRS, PCAR_INTERACTIONS)]:
        print(f"\n {tag} ({len(interactions)} interactions × "
              f"{len(dirs)} replicas) ")
        for inter in interactions:
            print(f"  {inter['short']}:")
            for rep_idx, sim_dir in enumerate(dirs, 1):
                run_mindist(sim_dir, inter, tag, rep_idx)

    print("\nPart 1 complete.\n")


# PART 2 — PARSE XVG AND COMPUTE STATS

def parse_xvg(path, stride=10):
    """Read GROMACS mindist XVG. Returns (time_ns, dist_nm)."""
    t, d = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ('@','#','&'): continue
            p = s.split()
            if len(p) < 2: continue
            try:
                t.append(float(p[0]))
                d.append(float(p[1]))
            except ValueError:
                continue
    t = np.array(t)[::stride] / 1000.0   # ps → ns
    d = np.array(d)[::stride]
    return t, d


def load_interaction(dirs, interaction, stride=10):
    """
    Load XVG from all replicas, interpolate onto common grid.
    Returns (t_grid, mean, sd, all_traces).
    """
    traces, t_ref = [], None
    for sim_dir in dirs:
        xvg = os.path.join(sim_dir, interaction["xvg"])
        if not os.path.exists(xvg):
            print(f"  MISSING: {xvg}")
            continue
        t, d = parse_xvg(xvg, stride)
        if t_ref is None:
            t_ref = t
        if len(d) != len(t_ref):
            d = np.interp(t_ref, t, d)
        traces.append(d)

    if not traces:
        return None, None, None, []
    arr  = np.array(traces)
    mean = arr.mean(axis=0)
    sd   = arr.std(axis=0, ddof=1) if len(traces)>1 else np.zeros_like(mean)
    return t_ref, mean, sd, traces


def occupancy(mean_trace, cutoff=CUTOFF):
    return 100.0 * np.mean(mean_trace < cutoff) if mean_trace is not None else 0.0


def mean_distance(mean_trace):
    return float(np.mean(mean_trace)) if mean_trace is not None else 0.0


# PART 2 — FIGURES


#  Per-residue interaction legend labels (user-defined) 
RESIDUE_TITLES = {
    183: "ARG183",
    280: "ARG280",
    184: "GLU184",
    126: "GLU126",
    129: "LYS129",
    18:  "GLN18",
    228: "TRP228",
    15:  "GLY15",
    29:  "LYS29",
}

# FAD legend label per residue (interaction description)
FAD_LEGEND = {
    183: "FAD@O4↔R183@NH",
    280: "FAD@PhosO↔R280@NH",
    184: "FAD@O14/15↔E184@OE",
    126: "FAD@N3↔E126@OE",
    129: "FAD@O5↔K129@NZ",
    18:  "FAD@O8/O11/N9↔Q18@OE/NE",
    228: "FAD@PhosO↔W228@NE",
    15:  "FAD@O6↔G15@O",
}

# PCAR legend label per residue
PCAR_LEGEND = {
    183: "PCAR@O1-4↔R183@NH",
    280: "PCAR@O1-4↔R280@NH",
    184: "PCAR@N⁺/CH↔E184@OE",
    126: "PCAR@N⁺/CH↔E126@OE",
    129: "PCAR@O1-4↔K129@NZ",
    18:  "PCAR@O1-4↔Q18@OE/NE",
    228: None,  # FAD only
    15:  None,  # FAD only
    29:  "PCAR@O1-4↔K29@NZ",
}

def fig_timeseries(fad_data, pcar_data):
    """
    Figure 1: Grid of distance time series.
    Title = residue name only (e.g. ARG183).
    Legend = specific interaction label + occupancy %.
    Single shared cutoff line in figure-level legend at top.
    """
    fad_resids  = {inter["resid"]: inter for inter in FAD_INTERACTIONS}
    pcar_resids = {inter["resid"]: inter for inter in PCAR_INTERACTIONS}
    all_resids  = sorted(set(fad_resids) | set(pcar_resids))

    ncols = 3
    nrows = int(np.ceil(len(all_resids) / ncols))

    # Extra top margin for the shared figure-level legend
    fig   = plt.figure(figsize=(ncols*5, nrows*3.2 + 0.8))
    fig.patch.set_facecolor("white")
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig,
                               hspace=0.60, wspace=0.32,
                               top=0.88, bottom=0.06)

    for idx, resid in enumerate(all_resids):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)

        has_fad  = resid in fad_resids  and fad_resids[resid]["xvg"]  in fad_data
        has_pcar = resid in pcar_resids and pcar_resids[resid]["xvg"] in pcar_data

        leg_lines = []   # collect handles for this subplot legend

        if has_fad:
            fd   = fad_data[fad_resids[resid]["xvg"]]
            t, m, s = fd["t"], fd["mean"], fd["sd"]
            ax.fill_between(t, m-s, m+s, color=FAD_SD, alpha=1.00)
            occ_f = occupancy(m)
            lbl_f = f"{FAD_LEGEND.get(resid,'FAD')} ({occ_f:.0f}%)"
            line_f, = ax.plot(t, m, color=FAD_COLOR, lw=1.4, label=lbl_f)
            leg_lines.append(line_f)

        if has_pcar:
            pd   = pcar_data[pcar_resids[resid]["xvg"]]
            t, m, s = pd["t"], pd["mean"], pd["sd"]
            ax.fill_between(t, m-s, m+s, color=PCAR_SD, alpha=1.00)
            occ_p = occupancy(m)
            lbl_p = f"{PCAR_LEGEND.get(resid,'PCAR')} ({occ_p:.0f}%)"
            line_p, = ax.plot(t, m, color=PCAR_COLOR, lw=1.2, label=lbl_p)
            leg_lines.append(line_p)

        # Cutoff — draw line but NOT in per-panel legend
        ax.axhline(CUTOFF, color="gray", lw=0.8, ls="--", alpha=0.55)

        # Y limits — tight
        all_vals = []
        for has, dkey, ddict in [(has_fad, fad_resids, fad_data),
                                  (has_pcar, pcar_resids, pcar_data)]:
            if has:
                xvg = ddict[dkey[resid]["xvg"]]
                all_vals.extend((xvg["mean"]-xvg["sd"]).tolist())
                all_vals.extend((xvg["mean"]+xvg["sd"]).tolist())
        if all_vals:
            ymin = max(0.0, np.min(all_vals)-0.02)
            ymax = np.max(all_vals)+0.04
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(0, 1000)
        ax.tick_params(labelsize=FONT_TICK)
        if row == nrows-1:
            ax.set_xlabel("Time (ns)", fontsize=FONT_LABEL,
                          fontweight="bold")
        if col == 0:
            ax.set_ylabel("Distance (nm)",
                          fontsize=FONT_LABEL, fontweight="bold")

        # Interaction type tag (FAD's type takes priority if both present)
        inter_type = None
        if has_fad:
            inter_type = fad_resids[resid]["type"]
        elif has_pcar:
            inter_type = pcar_resids[resid]["type"]
        if inter_type:
            type_col = "black" if inter_type=="Salt bridge" else "black"
            ax.text(0.03, 0.97, inter_type,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.5, color=type_col,
                    fontweight="bold", alpha=0.55, style="italic")

        # Title = residue name only
        ax.set_title(RESIDUE_TITLES.get(resid, str(resid)),
                     fontsize=FONT_TITLE+1, fontweight="bold", pad=6)

        # Per-panel legend (interaction labels + occupancy, no cutoff)
        if leg_lines:
            ax.legend(handles=leg_lines, fontsize=8.5,
                      frameon=False, loc="upper right")

    # Turn off empty panels
    for idx in range(len(all_resids), nrows*ncols):
        r, c = divmod(idx, ncols)
        fig.add_subplot(gs[r, c]).axis("off")

    #  Single shared figure-level legend at TOP 
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
               label=f"Cutoff {CUTOFF} nm"),
    ]
    fig.legend(handles=fig_handles, fontsize=FONT_LEG+0.5,
               loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.97),
               frameon=False, columnspacing=1.2)

    plt.suptitle(
        "Polar interactions Profile",
        fontsize=FONT_TITLE+1, fontweight="bold", y=1.01)
    plt.savefig("interactome_timeseries.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("interactome_timeseries.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: interactome_timeseries.png / .pdf")
    plt.show()


def fig_radar(fad_data, pcar_data):
    """
    Figure 2: Radar (spider) chart — occupancy % for all interactions.
    FAD (red) vs PCAR (black), each on a separate petal.
    """
    # Build unified label set
    fad_shorts  = [(i["short"], i["xvg"]) for i in FAD_INTERACTIONS]
    pcar_shorts = [(i["short"], i["xvg"]) for i in PCAR_INTERACTIONS]

    # All unique interaction short labels
    all_labels = list({s for s,_ in fad_shorts} |
                      {s for s,_ in pcar_shorts})
    all_labels.sort()
    N = len(all_labels)

    fad_occ  = np.zeros(N)
    pcar_occ = np.zeros(N)

    for j, lbl in enumerate(all_labels):
        for s, xvg in fad_shorts:
            if s == lbl and xvg in fad_data:
                fad_occ[j] = occupancy(fad_data[xvg]["mean"])
        for s, xvg in pcar_shorts:
            if s == lbl and xvg in pcar_data:
                pcar_occ[j] = occupancy(pcar_data[xvg]["mean"])

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fad_vals  = fad_occ.tolist()  + [fad_occ[0]]
    pcar_vals = pcar_occ.tolist() + [pcar_occ[0]]

    fig, ax = plt.subplots(figsize=(9, 9),
                            subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # Gridlines at 25, 50, 75, 100 %
    for grid_v in [25, 50, 75, 100]:
        ax.plot(angles, [grid_v]*len(angles),
                color="gray", lw=0.5, ls="--", alpha=0.45)
        ax.text(0, grid_v, f"{grid_v}%",
                ha="center", va="center",
                fontsize=8.5, color="gray")

    ax.fill(angles, fad_vals,  color=FAD_COLOR,  alpha=0.22)
    ax.fill(angles, pcar_vals, color=PCAR_COLOR,  alpha=0.14)
    ax.plot(angles, fad_vals,  color=FAD_COLOR,  lw=2.2,
            marker="o", markersize=6, label="cFAD-bound")
    ax.plot(angles, pcar_vals, color=PCAR_COLOR, lw=1.8,
            marker="s", markersize=5, ls="--", label="cPCar-bound")

    # Mark shared interactions
    for j, lbl in enumerate(all_labels):
        if fad_occ[j] > 0 and pcar_occ[j] > 0:
            ax.scatter(angles[j], max(fad_occ[j], pcar_occ[j])+5,
                       color="#FFD700", s=60, zorder=6,
                       marker="*", edgecolors="none")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(True)

    leg_handles = [
        Line2D([0],[0], color=FAD_COLOR,  lw=2.2, marker="",
               markersize=7, label="cFAD-bound"),
        Line2D([0],[0], color=PCAR_COLOR, lw=1.8, marker="",
               markersize=6, label="cPCar-bound"),
    ]
    ax.legend(handles=leg_handles, fontsize=11, frameon=True,
              loc="upper right", bbox_to_anchor=(1.35, 1.15),
              framealpha=0.9)

    ax.set_title(
        "Interaction occupancy profile\n"
        "cFAD-bound vs cPCar-bound\n"
        f"(% frames with distance < {CUTOFF} nm)",
        fontsize=FONT_TITLE+1, fontweight="bold", pad=25)

    plt.tight_layout()
    plt.savefig("interactome_radar.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("interactome_radar.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: interactome_radar.png / .pdf")
    plt.show()


def fig_summary(fad_data, pcar_data):
    """
    Figure 3: Lollipop chart comparing mean distance for each interaction.
    Left: FAD interactions ranked by occupancy.
    Right: PCAR interactions ranked by occupancy.
    Center: shared residue comparison bars.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                             wspace=0.40, width_ratios=[1.2,1,1.2])

    def lollipop_ax(ax, interactions, data, color, title):
        labels, occs, dists = [], [], []
        for inter in interactions:
            xvg = inter["xvg"]
            if xvg not in data: continue
            labels.append(inter["short"])
            occs.append(occupancy(data[xvg]["mean"]))
            dists.append(mean_distance(data[xvg]["mean"]))

        # Sort by occupancy descending
        order = np.argsort(occs)[::-1]
        labels = [labels[i] for i in order]
        occs   = [occs[i]   for i in order]
        dists  = [dists[i]  for i in order]

        y = np.arange(len(labels))
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)

        # Occupancy as horizontal bars (faint)
        ax.barh(y, occs, 0.55, color=color, alpha=0.18,
                edgecolor="none")
        # Lollipop sticks
        for yi, occ in zip(y, occs):
            ax.plot([0, occ], [yi, yi], color=color,
                    lw=1.2, alpha=0.55)
        # Lollipop heads colored by distance
        sc = ax.scatter(occs, y, c=dists,
                        cmap="coolwarm_r",
                        vmin=0.2, vmax=0.55,
                        s=120, zorder=5,
                        edgecolors=color, linewidth=1.2)

        # Annotate occupancy + distance
        for yi, occ, dist in zip(y, occs, dists):
            ax.text(occ+1.0, yi,
                    f"  {occ:.0f}%  |  {dist:.2f} nm",
                    va="center", ha="left", fontsize=9.5, color="dimgray")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Occupancy (%)", fontsize=FONT_LABEL+1,
                      fontweight="bold")
        ax.set_xlim(0, 115)
        ax.set_title(title, fontsize=FONT_TITLE+1,
                     fontweight="bold", color=color)
        ax.axvline(50, color="gray", lw=0.7, ls="--", alpha=0.45)
        ax.axvline(80, color="gray", lw=0.5, ls="--", alpha=0.30)
        return sc

    sc_f = lollipop_ax(fig.add_subplot(gs[0]),
                        FAD_INTERACTIONS, fad_data,
                        FAD_COLOR,
                        "cFAD-bound\nInteraction Occupancy")
    sc_p = lollipop_ax(fig.add_subplot(gs[2]),
                        PCAR_INTERACTIONS, pcar_data,
                        PCAR_COLOR,
                        "cPCar-bound\nInteraction Occupancy")

    # Center panel — shared residue direct comparison
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.set_facecolor("white"); ax_mid.grid(False)
    ax_mid.spines[["top","right"]].set_visible(False)

    shared_resids = set(i["resid"] for i in FAD_INTERACTIONS) & \
                    set(i["resid"] for i in PCAR_INTERACTIONS)
    fad_r  = {i["resid"]: i for i in FAD_INTERACTIONS}
    pcar_r = {i["resid"]: i for i in PCAR_INTERACTIONS}

    shared_labels, fad_occs_s, pcar_occs_s = [], [], []
    for resid in sorted(shared_resids):
        fi = fad_r[resid]; pi = pcar_r[resid]
        if fi["xvg"] not in fad_data or pi["xvg"] not in pcar_data:
            continue
        shared_labels.append(f"R{resid}")
        fad_occs_s.append(occupancy(fad_data[fi["xvg"]]["mean"]))
        pcar_occs_s.append(occupancy(pcar_data[pi["xvg"]]["mean"]))

    y_s = np.arange(len(shared_labels))
    bw  = 0.32
    ax_mid.barh(y_s + bw/2, fad_occs_s,  bw, color=FAD_COLOR,
                alpha=0.85, label="cFAD-bound")
    ax_mid.barh(y_s - bw/2, pcar_occs_s, bw, color=PCAR_COLOR,
                alpha=0.80, label="cPCar-bound")
    ax_mid.axvline(50, color="gray", lw=0.7, ls="--", alpha=0.45)
    ax_mid.set_yticks(y_s)
    ax_mid.set_yticklabels(shared_labels, fontsize=11)
    ax_mid.set_xlabel("Occupancy (%)", fontsize=FONT_LABEL+1,
                      fontweight="bold")
    ax_mid.set_title("Shared Residues\nDirect Comparison",
                     fontsize=FONT_TITLE+1, fontweight="bold",
                     color="#8B6000")
       # Shared colorbar for lollipop panels
    cbar_ax = fig.add_axes([0.01, 0.08, 0.01, 0.20])
    cb = fig.colorbar(sc_f, cax=cbar_ax)
    cb.set_label("Mean dist. (nm)", fontsize=9.5)
    cb.ax.tick_params(labelsize=7)

    plt.suptitle(
        "Polar interactions summary\n"
        f"Occupancy = % frames with distance < {CUTOFF} nm",
        fontsize=FONT_TITLE+1, fontweight="bold")
    plt.savefig("interactome_summary.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.savefig("interactome_summary.pdf",
                bbox_inches="tight", facecolor="white")
    print("  Saved: interactome_summary.png / .pdf")
    plt.show()


# MAIN

def main():
    #  Load XVG files (pre-computed manually) and compute stats 
    print("="*65)
    print("PART 2: LOADING XVG AND COMPUTING STATISTICS")
    print("="*65)

    fad_data, pcar_data = {}, {}

    print("\n FAD interactions ")
    for inter in FAD_INTERACTIONS:
        t, m, s, traces = load_interaction(FAD_DIRS, inter)
        if m is not None:
            fad_data[inter["xvg"]] = {"t": t, "mean": m, "sd": s}
            print(f"  {inter['short']:22} occ={occupancy(m):5.1f}%  "
                  f"mean={mean_distance(m):.3f} nm")

    print("\n PCAR interactions ")
    for inter in PCAR_INTERACTIONS:
        t, m, s, traces = load_interaction(PCAR_DIRS, inter)
        if m is not None:
            pcar_data[inter["xvg"]] = {"t": t, "mean": m, "sd": s}
            print(f"  {inter['short']:22} occ={occupancy(m):5.1f}%  "
                  f"mean={mean_distance(m):.3f} nm")

    if not fad_data and not pcar_data:
        print("ERROR: No XVG data loaded. "
              "Check that Part 1 ran successfully.")
        sys.exit(1)

    #  Figures 
    print("\n Generating figures ")
    fig_timeseries(fad_data, pcar_data)
    fig_radar(fad_data, pcar_data)
    fig_summary(fad_data, pcar_data)

    print("\n" + "="*65)
    print("COMPLETE")
    for f in ["interactome_timeseries.png",
              "interactome_radar.png",
              "interactome_summary.png"]:
        print(f"  [{'OK' if os.path.exists(f) else 'miss'}] {f}")


if __name__ == "__main__":
    main()
