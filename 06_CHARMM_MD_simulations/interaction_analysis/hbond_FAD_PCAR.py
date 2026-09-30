#!/usr/bin/env python3
"""
Automated detection of ALL protein–ligand H-bonds from MD trajectories.
Analyzes FAD and P-carnitine (PCAR) separately, 3 replicas × 1µs each.

H-bond criteria (conservative):
  D–A distance  < 3.5 Å
  D–H···A angle > 120°  (ideal > 150°, marked separately)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import os, sys, warnings, itertools
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances as mda_dist
except ImportError:
    sys.exit("Install MDAnalysis: conda install -c conda-forge mdanalysis")

def get_element(atom):
    """Infer element from atom name when Universe has no element info."""
    try:
        el = atom.element.strip()
        if el: return el.upper()
    except Exception:
        pass
    # Infer from atom name: strip leading digits, take first letter(s)
    name = atom.name.strip().lstrip("0123456789")
    if not name: return "X"
    # Common cases
    if name.startswith("CL"): return "CL"
    if name.startswith("BR"): return "BR"
    if name.startswith(("HT", "HE", "HZ", "HH", "HG",
                        "HD", "HN", "HB", "HA", "HW")):
        return "H"
    return name[0].upper()

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
GRO    = "step7_production.gro"
XTC    = "step7_production_fit.xtc"
STRIDE = 10   # analyse every Nth frame

# H-bond criteria
DA_CUTOFF   = 3.5    # Å donor–acceptor heavy atom distance
ANGLE_MIN   = 120.0  # ° D–H···A angle (conservative lower bound)
ANGLE_IDEAL = 150.0  # ° above this = "ideal" H-bond (marked on figures)

# Ligand residue name
LIGAND_RESNAME = "UNL"

#  PCAR-specific: acceptor atoms only (0-based indices) 
PCAR_ACCEPTORS_IDX = {
    4392: "O1 (COO⁻, charged)",    # carboxylate O1
    4393: "O2 (COO⁻, charged)",    # carboxylate O2
    4399: "O_ester (C–O–)",        # ester alkoxy
    4401: "O_carbonyl (C=O)",      # ester carbonyl
}
# Note: N+ (4395) is NOT an H-bond donor or acceptor — excluded

# Residue labels for annotation
KEY_RESIDS = {
    280:"ARG280", 183:"ARG183", 184:"GLU184",
    228:"TRP228",  79:"PHE79",   18:"GLN18",
    129:"LYS129", 126:"GLU126",  26:"ASP26",
    191:"MET191", 227:"PHE227",  87:"ARG87",
    196:"GLU196", 199:"LYS199", 293:"GLU293",
    296:"ARG296",   7:"ASP7",   238:"LYS238",
     29:"LYS29",  235:"ASP235",
}

COND_COLORS = {"FAD": "#C03830", "PCAR": "black"}
DPI = 1000


# H-BOND DETECTION ENGINE

def get_hbond_donors(atom_group):
    """
    Get all H-bond donor pairs (heavy_atom_idx, H_idx).
    Since GRO has no bond table, find H atoms within 1.25 Å
    of each N/O/S heavy atom (covalent bond distance).
    """
    positions = atom_group.positions
    indices   = atom_group.indices
    elements  = [get_element(a) for a in atom_group]

    heavy_mask = np.array([e in ('N','O','S') for e in elements])
    h_mask     = np.array([e == 'H' for e in elements])

    heavy_pos = positions[heavy_mask]
    h_pos     = positions[h_mask]
    heavy_idx = indices[heavy_mask]
    h_idx     = indices[h_mask]

    if len(heavy_pos) == 0 or len(h_pos) == 0:
        return []

    # Distance matrix heavy × H
    from scipy.spatial.distance import cdist
    dm = cdist(heavy_pos, h_pos)   # (n_heavy, n_H)

    donors = []
    for hi, (hrow, hidx) in enumerate(zip(dm, heavy_idx)):
        # All H within 1.25 Å are covalently bonded
        bonded_h = h_idx[hrow < 1.25]
        for hidx2 in bonded_h:
            donors.append((int(hidx), int(hidx2)))
    return donors


def get_hbond_acceptors_protein(protein_atoms):
    """Get protein H-bond acceptors: N, O, S atoms."""
    return [a.index for a in protein_atoms
            if get_element(a) in ('N','O','S')]


def calc_angle(d_pos, h_pos, a_pos):
    """D–H···A angle in degrees."""
    v1 = d_pos - h_pos
    v2 = a_pos - h_pos
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def detect_hbonds_frame(protein, ligand, is_pcar=False):
    """
    Detect all protein–ligand H-bonds in one frame.
    Returns list of (donor_resid, donor_resname, donor_atom_name,
                     acceptor_resid, acceptor_resname, acceptor_atom_name,
                     distance, angle, is_protein_donor)
    """
    found = []

    #  Case 1: protein DONOR → ligand ACCEPTOR 
    if is_pcar:
        lig_acc_indices = list(PCAR_ACCEPTORS_IDX.keys())
        lig_acc_atoms   = [ligand.atoms[ligand.atoms.indices == idx][0]
                           for idx in lig_acc_indices
                           if idx in ligand.atoms.indices]
    else:
        lig_acc_atoms = [a for a in ligand.atoms
                         if get_element(a) in ('N','O','S')]

    prot_donors = get_hbond_donors(protein)
    lig_acc_pos = np.array([a.position for a in lig_acc_atoms])

    if len(lig_acc_pos) > 0:
        for (d_idx, h_idx) in prot_donors:
            d_sel = protein.atoms[protein.atoms.indices == d_idx]
            h_sel = protein.atoms[protein.atoms.indices == h_idx]
            if len(d_sel) == 0 or len(h_sel) == 0: continue
            d_atom = d_sel[0]; h_atom = h_sel[0]
            h_pos  = h_atom.position

            for a_atom in lig_acc_atoms:
                da_dist = np.linalg.norm(d_atom.position - a_atom.position)
                if da_dist > DA_CUTOFF: continue
                angle = calc_angle(d_atom.position, h_pos, a_atom.position)
                if angle < ANGLE_MIN: continue
                found.append((
                    d_atom.resid, d_atom.resname, d_atom.name,
                    a_atom.resid, a_atom.resname, a_atom.name,
                    float(da_dist), float(angle), True))

    #  Case 2: ligand DONOR → protein ACCEPTOR (FAD only) 
    if not is_pcar:
        prot_acc_idx = get_hbond_acceptors_protein(protein.atoms)
        lig_donors   = get_hbond_donors(ligand)

        prot_acc_atoms = [protein.atoms[protein.atoms.indices == i][0]
                          for i in prot_acc_idx
                          if i in protein.atoms.indices]

        for (d_idx, h_idx) in lig_donors:
            d_sel2 = ligand.atoms[ligand.atoms.indices == d_idx]
            h_sel2 = ligand.atoms[ligand.atoms.indices == h_idx]
            if len(d_sel2) == 0 or len(h_sel2) == 0: continue
            d_atom = d_sel2[0]; h_atom = h_sel2[0]
            h_pos  = h_atom.position

            for a_atom in prot_acc_atoms:
                da_dist = np.linalg.norm(d_atom.position - a_atom.position)
                if da_dist > DA_CUTOFF: continue
                angle = calc_angle(d_atom.position, h_pos, a_atom.position)
                if angle < ANGLE_MIN: continue
                found.append((
                    d_atom.resid, d_atom.resname, d_atom.name,
                    a_atom.resid, a_atom.resname, a_atom.name,
                    float(da_dist), float(angle), False))
    return found


def hbond_key(hb):
    """Canonical key for one H-bond type."""
    return (hb[0], hb[1], hb[2], hb[3], hb[4], hb[5], hb[8])


def analyse_replica(gro_path, xtc_path, is_pcar=False, stride=STRIDE):
    """
    Analyse one replica. Returns:
      occupancy: {key: float}  — fraction of frames with H-bond
      lifetimes:  {key: [list of continuous-run lengths in ns]}
      mean_dist:  {key: float}
      mean_angle: {key: float}
    """
    u       = mda.Universe(gro_path, xtc_path)
    protein = u.select_atoms(f"protein")
    ligand  = u.select_atoms(f"resname {LIGAND_RESNAME}")

    print(f"      Protein: {len(protein)} atoms  "
          f"Ligand: {len(ligand)} atoms")

    # Per-frame H-bond presence: {key: [0/1 per frame]}
    presence  = {}
    sum_dist  = {}
    sum_angle = {}
    n_frames  = 0

    for i, ts in enumerate(u.trajectory):
        if i % stride != 0: continue
        n_frames += 1
        seen_this_frame = set()
        for hb in detect_hbonds_frame(protein, ligand, is_pcar):
            k = hbond_key(hb)
            if k in seen_this_frame:
                # Same donor heavy atom / acceptor pair already counted
                # this frame -- happens when a donor has more than one H
                # (e.g. Arg NH1/NH2, Lys NH3+) and more than one of them
                # independently satisfies the distance/angle criteria.
                # Count the frame once per key, not once per H.
                continue
            if k not in presence:
                presence[k]  = []
                sum_dist[k]  = 0.0
                sum_angle[k] = 0.0
            # Pad with zeros for any missed frames
            while len(presence[k]) < n_frames - 1:
                presence[k].append(0)
            presence[k].append(1)
            sum_dist[k]  += hb[6]
            sum_angle[k] += hb[7]
            seen_this_frame.add(k)
        # Zero-pad absent H-bonds
        for k in presence:
            if len(presence[k]) < n_frames:
                presence[k].append(0)

    # Pad to full length
    for k in presence:
        while len(presence[k]) < n_frames:
            presence[k].append(0)

    dt_ns = stride * 0.1   # 100 ps per frame × stride

    # Compute occupancy and lifetimes
    occupancy  = {}
    lifetimes  = {}
    mean_dist  = {}
    mean_angle = {}

    for k, pres in presence.items():
        arr = np.array(pres)
        occ = arr.mean()
        if occ < 0.01: continue   # skip < 1% occupancy
        occupancy[k] = float(occ)

        # Continuous run lengths
        runs = []
        in_run = False; run_len = 0
        for v in arr:
            if v == 1:
                in_run = True; run_len += 1
            else:
                if in_run: runs.append(run_len * dt_ns)
                in_run = False; run_len = 0
        if in_run: runs.append(run_len * dt_ns)
        lifetimes[k] = runs

        n_on = int(arr.sum())
        mean_dist[k]  = sum_dist[k]  / max(n_on, 1)
        mean_angle[k] = sum_angle[k] / max(n_on, 1)

    print(f"      {n_frames} frames analysed  "
          f"{len(occupancy)} H-bonds detected (occ ≥ 1%)")
    return occupancy, lifetimes, mean_dist, mean_angle


def average_replicas(rep_results):
    """Average occupancy, dist, angle across replicas. Pool lifetimes."""
    all_keys = set()
    for occ, lt, md, ma in rep_results:
        all_keys.update(occ.keys())

    avg_occ   = {}
    avg_dist  = {}
    avg_angle = {}
    pool_lt   = {}
    std_occ   = {}
    rep_occ   = {}   # per-replicate occupancy values, for violin plots

    for k in all_keys:
        occs = [r[0][k] for r in rep_results if k in r[0]]
        if not occs: continue
        avg_occ[k]   = float(np.mean(occs))
        std_occ[k]   = float(np.std(occs, ddof=1)) if len(occs)>1 else 0.0
        avg_dist[k]  = float(np.mean([r[2][k] for r in rep_results if k in r[2]]))
        avg_angle[k] = float(np.mean([r[3][k] for r in rep_results if k in r[3]]))
        pool_lt[k]   = list(itertools.chain.from_iterable(
                            [r[1][k] for r in rep_results if k in r[1]]))
        # Missing replicas (H-bond absent that replica) count as 0 occupancy
        rep_occ[k]   = [r[0].get(k, 0.0) for r in rep_results]

    return avg_occ, std_occ, pool_lt, avg_dist, avg_angle, rep_occ


# LOAD / COMPUTE

def load_or_compute(dirs, tag, is_pcar):
    cache = f"hbond_{tag}_averaged.npz"
    if os.path.exists(cache):
        print(f"  Loading cache: {cache}")
        d = np.load(cache, allow_pickle=True)
        if "rep_occ" in d.files:
            rep_occ = d["rep_occ"].item()
        else:
            # Older cache predates per-replicate tracking (needed for the
            # violin plot). Synthesize 3 pseudo-replicate values from
            # avg±std so the violin still renders; delete the cache and
            # rerun for the exact per-replicate values.
            print(f"  NOTE: {cache} has no per-replicate data — "
                  f"synthesizing from avg±std. Delete the cache and "
                  f"rerun for exact per-replicate violins.")
            avg_occ_tmp = d["avg_occ"].item()
            std_occ_tmp = d["std_occ"].item()
            rep_occ = {k: list(np.clip(
                            [avg_occ_tmp[k] - std_occ_tmp.get(k, 0.0),
                             avg_occ_tmp[k],
                             avg_occ_tmp[k] + std_occ_tmp.get(k, 0.0)],
                            0.0, 1.0))
                       for k in avg_occ_tmp}
        return (d["avg_occ"].item(), d["std_occ"].item(),
                d["pool_lt"].item(), d["avg_dist"].item(),
                d["avg_angle"].item(), rep_occ)

    print(f"\n  Computing {tag}...")
    rep_results = []
    for i, d in enumerate(dirs, 1):
        gp = os.path.join(d, GRO)
        xp = os.path.join(d, XTC)
        if not (os.path.exists(gp) and os.path.exists(xp)):
            print(f"  SKIP rep{i}: files missing"); continue
        print(f"    rep{i}:")
        rep_results.append(analyse_replica(gp, xp, is_pcar))

    if not rep_results: sys.exit(f"No data for {tag}")
    avg_occ, std_occ, pool_lt, avg_dist, avg_angle, rep_occ = \
        average_replicas(rep_results)

    np.savez(cache,
             avg_occ=avg_occ, std_occ=std_occ, pool_lt=pool_lt,
             avg_dist=avg_dist, avg_angle=avg_angle, rep_occ=rep_occ)
    print(f"  Saved: {cache}")
    return avg_occ, std_occ, pool_lt, avg_dist, avg_angle, rep_occ


print("="*65)
print("H-BOND OCCUPANCY AND LIFETIME ANALYSIS")
print("FAD vs P-carnitine | 3 × 1µs replicas")
print("="*65)

fad_occ, fad_std, fad_lt, fad_dist, fad_angle, fad_rep_occ = \
    load_or_compute(FAD_DIRS, "FAD", is_pcar=False)
pcar_occ, pcar_std, pcar_lt, pcar_dist, pcar_angle, pcar_rep_occ = \
    load_or_compute(PCAR_DIRS, "PCAR", is_pcar=True)

print(f"\nFAD: {len(fad_occ)} unique H-bonds (≥1% occupancy)")
print(f"PCAR: {len(pcar_occ)} unique H-bonds (≥1% occupancy)")


# HELPER: make sorted residue label from H-bond key

def hb_label(k, is_protein_donor):
    d_resid, d_resname, d_atom, a_resid, a_resname, a_atom, prot_donor = k
    if prot_donor:
        prot_res  = d_resid; prot_atom = d_atom
        lig_atom  = a_atom
    else:
        prot_res  = a_resid; prot_atom = a_atom
        lig_atom  = d_atom
    prot_name = KEY_RESIDS.get(prot_res, f"{d_resname}{prot_res}")
    return f"{prot_name}({prot_atom}) ↔ Lig({lig_atom})"


def build_sorted(occ_dict, angle_dict, dist_dict, min_occ=0.02):
    items = [(k, v) for k, v in occ_dict.items() if v >= min_occ]
    items.sort(key=lambda x: -x[1])
    labels = [hb_label(k, k[6]) for k, _ in items]
    occs   = [v for _, v in items]
    angles = [angle_dict.get(k, 0) for k, _ in items]
    dists  = [dist_dict.get(k, 0) for k, _ in items]
    keys   = [k for k, _ in items]
    return labels, occs, angles, dists, keys


fad_labels,  fad_occs,  fad_angles,  fad_dists,  fad_keys  = \
    build_sorted(fad_occ,  fad_angle, fad_dist)
pcar_labels, pcar_occs, pcar_angles, pcar_dists, pcar_keys = \
    build_sorted(pcar_occ, pcar_angle, pcar_dist)

#  Compact y-axis labels for Figure 1 (violin), in the same
# highest-occupancy-first order as fad_keys/pcar_keys above. 
FAD_LABELS_SHORT = [
    "R280@NH1↔FAD@O1",  "E126@OE1↔FAD@N3",  "R183@NH2↔FAD@O4",
    "R280@NH2↔FAD@O10", "E184@OE1↔FAD@O14", "E184@OE2↔FAD@O14",
    "E184@OE1↔FAD@O15", "E126@OE2↔FAD@N3",  "E184@OE2↔FAD@O15",
    "K129@NZ↔FAD@O5",   "G15@O↔FAD@O6",     "R280@NH1↔FAD@O10",
    "Q18@OE1↔FAD@N9",   "R280@NH2↔FAD@O11", "R183@NH1↔FAD@O4",
    "W228@NE1↔FAD@O10", "N188@ND2↔FAD@O15", "Q18@NE2↔FAD@O8",
    "R280@NH2↔FAD@O1",  "Q18@NE2↔FAD@O11",
]
PCAR_LABELS_SHORT = [
    "R280@NH2↔PCAR@O1", "R280@NH2↔PCAR@O2", "R280@NH1↔PCAR@O2",
    "R280@NH1↔PCAR@O1", "K129@NZ↔PCAR@O2",  "Q18@NE2↔PCAR@O1",
    "W228@NE1↔PCAR@O2", "R183@NH2↔PCAR@O2", "Q18@NE2↔PCAR@O2",
    "R183@NH2↔PCAR@O1", "R280@NH2↔PCAR@O4", "R183@NH1↔PCAR@O1",
    "K129@NZ↔PCAR@O1",  "K29@NZ↔PCAR@O2",   "R183@NH2↔PCAR@O4",
    "R280@NE↔PCAR@O1",  "R183@NH1↔PCAR@O2", "R280@NE↔PCAR@O2",
    "K29@NZ↔PCAR@O1",   "Q18@NE2↔PCAR@O4",
]

# Save CSVs
for tag, labels, occs, angles, dists, keys in [
        ("FAD",  fad_labels,  fad_occs,  fad_angles,  fad_dists,  fad_keys),
        ("PCAR", pcar_labels, pcar_occs, pcar_angles, pcar_dists, pcar_keys)]:
    with open(f"HBond_{tag}.csv", "w") as f:
        f.write("Residue,ProtAtom,LigAtom,ProtDonor,"
                "Occupancy,MeanDist_A,MeanAngle_deg\n")
        for k, occ, ang, dist in zip(keys, occs, angles, dists):
            f.write(f"{KEY_RESIDS.get(k[0],k[1]+str(k[0]))},"
                    f"{k[2]},{k[5]},{k[6]},"
                    f"{occ:.4f},{dist:.3f},{ang:.1f}\n")
    print(f"Saved: HBond_{tag}.csv")


# FIGURE 1 (merged) — Lifetime distributions, colored by occupancy
# Violin shape/spread = continuous H-bond lifetime distribution (ns).
# Violin fill color   = mean occupancy (gradient, own colorbar).

# Limit to top N for clarity
TOP_N = 20

def make_lifetime_violin_ax(ax, custom_labels, keys, occs, angles, dists,
                             lt_dict, color, title, top_n=TOP_N):
    n = min(len(keys), top_n)
    lbls    = custom_labels[:n][::-1]
    keys_v  = keys[:n][::-1]
    occ_v   = np.array(occs[:n][::-1])
    ang_v   = np.array(angles[:n][::-1])
    dist_v  = np.array(dists[:n][::-1])

    y    = np.arange(n)
    cmap = LinearSegmentedColormap.from_list(
        "occ", ["#FFFFFF", color], N=256)

    # Soft alternating row bands — purely a reading aid.
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#000000",
                       alpha=0.035, zorder=0, linewidth=0)

    # Continuous-lifetime distributions (ns) for each H-bond
    data_rows = [np.array(lt_dict.get(k, [])) for k in keys_v]
    violin_rows = [d if len(d) > 1 else np.array([0.0, 1e-6])
                   for d in data_rows]

    parts = ax.violinplot(violin_rows, positions=y, vert=False,
                           widths=0.80, showmedians=False,
                           showextrema=True, points=200,
                           bw_method=0.35)
    for i, pc in enumerate(parts['bodies']):
        # Fill color encodes mean occupancy
        pc.set_facecolor(cmap(0.35 + 0.65 * occ_v[i]))
        pc.set_edgecolor(color)
        pc.set_alpha(0.90)
        pc.set_linewidth(1.0)
        pc.set_zorder(3)
    for key in ('cbars', 'cmins', 'cmaxes'):
        parts[key].set_color(color)
        parts[key].set_linewidth(1.1)
        parts[key].set_zorder(3)

    # Individual continuous-lifetime events
    for i, d in enumerate(data_rows):
        if len(d) == 0: continue
        jitter_y = np.random.uniform(-0.12, 0.12, len(d))
        ax.scatter(d, np.full(len(d), y[i]) + jitter_y,
                   color=color, alpha=0.5, s=15, zorder=4)

    # Mean marker — crisp white disc with a colored ring
    means = np.array([np.mean(d) if len(d) else 0.0 for d in data_rows])
    ax.scatter(means, y, color="white", edgecolors=color,
               s=55, zorder=6, linewidth=1.5)

    # Angle quality indicator — placed just past each row's real max
    # lifetime, offset scaled to the panel's own range so it clears
    # the longest whisker without crowding the short ones.
    row_max = np.array([d.max() if len(d) else 0.0 for d in data_rows])
    overall_max = row_max.max() if row_max.max() > 0 else 1.0
    offset = 0.035 * overall_max
    for i in range(n):
        gx = row_max[i] + offset
        if ang_v[i] >= ANGLE_IDEAL:
            ax.scatter(gx, y[i], marker="*", s=140,
                      facecolor="black", edgecolor="black",
                      linewidth=0.7, zorder=7)
        else:
            ax.scatter(gx, y[i], marker="X", s=85,
                      facecolor="black", edgecolor="white",
                      linewidth=0.6, zorder=7)

    # Mean distance is shown alongside the residue label on the y-axis.
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{l}   ·  {d:.1f} Å" for l, d in zip(lbls, dist_v)],
        fontsize=9, fontweight="bold")
    ax.set_xlabel("Continuous lifetime (ns)", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02 * overall_max, overall_max + offset * 2.4)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title(title, fontsize=16, fontweight="bold", color=color,
                pad=10)
    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#333333")
    ax.spines[["left","bottom"]].set_linewidth(1.1)
    ax.tick_params(colors="#333333")

    # Mean-lifetime label — a fixed axes-fraction margin (x stays put
    # regardless of data range) so it can never collide with a long
    # whisker, no matter how far a given H-bond's lifetime extends.
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i in range(n):
        if len(data_rows[i]) == 0: continue
        ax.text(1.02, y[i], f"μ={means[i]:.2f} ns",
                va="center", ha="left", fontsize=7, color=color,
                fontweight="bold", transform=trans, clip_on=False)

    # Colourbar-style legend strip encoding occupancy (the x-axis
    # itself shows lifetime, not occupancy).
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(0, 1))
    sm.set_array([])
    return sm


fig1, (ax_fad, ax_pcar) = plt.subplots(
    1, 2, figsize=(22, max(10, TOP_N*0.55)))
fig1.patch.set_facecolor("white")

sm_f = make_lifetime_violin_ax(ax_fad, FAD_LABELS_SHORT, fad_keys, fad_occs,
                                fad_angles, fad_dists, fad_lt,
                                COND_COLORS["FAD"],
                                "FAD — H-bond Lifetimes")
sm_p = make_lifetime_violin_ax(ax_pcar, PCAR_LABELS_SHORT, pcar_keys, pcar_occs,
                                pcar_angles, pcar_dists, pcar_lt,
                                COND_COLORS["PCAR"],
                                "PCar — H-bond Lifetimes")

for ax, sm, col in [(ax_fad, sm_f, COND_COLORS["FAD"]),
                     (ax_pcar, sm_p, COND_COLORS["PCAR"])]:
    cax = ax.inset_axes([1.20, 0.0, 0.03, 1.0])
    cb  = fig1.colorbar(sm, cax=cax)
    cb.set_label("Occupancy", fontsize=13)
    cb.ax.tick_params(labelsize=12)

# Quality legend (occupancy lives in the color gradient / colorbar
# rather than as a separate threshold marker)
leg_handles = [
    Line2D([0],[0], marker="*", color="w",
           markerfacecolor="black", markeredgecolor="black",
           markersize=14, label=f"Ideal (angle ≥ {ANGLE_IDEAL:.0f}°)"),
    Line2D([0],[0], marker="X", color="w",
           markerfacecolor="black", markeredgecolor="black",
           markersize=9, label=f"Acceptable (angle {ANGLE_MIN:.0f}–{ANGLE_IDEAL:.0f}°)"),
]
fig1.legend(handles=leg_handles, fontsize=12, loc="lower center",
            ncol=2, bbox_to_anchor=(0.5,-0.02), frameon=False)

plt.suptitle(
    "Protein–Ligand H-bond Lifetime & Occupancy Map\n"
    f"[D–A < {DA_CUTOFF} Å, angle > {ANGLE_MIN:.0f}° | "
    f"violin = lifetime distribution, color = occupancy]",
    fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig("HBond_LifetimeOccupancy_map.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("HBond_LifetimeOccupancy_map.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: HBond_LifetimeOccupancy_map.png / .pdf")
plt.show()


# FIGURE 2 — Summary: ranked bar + angle quality + mean distance

TOP_S = 15

fig3 = plt.figure(figsize=(18, 12))
fig3.patch.set_facecolor("white")
gs3  = gridspec.GridSpec(3, 2, figure=fig3,
                          hspace=0.42, wspace=0.35)

def summary_panels(row_start, keys, labels, occs, stds,
                   angles, dists, lt_dict, col, tag):
    n = min(len(keys), TOP_S)
    y = np.arange(n)

    #  Angle quality + distance scatter 
    ax_ang = fig3.add_subplot(gs3[row_start, 1])
    ax_ang.set_facecolor("white"); ax_ang.grid(False)
    ax_ang.spines[["top","right"]].set_visible(False)

    ang_arr  = np.array(angles[:n])
    dist_arr = np.array(dists[:n])
    occ_norm = np.array(occs[:n])

    sc = ax_ang.scatter(dist_arr, ang_arr,
                        c=occ_norm, cmap="Reds" if col==COND_COLORS["FAD"]
                          else "Blues",
                        s=occ_norm*300 + 30,
                        edgecolors=col, linewidth=0.8,
                        vmin=0, vmax=1, zorder=4, alpha=0.90)

    ax_ang.axhline(ANGLE_IDEAL, color="gold",
                   lw=1.0, ls="--", alpha=0.9,
                   label=f"Ideal ≥ {ANGLE_IDEAL:.0f}°")
    ax_ang.axvline(3.0, color="darkgray",
                   lw=0.8, ls="--", alpha=0.9,
                   label="Strong H-bond ≤ 3.0 Å")

    # Label key H-bonds
    for i, (k, ang, dist, occ) in enumerate(
            zip(keys[:n], ang_arr, dist_arr, occ_norm)):
        if occ >= 0.60 or ang >= ANGLE_IDEAL:
            prot_res = KEY_RESIDS.get(k[0], f"R{k[0]}")
            ax_ang.annotate(
                f"{prot_res}\n{k[2]}",
                xy=(dist, ang),
                xytext=(dist+0.03, ang+1.5),
                fontsize=6.5, color=col, fontweight="bold")

    cbar = plt.colorbar(sc, ax=ax_ang, fraction=0.046, pad=0.04)
    cbar.set_label("Occupancy", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax_ang.set_xlabel("Mean D–A Distance (Å)",
                      fontsize=12, fontweight="bold")
    ax_ang.set_ylabel("Mean D–H···A Angle (°)",
                      fontsize=12, fontweight="bold")
    ax_ang.set_title(f"{tag}: Distance vs Angle Quality\n"
                     f"(size ∝ occupancy)",
                     fontsize=12, fontweight="bold", color=col)
    ax_ang.legend(fontsize=8, frameon=False, loc="upper left")
    ax_ang.set_ylim(ANGLE_MIN - 5, 185)
    ax_ang.set_xlim(2.0, DA_CUTOFF + 0.1)


summary_panels(0, fad_keys,  fad_labels,  fad_occs,  fad_std,
               fad_angles,  fad_dists,  fad_lt,
               COND_COLORS["FAD"],  "FAD")
summary_panels(1, pcar_keys, pcar_labels, pcar_occs, pcar_std,
               pcar_angles, pcar_dists, pcar_lt,
               COND_COLORS["PCAR"], "PCar")
plt.suptitle(
    "H-bond Quality Summary — FAD vs P-carnitine",
    fontsize=13, fontweight="bold")
plt.savefig("HBond_summary.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("HBond_summary.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: HBond_summary.png / .pdf")
plt.show()

print("\n" + "="*65)
print("COMPLETE")
for f in ["HBond_LifetimeOccupancy_map.png","HBond_summary.png",
          "HBond_FAD.csv","HBond_PCAR.csv"]:
    print(f"  [{'OK' if os.path.exists(f) else 'miss'}] {f}")
