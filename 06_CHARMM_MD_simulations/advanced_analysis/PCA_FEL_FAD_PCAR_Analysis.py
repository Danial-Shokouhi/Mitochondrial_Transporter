#!/usr/bin/env python3
"""
PCA analysis for FAD (3 replicas) and P-carnitine (3 replicas)
separately, each with their own shared eigenvector basis.
All output files are written to the current working directory with
condition-specific prefixes: fad_* and pcar_*
Run this first, then run PCA_FEL_FAD_PCAR_Figures.py for figures.
"""

import subprocess, os, sys

# ── Directories ────────────────────────────────────────────────────────
CONDITIONS = {
    "fad": {
        "dirs": [
            ".../rep1",
            ".../rep2",
            ".../rep3",
        ],
        "tpr":   ".../step7_production.tpr",
        "index": ".../index.ndx",
    },
    "pcar": {
        "dirs": [
            ".../rep1",
            ".../rep2",
            ".../rep3",
        ],
        "tpr":   ".../step7_production.tpr",
        "index": ".../index.ndx",
    },
}

CA_GROUP     = "3"   # C-alpha group in the full index
CA_NDX_GROUP = "1"   # Protein group in the minimal 300-atom index


def run_interactive(cmd, input_text, label):
    print(f"  [{label}]")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    stdout, stderr = proc.communicate(input=input_text)
    if proc.returncode != 0:
        print(f"FAILED: {label}")
        print((stdout + stderr)[-2000:])
        sys.exit(1)

def run_simple(cmd, label):
    print(f"  [{label}]")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED: {label}")
        print((r.stdout + r.stderr)[-2000:])
        sys.exit(1)


def run_condition(cond_name, cfg):
    prefix = cond_name   # "fad" or "pcar"
    dirs   = cfg["dirs"]
    tpr    = cfg["tpr"]
    idx    = cfg["index"]
    labels = [f"rep{i+1}" for i in range(len(dirs))]

    ca_ref     = f"{prefix}_ca_ref.pdb"
    ca_ndx     = f"{prefix}_ca_only.ndx"
    combo_xtc  = f"{prefix}_combined_ca.xtc"
    eigenval   = f"{prefix}_combined_eigenval.xvg"
    eigenvec   = f"{prefix}_combined_eigenvec.trr"
    average    = f"{prefix}_combined_average.pdb"

    print(f"\n{'='*60}")
    print(f"  CONDITION: {cond_name.upper()} — {len(dirs)} replicas")
    print(f"{'='*60}")

    # A: Extract CA trajectories
    ca_xtcs = []
    for d, lbl in zip(dirs, labels):
        out = os.path.join(d, f"{prefix}_ca_only.xtc")
        ca_xtcs.append(out)
        if os.path.exists(out):
            print(f"  {lbl}: {prefix}_ca_only.xtc exists — skip")
            continue
        run_interactive(
            ["gmx","trjconv","-s",tpr,
             "-f", os.path.join(d,"step7_production_fit.xtc"),
             "-n",idx,"-o",out],
            f"{CA_GROUP}\n", f"{lbl}: extract CA traj")

    # B: Concatenate
    if not os.path.exists(combo_xtc):
        run_simple(["gmx","trjcat","-f"]+ca_xtcs+
                   ["-o",combo_xtc,"-cat"],
                   "Concatenate CA trajs")
    else:
        print(f"  {combo_xtc} exists — skip")

    # C: Reference PDB
    if not os.path.exists(ca_ref):
        run_interactive(
            ["gmx","trjconv","-s",tpr,
             "-f",os.path.join(dirs[0],"step7_production_fit.xtc"),
             "-n",idx,"-dump","0","-o",ca_ref],
            f"{CA_GROUP}\n", "Extract CA reference PDB")
    else:
        print(f"  {ca_ref} exists — skip")

    # D: Minimal index
    if not os.path.exists(ca_ndx):
        run_interactive(
            ["gmx","make_ndx","-f",ca_ref,"-o",ca_ndx],
            "q\n", "Build minimal CA index")
    else:
        print(f"  {ca_ndx} exists — skip")

    # E: Covariance matrix
    if not os.path.exists(eigenval):
        run_interactive(
            ["gmx","covar","-s",ca_ref,"-f",combo_xtc,
             "-n",ca_ndx,"-o",eigenval,"-v",eigenvec,"-av",average],
            f"{CA_NDX_GROUP}\n{CA_NDX_GROUP}\n",
            "gmx covar: combined covariance")
    else:
        print(f"  {eigenval} exists — skip")

    # F: Project each replica
    pca_files = []
    for d, lbl in zip(dirs, labels):
        out    = f"{prefix}_{lbl}_pca.xvg"
        ca_xtc = os.path.join(d, f"{prefix}_ca_only.xtc")
        pca_files.append(out)
        if os.path.exists(out):
            print(f"  {lbl}: {out} exists — skip")
            continue
        run_interactive(
            ["gmx","anaeig","-s",ca_ref,
             "-f",ca_xtc,"-v",eigenvec,
             "-eig",eigenval,"-n",ca_ndx,
             "-first","1","-last","2","-2d",out],
            f"{CA_NDX_GROUP}\n{CA_NDX_GROUP}\n",
            f"{lbl}: project onto eigenvectors")

    print(f"\n  {cond_name.upper()} PCA complete.")
    print(f"  PCA files: {pca_files}")
    return pca_files


if __name__ == "__main__":
    for cond, cfg in CONDITIONS.items():
        run_condition(cond, cfg)
    print("\n\nAll conditions done. Now run: python3 combined_FEL_FAD_PCAR.py")
