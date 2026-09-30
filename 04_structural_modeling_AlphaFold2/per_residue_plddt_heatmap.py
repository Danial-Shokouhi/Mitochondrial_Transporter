#!/usr/bin/env python3
"""
Builds a per-residue pLDDT heatmap (models x residues) from a folder
of AlphaFold2 model .pdb files, for either the c-state or m-state
model set. pLDDT values are read from the CA atoms' B-factor column.

Run from the directory containing that state's .pdb files.

Usage:
    python3 plddt_heatmap.py c
    python3 plddt_heatmap.py m
"""

import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

STATE_CONFIG = {
    "c": {"title": "C-state pLDDT Heatmap", "out": "C_state_pLDDT_heatmap.png"},
    "m": {"title": "M-state pLDDT Heatmap", "out": "M_state_pLDDT_heatmap.png"},
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in STATE_CONFIG:
        sys.exit("Usage: python3 plddt_heatmap.py {c|m}")
    cfg = STATE_CONFIG[sys.argv[1]]

    files = sorted(glob.glob("*.pdb"))

    plddt_matrix = []
    for pdb in files:
        plddt = []
        with open(pdb) as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom = line[12:16].strip()
                    if atom == "CA":
                        bfactor = float(line[60:66])
                        plddt.append(bfactor)
        plddt_matrix.append(plddt)

    plddt_matrix = np.array(plddt_matrix)

    plt.figure(figsize=(15, 12))
    plt.imshow(
        plddt_matrix,
        aspect='auto',
        cmap='jet_r',
        vmin=50,
        vmax=100
    )
    plt.colorbar(label="pLDDT")
    plt.xlabel("Residue Number")
    plt.ylabel("Model Number")
    plt.title(cfg["title"])
    plt.tight_layout()
    plt.savefig(cfg["out"], dpi=1000)

    print("Done")


if __name__ == "__main__":
    main()
