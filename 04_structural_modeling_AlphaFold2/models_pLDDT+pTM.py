#!/usr/bin/env python3
"""
Renders AlphaFold2 models via PyMOL, labels each PNG with its
model/seed/pLDDT/pTM, and merges all labeled images into one montage
figure -- for either the c-state or m-state model set.

Prerequisites:
    sudo apt install pymol imagemagick

Run from the directory containing render_plddt.py and the matching
rank file (c_rank.txt or m_rank.txt) for that state.

Usage:
    python3 render_and_label_models.py c
    python3 render_and_label_models.py m
"""

import glob
import os
import re
import subprocess
import sys

STATE_CONFIG = {
    "c": {"rank_file": "c_rank.txt", "montage_out": "C_state_80models.png"},
    "m": {"rank_file": "m_rank.txt", "montage_out": "M_state_80models.png"},
}

MODEL_SEED_RE = re.compile(r".*model_(\d+)_seed_(\d+)")
PLDDT_RE = re.compile(r"pLDDT=([0-9.]+)")
PTM_RE = re.compile(r"pTM=([0-9.]+)")


def render():
    subprocess.run(["pymol", "-cq", "render_plddt.py"], check=True)


def find_rank_line(rank_file, model, seed):
    target = f"model_{model}_seed_{seed}"
    with open(rank_file) as f:
        for line in f:
            if target in line:
                return line
    return None


def label_images(rank_file):
    for f in sorted(glob.glob("*.png")):
        name = os.path.splitext(f)[0]
        m = MODEL_SEED_RE.match(name)
        if not m:
            print(f"  skip (no model/seed match): {f}")
            continue
        model, seed = m.group(1), m.group(2)

        line = find_rank_line(rank_file, model, seed)
        if line is None:
            print(f"  WARNING: no rank entry for model_{model}_seed_{seed}")
            continue

        plddt_m = PLDDT_RE.search(line)
        ptm_m = PTM_RE.search(line)
        plddt = plddt_m.group(1) if plddt_m else "NA"
        ptm = ptm_m.group(1) if ptm_m else "NA"

        label = f"M{model} S{seed}\\npLDDT={plddt}   pTM={ptm}"

        subprocess.run([
            "convert", f,
            "-background", "white",
            "-gravity", "north",
            "-splice", "0x120",
            "-gravity", "north",
            "-pointsize", "165",
            "-interline-spacing", "5",
            "-fill", "black",
            "-annotate", "+0+10", label,
            f"lab_{f}",
        ], check=True)


def make_montage(montage_out):
    lab_files = sorted(glob.glob("lab_*.png"))
    # Single -geometry (thumbnail size + border): a second, later
    # -geometry always overrides an earlier one in montage, so only
    # the final value has any effect.
    subprocess.run([
        "montage", *lab_files,
        "-tile", "10x8",
        "-geometry", "600x600+5+5",
        montage_out,
    ], check=True)


def make_small_copies():
    os.makedirs("small", exist_ok=True)
    for f in sorted(glob.glob("lab_*.png")):
        subprocess.run(
            ["convert", f, "-resize", "500x500", os.path.join("small", f)],
            check=True,
        )


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in STATE_CONFIG:
        sys.exit("Usage: python3 render_and_label_models.py {c|m}")

    cfg = STATE_CONFIG[sys.argv[1]]

    render()
    label_images(cfg["rank_file"])
    make_montage(cfg["montage_out"])
    make_small_copies()


if __name__ == "__main__":
    main()
