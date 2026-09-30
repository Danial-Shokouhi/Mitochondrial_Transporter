#!/usr/bin/env python3
"""
Generates the .xvg files for the m-gate and c-gate interaction
networks using GROMACS command-line tools directly (make_ndx,
mindist, hbond, angle)

m-gate network:
  Examined in c-state systems (ligand-bound: FAD, PCAR; apo: Apo-c) --
  watches for m-gate rearrangement upon c-state ligand binding.
  Also includes the FAD-specific m-gate rearrangement network (new
  interactions that only exist when FAD is bound).

c-gate network:
  Examined in m-state systems (ligand-bound: CAR; apo: Apo-m) --
  watches for c-gate rearrangement upon carnitine binding.

Custom index-group numbers in make_ndx shift by exactly +1 between
Apo and Ligand-bound systems (the ligand adds one extra auto-generated
default group before any custom groups are created) -- this is handled
here via the --state flag rather than needing separate hardcoded
numbering per system. Downstream mindist/hbond/angle group selections
use group NAMES, not numbers, so they are unaffected by this offset.

Run once per replica directory; each must already contain
step7_production.tpr and step7_production_fit.xtc.

Usage:
  python3 gate_xvg_files.py <replica_dir> [<replica_dir> ...] --gate m --state apo
  python3 gate_xvg_files.py <replica_dir> [<replica_dir> ...] --gate m --state ligand
  python3 gate_xvg_files.py <replica_dir> [<replica_dir> ...] --gate m --fad-rearrangement
  python3 gate_xvg_files.py <replica_dir> [<replica_dir> ...] --gate c --state apo
  python3 gate_xvg_files.py <replica_dir> [<replica_dir> ...] --gate c --state ligand
"""

import argparse
import os
import subprocess
import sys

TPR = "step7_production.tpr"
XTC = "step7_production_fit.xtc"

# Custom group numbering baseline: Apo systems' first custom group is
# 28; ligand-bound systems' first custom group is 29 (one extra
# default group from the ligand). Applies uniformly to every .ndx
# file built here.
START_INDEX = {"apo": 28, "ligand": 29}


# ── Index-group definitions ─────────────────────────────────────────

M_GATE_SALTBRIDGE_GROUPS = [
    ("a 408",           "LYS29_NZ"),
    ("a 3382 3383",     "ASP235_OD"),
    ("a 1814 1815",     "GLU126_OE"),
    ("a 3434",          "LYS238_NZ"),
    ("a 355 356",       "ASP26_OD"),
    ("a 1872",          "LYS129_NZ"),
    ("a 2656 2659",     "ARG183_NH"),
]

M_GATE_SALTBRIDGE_PAIRS = [
    # (group1, group2, mindist output .xvg)
    ("LYS29_NZ",  "ASP235_OD", "dist_ASP235_LYS29.xvg"),
    ("GLU126_OE", "LYS238_NZ", "dist_GLU126_LYS238.xvg"),
    ("ASP26_OD",  "LYS129_NZ", "dist_ASP26_LYS129.xvg"),
    ("ASP26_OD",  "LYS29_NZ",  "dist_ASP26_LYS29.xvg"),
    ("LYS238_NZ", "ASP235_OD", "dist_LYS238_ASP235.xvg"),
    ("ARG183_NH", "ASP235_OD", "dist_ARG183_ASP235.xvg"),
    ("ARG183_NH", "GLU126_OE", "dist_ARG183_GLU126.xvg"),
]

M_GATE_HBOND_GROUPS = [
    ("a 3383 3382",       "ASP235_OD"),
    ("a 483 484 485",     "GLN33_NE"),
    ("a 1944 1945 3383",  "D235_Q33_ang"),
    ("a 3385",            "ASP235_O"),
    ("a 3447 3448",       "SER239_OG"),
    ("a 3447 3448 3385",  "S239_D235_ang"),
]

M_GATE_HBOND_INTERACTIONS = [
    {
        "g1": "ASP235_OD", "g2": "GLN33_NE", "ang": "D235_Q33_ang",
        "mindist_xvg": "dist_ASP235_GLN33.xvg",
        "hbnum_xvg":   "hbnum_D235_Q33.xvg",
        "hbdist_xvg":  "hbdist_D235_Q33.xvg",
        "hbang_xvg":   "hbang_D235_Q33.xvg",
    },
    {
        "g1": "ASP235_O", "g2": "SER239_OG", "ang": "S239_D235_ang",
        "mindist_xvg": "dist_ASP235_SER239.xvg",
        "hbnum_xvg":   "hbnum_S239_D235.xvg",
        "hbdist_xvg":  "hbdist_S239_D235.xvg",
        "hbang_xvg":   "hbang_S239_D235.xvg",
    },
]

# FAD_mgrr.ndx has no Apo variant -- FAD@O5/FAD@N3 atoms don't exist
# without the ligand -- so it always uses the ligand-bound offset.
FAD_MGRR_GROUPS = [
    ("a 1872",                 "LYS129_NZ"),
    ("a 4430",                 "FAD_O5"),
    ("a 1814 1815",            "GLU126_OE"),
    ("a 4392",                 "FAD_N3"),
    ("a 355 356",              "ASP26_OD"),
    ("a 4094 4097 4100",       "ARG280_NH"),
    ("a 1872 1873 1874 1875",  "LYS129_donor"),
    ("a 4392 4443",            "FAD_N3_H1_donor"),
]

FAD_MGRR_MINDIST = [
    ("LYS129_NZ", "GLU126_OE", "dist_LYS129_GLU126.xvg"),
    ("ASP26_OD",  "ARG280_NH", "dist_ASP26_ARG280.xvg"),
    ("LYS129_NZ", "FAD_O5",    "dist_LYS129_FADO5.xvg"),
    ("GLU126_OE", "FAD_N3",    "dist_GLU126_FADN3.xvg"),
]

FAD_MGRR_HBOND = [
    ("FAD_O5",    "LYS129_donor",     "hbnum_LYS129_FADO5.xvg"),
    ("GLU126_OE", "FAD_N3_H1_donor",  "hbnum_GLU126_FADN3.xvg"),
]

C_GATE_SALTBRIDGE_GROUPS = [
    ("a 1268 1271 1265", "ARG87_NH"),
    ("a 2843 2844",      "GLU196_OE"),
    ("a 2893",           "LYS199_NZ"),
    ("a 4282 4283",      "GLU293_OE"),
    ("a 4333 4336 4330", "ARG296_NH"),
    ("a 106 107",        "ASP7_OD"),
    ("r 87",             "ARG87"),    # whole-residue group, kept for
    ("r 196",            "GLU196"),   # completeness; not used below
]

C_GATE_SALTBRIDGE_PAIRS = [
    ("ARG87_NH",  "GLU196_OE", "dist_ARG87_GLU196.xvg"),
    ("LYS199_NZ", "GLU293_OE", "dist_LYS199_GLU293.xvg"),
    ("ARG296_NH", "ASP7_OD",   "dist_ARG296_ASP7.xvg"),
    ("LYS199_NZ", "GLU196_OE", "dist_LYS199_GLU196.xvg"),
]

C_GATE_HBOND_GROUPS = [
    ("a 1265 1266",           "ARG87_NE"),
    ("a 1565",                "GLN107_OE1"),
    ("a 1265 1266 1565",      "R87_Q107_ang"),
    ("a 4336 4333 4335 4338", "ARG296_NH"),
    ("a 29",                  "ALA_O"),
    ("a 4336 4338 29",        "R296_A2_ang"),
    ("a 4282 4283",           "E293_OE"),
    ("a 2824 2825",           "Y195_OH"),
    ("a 2824 2825 4282",      "E293_Y195_ang"),
    ("a 106 107",             "D7_OD"),
    ("a 4263 4264",           "Y292_OH"),
    ("a 4263 4264 106",       "D7_Y292_ang"),
]

C_GATE_HBOND_INTERACTIONS = [
    {
        "g1": "E293_OE", "g2": "Y195_OH", "ang": "E293_Y195_ang",
        "mindist_xvg": "dist_GLU293_TYR195.xvg",
        "hbnum_xvg":   "hbnum_E293_Y195.xvg",
        "hbdist_xvg":  "hbdist_E293_Y195.xvg",
        "hbang_xvg":   "hbang_E293_Y195.xvg",
    },
    {
        "g1": "D7_OD", "g2": "Y292_OH", "ang": "D7_Y292_ang",
        "mindist_xvg": "dist_ASP7_TYR292.xvg",
        "hbnum_xvg":   "hbnum_D7_Y292.xvg",
        "hbdist_xvg":  "hbdist_D7_Y292.xvg",
        "hbang_xvg":   "hbang_D7_Y292.xvg",
    },
    {
        "g1": "ARG87_NE", "g2": "GLN107_OE1", "ang": "R87_Q107_ang",
        "mindist_xvg": "dist_ARG87_GLN107.xvg",
        "hbnum_xvg":   "hbnum_R87_Q107.xvg",
        "hbdist_xvg":  "hbdist_R87_Q107.xvg",
        "hbang_xvg":   "hbang_R87_Q107.xvg",
    },
    {
        "g1": "ARG296_NH", "g2": "ALA_O", "ang": "R296_A2_ang",
        "mindist_xvg": "dist_ARG296_ALA2.xvg",
        "hbnum_xvg":   "hbnum_R296_A2.xvg",
        "hbdist_xvg":  "hbdist_R296_A2.xvg",
        "hbang_xvg":   "hbang_R296_A2.xvg",
    },
]


# ── GROMACS runners ──────────────────────────────────────────────────

def run(cmd, stdin_text, cwd, label):
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, input=stdin_text, capture_output=True,
                            text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"      FAILED: {label}")
        print(f"      {result.stderr[-500:]}")
        return False
    print(f"      OK: {label}")
    return True


def build_ndx(sim_dir, ndx_name, groups, start_index):
    """Build one .ndx file from a list of (selection_command, group_name)."""
    ndx_path = os.path.join(sim_dir, ndx_name)
    if os.path.exists(ndx_path):
        print(f"  {ndx_name} already exists -- skip")
        return True

    lines = []
    for i, (sel, name) in enumerate(groups):
        lines.append(sel)
        lines.append(f"name {start_index + i} {name}")
    lines.append("q")
    stdin_text = "\n".join(lines) + "\n"

    print(f"  Building {ndx_name} ({len(groups)} groups)...")
    return run(["gmx", "make_ndx", "-f", TPR, "-o", ndx_name],
              stdin_text, sim_dir, ndx_name)


def run_mindist(sim_dir, ndx_name, g1, g2, out_xvg):
    if os.path.exists(os.path.join(sim_dir, out_xvg)):
        print(f"    {out_xvg} exists -- skip"); return
    cmd = ["gmx", "mindist", "-s", TPR, "-f", XTC,
          "-n", ndx_name, "-od", out_xvg]
    run(cmd, f"{g1}\n{g2}\n", sim_dir, out_xvg)


def run_hbond_num(sim_dir, ndx_name, g1, g2, out_xvg):
    if os.path.exists(os.path.join(sim_dir, out_xvg)):
        print(f"    {out_xvg} exists -- skip"); return
    cmd = ["gmx", "hbond", "-s", TPR, "-f", XTC,
          "-n", ndx_name, "-num", out_xvg]
    run(cmd, f"{g1}\n{g2}\n", sim_dir, out_xvg)


def run_hbond_dist(sim_dir, ndx_name, g1, g2, out_xvg):
    if os.path.exists(os.path.join(sim_dir, out_xvg)):
        print(f"    {out_xvg} exists -- skip"); return
    cmd = ["gmx", "hbond", "-s", TPR, "-f", XTC,
          "-n", ndx_name, "-dist", out_xvg]
    run(cmd, f"{g1}\n{g2}\n", sim_dir, out_xvg)


def run_angle(sim_dir, ndx_name, ang_group, out_xvg):
    if os.path.exists(os.path.join(sim_dir, out_xvg)):
        print(f"    {out_xvg} exists -- skip"); return
    cmd = ["gmx", "angle", "-f", XTC, "-n", ndx_name,
          "-type", "angle", "-ov", out_xvg]
    run(cmd, f"{ang_group}\n", sim_dir, out_xvg)


# ── Gate pipelines ───────────────────────────────────────────────────

def run_mgate(sim_dir, state):
    print(f"\n== m-gate network ({state}) : {sim_dir} ==")
    start = START_INDEX[state]

    if build_ndx(sim_dir, "m-gate.ndx", M_GATE_SALTBRIDGE_GROUPS, start):
        for g1, g2, out_xvg in M_GATE_SALTBRIDGE_PAIRS:
            run_mindist(sim_dir, "m-gate.ndx", g1, g2, out_xvg)

    if build_ndx(sim_dir, "mg_hbond.ndx", M_GATE_HBOND_GROUPS, start):
        for inter in M_GATE_HBOND_INTERACTIONS:
            run_mindist(sim_dir, "mg_hbond.ndx",
                       inter["g1"], inter["g2"], inter["mindist_xvg"])
            run_hbond_num(sim_dir, "mg_hbond.ndx",
                         inter["g1"], inter["g2"], inter["hbnum_xvg"])
            run_hbond_dist(sim_dir, "mg_hbond.ndx",
                          inter["g1"], inter["g2"], inter["hbdist_xvg"])
            run_angle(sim_dir, "mg_hbond.ndx", inter["ang"], inter["hbang_xvg"])


def run_fad_mgate_rearrangement(sim_dir):
    print(f"\n== m-gate rearrangement in FAD-bound systems : {sim_dir} ==")
    start = START_INDEX["ligand"]   # FAD@O5/N3 only exist when FAD is bound

    if build_ndx(sim_dir, "FAD_mgrr.ndx", FAD_MGRR_GROUPS, start):
        for g1, g2, out_xvg in FAD_MGRR_MINDIST:
            run_mindist(sim_dir, "FAD_mgrr.ndx", g1, g2, out_xvg)
        for g1, g2, out_xvg in FAD_MGRR_HBOND:
            run_hbond_num(sim_dir, "FAD_mgrr.ndx", g1, g2, out_xvg)


def run_cgate(sim_dir, state):
    print(f"\n== c-gate network ({state}) : {sim_dir} ==")
    start = START_INDEX[state]

    if build_ndx(sim_dir, "c-gate.ndx", C_GATE_SALTBRIDGE_GROUPS, start):
        for g1, g2, out_xvg in C_GATE_SALTBRIDGE_PAIRS:
            run_mindist(sim_dir, "c-gate.ndx", g1, g2, out_xvg)

    if build_ndx(sim_dir, "cg_hbond.ndx", C_GATE_HBOND_GROUPS, start):
        for inter in C_GATE_HBOND_INTERACTIONS:
            run_mindist(sim_dir, "cg_hbond.ndx",
                       inter["g1"], inter["g2"], inter["mindist_xvg"])
            run_hbond_num(sim_dir, "cg_hbond.ndx",
                         inter["g1"], inter["g2"], inter["hbnum_xvg"])
            run_hbond_dist(sim_dir, "cg_hbond.ndx",
                          inter["g1"], inter["g2"], inter["hbdist_xvg"])
            run_angle(sim_dir, "cg_hbond.ndx", inter["ang"], inter["hbang_xvg"])


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replica_dirs", nargs="+",
                    help="One or more replica directories, each containing "
                         f"{TPR} and {XTC}")
    ap.add_argument("--gate", choices=["m", "c"], required=True)
    ap.add_argument("--state", choices=["apo", "ligand"],
                    help="Required unless --fad-rearrangement is used")
    ap.add_argument("--fad-rearrangement", action="store_true",
                    help="Run the FAD-specific m-gate rearrangement network "
                         "instead of the main m-gate salt-bridge/H-bond network "
                         "(only valid with --gate m)")
    args = ap.parse_args()

    if args.fad_rearrangement and args.gate != "m":
        sys.exit("--fad-rearrangement is only valid with --gate m")
    if not args.fad_rearrangement and args.state is None:
        sys.exit("--state {apo,ligand} is required unless --fad-rearrangement is used")

    for sim_dir in args.replica_dirs:
        tpr_path = os.path.join(sim_dir, TPR)
        xtc_path = os.path.join(sim_dir, XTC)
        if not (os.path.exists(tpr_path) and os.path.exists(xtc_path)):
            print(f"SKIP {sim_dir}: {TPR}/{XTC} not found")
            continue

        if args.gate == "m":
            if args.fad_rearrangement:
                run_fad_mgate_rearrangement(sim_dir)
            else:
                run_mgate(sim_dir, args.state)
        else:
            run_cgate(sim_dir, args.state)

    print("\nDone.")


if __name__ == "__main__":
    main()
