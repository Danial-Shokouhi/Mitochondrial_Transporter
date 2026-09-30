#!/usr/bin/env python3
"""
Protein preparation pipeline for AutoDock Vina docking, with mandatory
per-residue formal-charge verification after PDBQT conversion.

Rationale: charge-assignment tools can silently fail on individual
residues (e.g. a charge-equilibration method delocalizing a
guanidinium/carboxylate group's charge across too many atoms under
unusual local electrostatic conditions), producing a receptor that
looks fine but docks incorrectly. Every charged/titratable residue is
independently verified by summing its own atomic partial charges after
assignment, and any residue that deviates from its expected formal
charge is flagged before docking.

Pipeline stages:
  1. Clean PDB        (remove waters/heteroatoms/altlocs)
  2. Fix structure    (PDBFixer -- missing atoms; internal gaps only,
                        not missing terminal residues)
  3. Protonate        (PDB2PQR + PROPKA -- pH-aware, per-residue pKa)
  4. Convert to PDBQT (AutoDockTools' prepare_receptor4.py, which
                        assigns charges from a residue template library;
                        OpenBabel/Gasteiger is used only as a last-resort
                        fallback -- see step4_to_pdbqt for why)
  5. Verify every charged residue's summed charge against its expected
     formal charge

Requirements:
    conda install -c conda-forge pdbfixer openmm openbabel
    pip install pdb2pqr

Usage:
    python3 protein_preparation.py receptor.pdb
"""

import subprocess
import sys
import os

TARGET_PH = 7.0  

# Expected NET formal charge for each residue type -- the ground truth
# every titratable residue is verified against after preparation
EXPECTED_CHARGE = {
    "ARG": +1.0,   # guanidinium, always protonated at physiological pH (pKa ~12.5)
    "LYS": +1.0,   # primary amine, protonated at physiological pH (pKa ~10.5)
    "HIS": 0.0,    # neutral by default (HID/HIE) -- but CAN be +1 (HIP) if
                   # PROPKA detects a locally shifted pKa; flagged, not failed
    "ASP": -1.0,   # carboxylate, deprotonated at physiological pH (pKa ~3.9)
    "GLU": -1.0,   # carboxylate, deprotonated at physiological pH (pKa ~4.1)
    "CYS": 0.0,    # neutral thiol by default (unless disulfide-bonded or
                   # explicitly deprotonated -- flagged, not failed)
    "TYR": 0.0,    # neutral phenol (pKa ~10.5, essentially always neutral)
}

# Tolerance for "this looks fine" vs "this needs attention"
CHARGE_TOLERANCE = 0.15   # typical charge-assignment noise is <0.05; >0.15
                          # indicates a real charge-localization problem


def run(cmd, label, allow_fail=False):
    print(f"\n--- {label} ---")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout[-2500:])
    if result.returncode != 0:
        print("  STDERR:", result.stderr[-2000:])
        if not allow_fail:
            sys.exit(f"FAILED: {label}")
        return False
    return True


def step1_clean(input_pdb, cleaned_pdb):
    print("=== STEP 1: Clean PDB (remove HETATM/water, fix altlocs) ===")
    kept = 0
    with open(input_pdb) as fin, open(cleaned_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "TER", "END")):
                if line.startswith("ATOM"):
                    altloc = line[16] if len(line) > 16 else " "
                    if altloc not in (" ", "A"):
                        continue
                fout.write(line)
                kept += 1
    print(f"  Kept {kept} lines -> {cleaned_pdb}")


def step2_fix(cleaned_pdb, fixed_pdb):
    print("\n=== STEP 2: Fix missing heavy atoms (PDBFixer) ===")
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        print("  PDBFixer not available -- copying file through unchanged.")
        print("  Install: conda install -c conda-forge pdbfixer openmm")
        import shutil
        shutil.copy(cleaned_pdb, fixed_pdb)
        return

    fixer = PDBFixer(filename=cleaned_pdb)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    # Do not fill missing residues at chain termini -- only internal gaps
    for key in list(fixer.missingResidues.keys()):
        chain_idx, res_idx = key
        chain_len = len(list(chains[chain_idx].residues()))
        if res_idx == 0 or res_idx == chain_len:
            del fixer.missingResidues[key]

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    with open(fixed_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"  Written: {fixed_pdb}")
    if fixer.missingResidues:
        print(f"  NOTE: filled {len(fixer.missingResidues)} internal gap(s).")
        print(f"  >>> Manually inspect these regions before trusting docking near them.")


def _detect_pdb2pqr_titration_flag():
    """
    pdb2pqr's CLI flag for enabling PROPKA has changed names across
    versions (--ph-calc-method=propka in some, --titration-state-method=propka
    in others). Detect which one this installation actually accepts by
    parsing `pdb2pqr30 --help` rather than hard-coding a guess.
    """
    result = subprocess.run(["pdb2pqr30", "--help"], capture_output=True, text=True)
    help_text = result.stdout + result.stderr
    if "--titration-state-method" in help_text:
        return "--titration-state-method=propka"
    if "--ph-calc-method" in help_text:
        return "--ph-calc-method=propka"
    print("  WARNING: could not detect PROPKA flag name from --help output.")
    print("  Falling back to --titration-state-method=propka (most common in")
    print("  current pdb2pqr releases). If this fails, run 'pdb2pqr30 --help'")
    print("  yourself and check the exact flag name.")
    return "--titration-state-method=propka"


def step3_protonate(fixed_pdb, protonated_pdb, propka_log):
    print(f"\n=== STEP 3: Protonate at pH {TARGET_PH} (PDB2PQR + PROPKA) ===")
    pqr_out = protonated_pdb.replace(".pdb", ".pqr")
    titration_flag = _detect_pdb2pqr_titration_flag()
    print(f"  Detected PROPKA flag: {titration_flag}")
    cmd = [
        "pdb2pqr30",
        "--ff=AMBER",
        f"--with-ph={TARGET_PH}",
        titration_flag,
        "--drop-water",
        "--pdb-output", protonated_pdb,
        fixed_pdb,
        pqr_out,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-1500:] if result.stdout.strip() else "")

    propka_crashed = (
        result.returncode != 0
        and ("__annotations__" in result.stderr or "propka" in result.stderr.lower())
    )

    if result.returncode != 0 and propka_crashed:
        print("  PROPKA crashed (known Python 3.12+/3.14 incompatibility bug in")
        print("  the propka package -- this is an upstream bug, not an input issue).")
        print("  Falling back to PLAIN pH-based protonation (no PROPKA).")
        print("  This uses standard pKa values per residue TYPE without PROPKA's")
        print("  structure-aware local-environment correction. For Arg/Lys/Asp/Glu,")
        print("  whose standard pKa values (12.5 / 10.5 / 3.9 / 4.1) are all far")
        print(f"  from pH {TARGET_PH}, this fallback should still give the correct")
        print("  ionization state for the vast majority of residues. Only unusual")
        print("  buried/salt-bridged residues with a genuinely shifted pKa would")
        print("  be treated differently by PROPKA -- worth a manual check later")
        print("  if Step 5 still flags anything unexpected.")
        cmd_fallback = [
            "pdb2pqr30",
            "--ff=AMBER",
            f"--with-ph={TARGET_PH}",
            "--drop-water",
            "--pdb-output", protonated_pdb,
            fixed_pdb,
            pqr_out,
        ]
        ok = run(cmd_fallback, "pdb2pqr30 (no PROPKA, plain pH rule)", allow_fail=True)
        if not ok:
            sys.exit(
                "pdb2pqr30 failed even without PROPKA.\n"
                "Run it manually to see the real error:\n"
                f"  pdb2pqr30 --ff=AMBER --with-ph={TARGET_PH} --drop-water "
                f"--pdb-output {protonated_pdb} {fixed_pdb} {pqr_out}"
            )
    elif result.returncode != 0:
        print("  STDERR:", result.stderr[-2000:])
        sys.exit(
            "pdb2pqr30 failed for a reason other than the known PROPKA bug.\n"
            "Run it manually to see the real error."
        )
    print(f"  Written: {protonated_pdb}")

    # Save PROPKA's own pKa log separately for manual inspection -- this is
    # the single most useful diagnostic file when something looks wrong.
    # PDB2PQR/PROPKA's naming convention for this file varies by version and
    # by whether it's based on the input or output filename, so check a
    # few likely candidates rather than assuming just one.
    candidate_bases = [
        fixed_pdb.replace(".pdb", ""),
        protonated_pdb.replace(".pdb", ""),
        pqr_out.replace(".pqr", ""),
    ]
    summary_path = None
    for base in candidate_bases:
        candidate = base + ".propka"
        if os.path.exists(candidate):
            summary_path = candidate
            break

    if summary_path:
        import shutil
        shutil.copy(summary_path, propka_log)
        print(f"  PROPKA pKa summary saved: {propka_log}")
        print(f"  >>> Open this file and look for any residue with a pKa")
        print(f"  >>> shifted by more than ~2 units from its standard value")
        print(f"  >>> (standard: Asp 3.9, Glu 4.1, His 6.5, Lys 10.5, Arg 12.5)")
    else:
        print("  (No .propka summary file found under the expected names --")
        print("   check pdb2pqr stdout above for an embedded pKa table instead,")
        print(f"   or look for a file matching one of: "
              f"{', '.join(b + '.propka' for b in candidate_bases)})")


def step4_to_pdbqt(protonated_pdb, output_pdbqt):
    """
    Do NOT use OpenBabel's Gasteiger charges for the whole protein.
    OpenBabel re-derives bond connectivity/charges from raw 3D geometry,
    and on multi-thousand-atom proteins this can systematically fail --
    charged sidechains can collapse toward ~0 net charge, because
    OpenBabel has no built-in concept of "this is residue X with a known
    formal charge", unlike tools built specifically for proteins.

    Use MGLTools' prepare_receptor4.py instead. It assigns charges from
    a residue TEMPLATE library (it knows what Arg/Lys/Asp/Glu are and
    what charge they should carry).
    """
    print(f"\n=== STEP 4: Convert to PDBQT (AutoDockTools, NOT OpenBabel) ===")

    # Locate prepare_receptor4.py -- it ships with MGLTools / ADFR suite
    candidates = [
        "prepare_receptor4.py",
        os.path.expanduser("~/MGLTools-1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
        "/usr/local/MGLTools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
    ]
    script_path = None
    for c in candidates:
        which = subprocess.run(["which", c], capture_output=True, text=True)
        if which.returncode == 0 and which.stdout.strip():
            script_path = which.stdout.strip()
            break
        if os.path.exists(c):
            script_path = c
            break

    if script_path is None:
        print("  prepare_receptor4.py not found on PATH.")
        print("  Install MGLTools (includes AutoDockTools) and either:")
        print("    a) add it to PATH, or")
        print("    b) edit `candidates` list in this script with the full path")
        print()
        print("  MGLTools install (conda, easiest route):")
        print("    conda install -c bioconda mgltools")
        print()
        print("  Falling back to OpenBabel as last resort (KNOWN UNRELIABLE")
        print("  for whole proteins). Installing MGLTools instead is strongly")
        print("  recommended over relying on this fallback.")
        cmd = [
            "obabel", protonated_pdb,
            "-opdbqt", "-O", output_pdbqt,
            "-xr", "-xc",
            "--partialcharge", "gasteiger",
        ]
        run(cmd, "obabel PDB -> PDBQT (fallback, unreliable)")
        return

    print(f"  Found: {script_path}")
    # prepare_receptor4.py reads charges from its built-in AD4 parameter
    # set (residue-template based) -- this is the correct, protein-aware
    # charge assignment method.
    cmd = [
        "pythonsh", script_path,
        "-r", protonated_pdb,
        "-o", output_pdbqt,
        "-A", "checkhydrogens",   # add/check hydrogens, don't strip existing protonation
        "-U", "nphs_lps",         # merge nonpolar H, remove lone pairs (standard receptor prep)
    ]
    ok = run(cmd, "prepare_receptor4.py", allow_fail=True)
    if not ok:
        # pythonsh is MGLTools' bundled interpreter; some installs expose
        # the script as directly executable with system python instead
        print("  Retrying with system python3 instead of pythonsh...")
        cmd2 = ["python3", script_path, "-r", protonated_pdb,
                "-o", output_pdbqt, "-A", "checkhydrogens", "-U", "nphs_lps"]
        ok = run(cmd2, "prepare_receptor4.py (python3)", allow_fail=True)
        if not ok:
            sys.exit(
                "prepare_receptor4.py failed under both pythonsh and python3.\n"
                "Run it manually to see the real error:\n"
                f"  pythonsh {script_path} -r {protonated_pdb} -o {output_pdbqt}"
            )
    print(f"  Written: {output_pdbqt}")


def parse_pdbqt_residues(pdbqt_path):
    """Return {(resname, resnum): [(atomname, charge), ...]}"""
    residues = {}
    with open(pdbqt_path, errors="replace") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                resnum = line[22:26].strip()
                atomname = line[12:16].strip()
                try:
                    charge = float(line[70:76].strip())
                except ValueError:
                    continue
                key = (resname, resnum)
                residues.setdefault(key, []).append((atomname, charge))
    return residues


def step5_verify_all_charges(output_pdbqt, report_path):
    """
    Sum partial charges per residue and compare against the expected
    formal charge for that residue type. Any residue outside tolerance
    is flagged, with the exact atom-by-atom breakdown printed so the
    source of the discrepancy can be identified.
    """
    print(f"\n=== STEP 5: Verify per-residue formal charges (CRITICAL) ===")
    residues = parse_pdbqt_residues(output_pdbqt)

    flagged = []
    checked = 0
    lines_out = []
    lines_out.append(f"{'Residue':10} {'Sum charge':12} {'Expected':10} {'Deviation':10} {'Status'}")
    lines_out.append("=" * 65)

    def sort_key(item):
        (resname, resnum), _ = item
        try:
            n = int(resnum)
        except ValueError:
            n = 0
        return (resname, n)

    for (resname, resnum), atoms in sorted(residues.items(), key=sort_key):
        if resname not in EXPECTED_CHARGE:
            continue
        checked += 1
        total = sum(q for _, q in atoms)
        expected = EXPECTED_CHARGE[resname]
        deviation = abs(total - expected)

        # HIS, CYS, TYR can legitimately deviate (HIP/CYM/charged-Tyr cases)
        # -- these get a NOTE instead of a hard FLAG
        soft_check_residues = {"HIS", "CYS", "TYR"}

        if deviation > CHARGE_TOLERANCE:
            if resname in soft_check_residues:
                status = "NOTE (check manually)"
            else:
                status = "*** FLAGGED ***"
                flagged.append((resname, resnum, total, expected, atoms))
        else:
            status = "OK"

        line = f"{resname}{resnum:6} {total:+12.3f} {expected:+10.1f} {deviation:10.3f} {status}"
        lines_out.append(line)
        print(f"  {line}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines_out))
    print(f"\n  Full report written: {report_path}")
    print(f"  Checked {checked} titratable residues.")

    if flagged:
        print(f"\n  *** {len(flagged)} RESIDUE(S) FAILED CHARGE VERIFICATION ***")
        for resname, resnum, total, expected, atoms in flagged:
            print(f"\n  --- {resname}{resnum}: sum={total:+.3f}, expected={expected:+.1f} ---")
            print(f"      Atom-by-atom breakdown:")
            for atomname, q in atoms:
                print(f"        {atomname:6}: {q:+.3f}")
        print(f"\n  >>> THESE RESIDUES NEED MANUAL ATTENTION before docking. <<<")
        print(f"  Likely causes, in order of probability:")
        print(f"    1. PROPKA detected an unusual local pKa shift (check the")
        print(f"       .propka log from Step 3 for this residue number)")
        print(f"    2. The residue sits in a tight salt-bridge network and")
        print(f"       charge-equilibration smeared charge away from the")
        print(f"       formal site")
        print(f"    3. A genuine structural/connectivity issue (missing atom,")
        print(f"       wrong bond order) introduced during PDBFixer or PDB2PQR")
        print(f"\n  Fix options:")
        print(f"    A) Re-run charge assignment with MMFF94 instead of Gasteiger")
        print(f"       for just this residue's local environment, OR")
        print(f"    B) Manually inspect/rebuild the sidechain geometry in")
        print(f"       PyMOL/Chimera and re-run from Step 4, OR")
        print(f"    C) If using this for DOCKING ONLY (not MD), you can")
        print(f"       manually patch the PDBQT: redistribute the missing")
        print(f"       charge onto the terminal guanidinium/carboxylate atom")
        print(f"       (see patch_residue_charge() function below)")
    else:
        print(f"\n  ALL {checked} TITRATABLE RESIDUES PASSED VERIFICATION.")
        print(f"  Receptor is ready for Vina docking.")

    return flagged


def patch_residue_charge(pdbqt_path, resname, resnum, target_atom_names,
                          expected_total, output_path=None):
    """
    Manual rescue patch for a flagged residue. Redistributes the charge
    deficit evenly across the named "anchor" atoms (e.g. CZ, NH1, NH2
    for Arg) so the residue sums to the correct formal charge, without
    touching any other residue's charges.

    Use only as a last resort if re-running PROPKA/Gasteiger does not
    fix the residue and a working PDBQT is needed immediately.

    Example, for a flagged ARG at residue number 280:
        patch_residue_charge(
            "protein_prepared.pdbqt", "ARG", "280",
            target_atom_names=["CZ", "NH1", "NH2"],
            expected_total=1.0,
            output_path="protein_prepared_patched.pdbqt"
        )
    """
    if output_path is None:
        output_path = pdbqt_path

    lines = open(pdbqt_path, errors="replace").readlines()
    residue_lines_idx = []
    current_total = 0.0
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            rn = line[17:20].strip()
            num = line[22:26].strip()
            if rn == resname and num == str(resnum):
                current_total += float(line[70:76].strip())
                atomname = line[12:16].strip()
                if atomname in target_atom_names:
                    residue_lines_idx.append(i)

    if not residue_lines_idx:
        print(f"  No matching atoms found for {resname}{resnum} -- no patch applied.")
        return

    deficit = expected_total - current_total
    per_atom = deficit / len(residue_lines_idx)
    print(f"  Patching {resname}{resnum}: current={current_total:+.3f}, "
          f"target={expected_total:+.1f}, deficit={deficit:+.3f}, "
          f"distributing {per_atom:+.3f} across {len(residue_lines_idx)} atom(s)")

    for i in residue_lines_idx:
        line = lines[i]
        old_q = float(line[70:76].strip())
        new_q = old_q + per_atom
        new_line = line[:70] + f"{new_q:6.3f}" + line[76:]
        lines[i] = new_line

    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"  Patched file written: {output_path}")
    print(f"  Re-run step5_verify_all_charges() to confirm the fix.")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 protein_preparation.py <input.pdb> [--stop-after-step3] [--start-from-step4]")
    input_pdb = sys.argv[1]
    stop_after_3 = "--stop-after-step3" in sys.argv
    start_from_4 = "--start-from-step4" in sys.argv
    base = os.path.splitext(input_pdb)[0]

    cleaned_pdb     = f"{base}_cleaned.pdb"
    fixed_pdb       = f"{base}_fixed.pdb"
    protonated_pdb  = f"{base}_protonated.pdb"
    propka_log      = f"{base}_propka_summary.txt"
    output_pdbqt    = f"{base}_prepared.pdbqt"
    charge_report   = f"{base}_charge_verification.txt"

    if start_from_4:
        if not os.path.exists(protonated_pdb):
            sys.exit(f"--start-from-step4 requires {protonated_pdb} to already "
                      f"exist (run Steps 1-3 first, in the pdb2pqr environment).")
        print(f"Resuming from Step 4 using existing: {protonated_pdb}")
        step4_to_pdbqt(protonated_pdb, output_pdbqt)
        flagged = step5_verify_all_charges(output_pdbqt, charge_report)
    else:
        step1_clean(input_pdb, cleaned_pdb)
        step2_fix(cleaned_pdb, fixed_pdb)
        step3_protonate(fixed_pdb, protonated_pdb, propka_log)

        if stop_after_3:
            print(f"\n{'='*65}")
            print(f"STOPPED AFTER STEP 3 (as requested).")
            print(f"Protonated structure written: {protonated_pdb}")
            print(f"Now switch environments and run:")
            print(f"  conda activate prot_prep")
            print(f"  python3 protein_preparation.py {input_pdb} --start-from-step4")
            print(f"{'='*65}")
            return

        step4_to_pdbqt(protonated_pdb, output_pdbqt)
        flagged = step5_verify_all_charges(output_pdbqt, charge_report)

    print(f"\n{'='*65}")
    if flagged:
        print(f"DONE WITH WARNINGS: {output_pdbqt}")
        print(f"Review {charge_report} and fix flagged residues before docking.")
    else:
        print(f"DONE: {output_pdbqt} is verified and ready for Vina.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
