#!/usr/bin/env python3
"""
Standalone charge verifier for an existing receptor PDBQT.

Scans every charged/titratable residue in a PDBQT file, sums each
residue's atomic partial charges, and compares the sum against its
expected formal charge. This can be run independently on any PDBQT --
it does not require running protein_preparation.py first.

Usage:
    python3 verify_pdbqt_charges.py protein.pdbqt
"""

import sys

EXPECTED_CHARGE = {
    "ARG": +1.0, "LYS": +1.0, "ASP": -1.0, "GLU": -1.0,
    "HIS": 0.0, "CYS": 0.0, "TYR": 0.0,
}
SOFT_CHECK = {"HIS", "CYS", "TYR"}   # legitimately variable, NOTE not FLAG
TOLERANCE = 0.15


def main(pdbqt_path):
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
                residues.setdefault((resname, resnum), []).append((atomname, charge))

    def sort_key(item):
        (resname, resnum), _ = item
        try:
            n = int(resnum)
        except ValueError:
            n = 0
        return (resname, n)

    print(f"{'Residue':10} {'Sum':>9} {'Expect':>8} {'Dev':>7}  Status")
    print("=" * 55)

    flagged, noted, total_checked = [], [], 0

    for (resname, resnum), atoms in sorted(residues.items(), key=sort_key):
        if resname not in EXPECTED_CHARGE:
            continue
        total_checked += 1
        total = sum(q for _, q in atoms)
        expected = EXPECTED_CHARGE[resname]
        dev = abs(total - expected)

        if dev > TOLERANCE:
            if resname in SOFT_CHECK:
                status = "NOTE"
                noted.append((resname, resnum, total, atoms))
            else:
                status = "*** FLAG ***"
                flagged.append((resname, resnum, total, expected, atoms))
        else:
            status = "ok"

        if status != "ok":
            print(f"{resname}{resnum:<6} {total:9.3f} {expected:8.1f} {dev:7.3f}  {status}")

    print(f"\nChecked {total_checked} titratable residues.")
    print(f"Flagged (hard errors): {len(flagged)}")
    print(f"Noted (His/Cys/Tyr variability, check manually): {len(noted)}")

    if flagged:
        print("\n" + "=" * 55)
        print("DETAILED BREAKDOWN OF FLAGGED RESIDUES")
        print("=" * 55)
        for resname, resnum, total, expected, atoms in flagged:
            print(f"\n{resname}{resnum}  (sum={total:+.3f}, expected={expected:+.1f}, "
                  f"deficit={expected-total:+.3f})")
            for atomname, q in atoms:
                print(f"    {atomname:6}: {q:+.3f}")
        print(f"\n>>> Fix these before docking. See protein_preparation.py")
        print(f">>> for the patch_residue_charge() helper function.")
    else:
        print("\nNo hard charge errors found. Receptor looks consistent.")

    if noted:
        print(f"\nHis/Cys/Tyr residues with non-default charge (verify these")
        print(f"are intentional -- e.g. a protonated His in an active site,")
        print(f"or a disulfide-bonded Cys):")
        for resname, resnum, total, atoms in noted:
            print(f"  {resname}{resnum}: sum={total:+.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 verify_pdbqt_charges.py <protein.pdbqt>")
    main(sys.argv[1])
