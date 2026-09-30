# CHARMM-GUI Bilayer Builder pipeline

CHARMM-GUI (https://www.charmm-gui.org/) is a web-based tool, not a
script -- this documents the exact sequence of choices made in its
Membrane Builder / Bilayer Builder wizard to build each of the 5
systems (Apo-c, Apo-m, FAD, PCAR, CAR). Generated output files are
deposited under `06_CHARMM_MD_simulations/Files/CHARMM-GUI/` (see that
folder's README).

## 1. Navigate to the builder

CHARMM-GUI -> **Input Generator** -> **Membrane Builder** -> **Bilayer Builder**

## 2. Upload the input structure

Input `.pdb`: either a docking-output structure (ligand-bound: FAD,
PCAR, or CAR) or an Apo-state structure (Apo-c or Apo-m). Click **Next
Step**.

## 3. PDB manipulation options

- **Protein SEGID**: `PROA`, PDB ID chain `A`
- For docked (ligand-bound) structures only: also check **Hetero
  SEGID** `HETA`, PDB ID chain `B`, `UNL`
- Check **Check pKa**

Click **Next Step**.

## 4. Protein pKa / protonation

- **pH**: 7 (representative of the mitochondrial inner membrane space)
- Check **Use CHARMM General Force Field** (generates CHARMM top/par
  files from the PDB coordinates)
- Check **Protonated/Deprotonated based on selected pH**

Click **Next Step**.

## 5. Orientation

Run **PPM 2.0** for `PROA` to determine membrane orientation via the
PPM server. Click **Next Step**.

## 6. Lipid bilayer setup

- Check `step2_orient.pdb` for consistency before proceeding
- **System type**: Heterogeneous Lipid
- **Box type**: Rectangular
- **Water thickness**: 14 A
- **XY Dimension Ratio**: 1 (with "Number of lipids" selected as the
  sizing method)
- **Lipid composition**: as defined in the manuscript's Materials and
  Methods (composition table)

Click **Next Step**.

## 7. System size / ions

- Check `step3_packing.pdb` for consistency; confirm the determined
  system size is correct
- **System Building Options**: Replacement method; check **Check
  lipid ring (and protein surface) penetration**
- **Component Building Options**: check **Include Ions**
  - Ion Placing Method: Distance
  - Basic Ion Types: KCl
  - Concentration: 0.15 M
  - Check **Neutralizing**
- Confirm the calculated solvent composition is correct

Click **Next Step**.

## 8. Penetration check

Confirm CHARMM-GUI reports both:
- "No protein surface penetration is found"
- "No lipid ring penetration is found"

Click **Next Step** to generate the membrane lipid components.

## 9. Lipid generation check

Check `step4_lipid.pdb` to confirm complete, correct lipid generation.

## 10. Final assembly and force field / MD input options

Check `step5_assembly.pdb` to confirm the correct components were
generated, then set:

- **Force field**: CHARMM36m
- **Hydrogen mass repartitioning**: checked, for all 5 systems
  (Apo-c, Apo-m, FAD, PCAR, CAR)
- **WYF parameters (cation-pi interactions)**: checked **only** for the
  PCAR and CAR docked systems (their quaternary ammonium group is the
  relevant cation partner for this correction) -- left unchecked for
  FAD and both Apo systems
- **Input Generation Options**: check **More CHARMM minimization
  during input generation**; check **GROMACS**
- **Equilibration Options**:
  - Check **Generate grid information for PME FFT automatically**
  - Ensemble: NVT
  - Surface tension: 0
  - Temperature: 303.15 K

## 11. Post-download adjustments

After downloading the generated bilayer system, manually set the
required GROMACS `.mdp` parameters (e.g. `nsteps`) for each
minimization, equilibration, and production MD stage to match the
values defined in the manuscript's Materials and Methods -- these are
not CHARMM-GUI defaults and must be edited in per stage before running.
