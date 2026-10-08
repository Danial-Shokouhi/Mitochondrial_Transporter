# Analysis code

Analysis, statistics and figure-generation code used in study: 
"The mitochondrial carrier BOU is required for mitochondrial carnitine-dependent acyl and flavin cofactor transport in Arabidopsis"

| | |
|---|---|
| **Publication** | *to be added on acceptance* |
| **Data (Zenodo)** | *https://doi.org/10.5281/zenodo.22040455/* |
| **This code (archived)** | *10.5281/zenodo.23061840* |
| **Licence** | MIT (see [`LICENSE`](LICENSE)) |

All raw and processed data, simulation inputs and production trajectories are
deposited separately on Zenodo. This repository contains **code only**. The two
are organised in the same numbered sections.

---

## Repository layout

```
01_metabolomics/            metabolite statistics and figures
02_enzymes_OCR/             marker enzymes, enzyme activities, respirometry
03_Mitochondrial_uptake_assays/  acyl-CoA / acylcarnitine and FAD uptake
04_structural_modeling_AlphaFold2/  model screening by TM-align
05_docking/                 receptor preparation, docking, interaction profiling
06_CHARMM_MD_simulations/   simulation setup and all trajectory analysis
    simulation_setup_and_running/
    trajectory_processing/
    interaction_analysis/
    gate_networks_analysis/
    advanced_analysis/
```

---

## What each script does

The **Figure** column is for the manuscript figure each script feeds; fill it in
before release.

### 01 — Metabolomics

| Script | Produces | Figure |
|---|---|---|
| `Metabolites_Mitochondrial.R` | Mitochondrial metabolite statistics (t-test / Wilcoxon, Benjamini–Hochberg) and plots | |
| `Metabolites_WC.R` | Whole-tissue metabolite statistics and plots | |

### 02 — Marker enzymes, enzyme activities and respirometry

| Script | Produces | Figure |
|---|---|---|
| `Markers_Statistics.R` | Two-way ANOVA (Fraction × Genotype) on subcellular markers and chlorophyll | |
| `Markers_figure.py` | Marker specific activities, crude vs purified, and enrichment ratios | |
| `Enzyme_Statistics.R` | GDC, PDH, OGDH, BCKDH and SDH activity comparisons with Cohen's *d* | |
| `OCR_statistics.R` | Oxygen consumption rates, respiratory control and ADP/O ratios | |

### 03 — Mitochondrial uptake assays

| Script | Produces | Figure |
|---|---|---|
| `Acyl_Statistics.py` | Acyl uptake statistics: LOQ filtering, Welch tests, Bonferroni over 13 planned comparisons | |
| `carnitine_coa_uptake_figure.py` | Acylcarnitine and acyl-CoA uptake bar figures (quantifier and qualifier transitions) | |
| `chromatogram_carnitine.py` | Octanoyl- and palmitoylcarnitine MRM chromatograms | |
| `chromatogram_coa.py` | ¹³C-octanoyl-CoA and ¹³C-palmitoyl-CoA standard chromatograms | |
| `FAD_Statistics.R` | FAD net uptake: two-way ANOVA (Genotype × Time), uptake rates, endogenous pools | |
| `flavin_uptake_figure.py` | FAD uptake kinetics figures | |

### 04 — Structural modelling

| Script | Produces | Figure |
|---|---|---|
| `TM_align.sh` | TM-align screening of all ColabFold models against 1OKC (c-state) and 6GCI (m-state) | |
| `TM_align_setup.md` | Installation notes | — |

### 05 — Docking

| Script | Produces | Figure |
|---|---|---|
| `protein_preparation.py` | Receptor pipeline: clean → PDBFixer → PDB2PQR/PROPKA (pH 7.0) → PDBQT, with per-residue formal-charge verification | — |
| `verify_pdbqt_charges.py` | Standalone charge verifier for any receptor PDBQT | — |
| `Docking_Pipeline.md` | Full ligand preparation, docking and PLIP pose-selection workflow | — |
| `AutoDockVina_setup.md`, `PLIP_setup.md` | Installation notes | — |

### 06 — Molecular dynamics

**Setup and running**

| File | Purpose |
|---|---|
| `CHARMM-GUI_Bilayer_pipeline.md` | Membrane Builder settings for all five systems |
| `GROMACS_CUDA_setup.md` | GROMACS 2026.2 build (CUDA, Colvars, PLUMED) |
| `run_equilibration.sh` | Minimisation → six equilibration stages → 1 µs production |

**Trajectory processing**

| Script | Produces | Figure |
|---|---|---|
| `trajectory_processing.txt` | Quality checks, PBC removal and backbone fitting; ligand contact occupancy | — |
| `Protein_structure_analysis.txt` | Helix classification (DSSP + Cα tilt + phosphate-density midplane); defines H1–H6, h12/h34/h56 and loops | — |
| `rmsd_gyrate_FAD_PCAR_ApoC.py` | Backbone RMSD and radius of gyration, c-state systems | |
| `rmsd_gyrate_CAR_ApoM.py` | Backbone RMSD and radius of gyration, m-state systems | |
| `rmsf_regional_flexibility.py` | Per-residue RMSF decomposed by structural region | |

**Interaction analysis** — ligand–protein contacts, all on heavy atoms

| Script | Interaction class and criteria | Figure |
|---|---|---|
| `polar_contacts_FAD_PCAR.py`, `polar_contacts_CAR.py` | Polar contacts and salt bridges, < 0.35 nm | |
| `hydrophobic_contacts_FAD_PCAR.py`, `hydrophobic_contacts_CAR.py` | Hydrophobic contacts, < 0.45 nm | |
| `hbond_FAD_PCAR.py` | H-bonds: D–A ≤ 3.5 Å, D–H···A > 120° (ideal > 150°) | |
| `aromatic_contacts_FAD_PCAR.py` | π–π (centroid ≤ 5.5 Å, interplanar < 30°) and π–cation (≤ 6.0 Å, < 45°) | |
| `aromatic_contacts_CAR.py` | π–cation for the carnitine trimethylammonium group | |

**Gate networks** — salt-bridge threshold 0.35 nm throughout

| Script | Produces | Figure |
|---|---|---|
| `gate_xvg_files.py` | Generates all gate distance/H-bond/angle `.xvg` files (`gmx mindist`, `hbond`, `angle`) | — |
| `mgate_network.py` | Matrix-gate salt bridges in the c-state systems | |
| `mgate_network_brace.py` | Matrix-gate brace interactions | |
| `FAD_mgate_rearrangement.py` | FAD-specific matrix-gate rearrangement | |
| `cgate_network.py` | Cytoplasmic-gate salt bridges and braces in the m-state systems | |

**Advanced analysis**

| Script | Produces | Figure |
|---|---|---|
| `PCA_FEL_FAD_PCAR_Analysis.py` | PCA: Cα extraction, replicate concatenation, `gmx covar`, projection on PC1/PC2 (`gmx anaeig`). **Run before the Figures script.** | — |
| `PCA_FEL_FAD_PCAR_Figures.py` | Free energy landscapes from the PCA projections | |
| `basin_autocompare_FAD_PCAR.py` | Interaction fingerprints of frames within each FEL basin | |
| `DCCM.py` | Cross-correlation matrices, contact networks, degree and betweenness centrality, optimal paths | |
| `cardiolipin_analysis.py` | Cardiolipin contact occupancy (6.0 Å), headgroup vs acyl chain, leaflet assignment, hotspots | |
| `cavity_hole2.py` | Pore radius profiles (HOLE) | |
| `alignment_BOU_vs_SLC25A20.py` | BOU vs human SLC25A20 alignment at gate and binding-site positions | |
| `mmgbsa_analysis.sh` | MM-GBSA pipeline (gmx_MMPBSA, igb = 2, 0.150 M salt, every 100th frame) | — |
| `parse_mmgbsa_dat.py` | Parses and averages MM-GBSA output across replicates | |
| `MMGBSA_energy_summary.txt` | Averaged effective binding energies for FAD, palmitoylcarnitine and carnitine | — |

> **MM-GBSA, not MM-PBSA.** These calculations use the generalised Born model
> (`igb = 2`). The tool is named `gmx_MMPBSA` and requires an input file called
> `mmpbsa.in`; those names are the tool's, not a description of the method. No
> entropic term was computed, so the reported values are **effective binding
> energies** for comparison between ligands, not absolute binding free energies.

---

## Software

| | Version |
|---|---|
| R | 4.5.2 — readxl, dplyr, tidyr, stringr, ggplot2, ggnewscale, emmeans, car, rstatix, writexl |
| Python | 3.14.7 — pandas, NumPy, SciPy, matplotlib, MDAnalysis, NetworkX |
| ColabFold | 1.6.1 (AlphaFold2 + MMseqs2) |
| AutoDock Vina | 1.2.7 |
| PLIP | 3.0.0 |
| GROMACS | 2026.2, CUDA 13.2, built with Colvars and PLUMED |
| HOLE | 2 |
| gmx_MMPBSA | see `mmgbsa_analysis.sh` |

Ligand and receptor preparation additionally use RDKit, Meeko, PDBFixer,
PDB2PQR with PROPKA, and AutoDockTools; installation notes are in `05_docking/`.

---

## Running the code

Scripts read their inputs from the corresponding Zenodo archive. Trajectory
paths inside the MD scripts are given as `.../rep1`, `.../rep2`, `.../rep3` —
replace these with the paths to your local copy of the deposited replicates.

Two ordering constraints:

1. `gate_xvg_files.py` must run before any gate-network plotting script.
2. `PCA_FEL_FAD_PCAR_Analysis.py` must run before `PCA_FEL_FAD_PCAR_Figures.py`
   and before `basin_autocompare_FAD_PCAR.py`.

---

## Citation

If you use this code, please cite the publication above and the archived release
of this repository.
