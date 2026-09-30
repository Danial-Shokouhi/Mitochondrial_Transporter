# Docking pipeline: FAD, Palmitoyl-carnitine, and L-carnitine vs. BOU

End-to-end procedure from ligand preparation through pose selection and
interaction analysis. Setup instructions for the two external tools
used (AutoDock Vina, PLIP) are kept in their own files in this folder;
this document covers the full workflow and how they fit together.

## 1. Ligand preparation

Ligand `.pdbqt` files (`FAD.pdbqt`, `PCAR.pdbqt`, `CAR.pdbqt`) were
generated from SMILES using RDKit (3D embedding + MMFF94 optimization)
and converted to PDBQT with Meeko, rather than derived from a
downloaded structure file -- this ensures correct, consistent charge
assignment for each ligand from a single well-defined SMILES string.

SMILES were obtained from the Human Metabolome Database (HMDB):

| Ligand | HMDB entry | Accession |
|--------|-----------|-----------|
| L-Palmitoylcarnitine (PCAR) | L-Palmitoylcarnitine | HMDB0240774 |
| L-Carnitine (CAR) | L-Carnitine | HMDB0000062 |
| FAD | FAD | HMDB0001248 |

Install RDKit and Meeko first:
```
conda install -c conda-forge rdkit
pip install meeko
```

For each ligand: generate a 3D structure from SMILES with RDKit, then
convert to PDBQT with Meeko's `mk_prepare_ligand.py`.

**PCAR:**
```python
from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "[H][C@](CC([O-])=O)(C[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=2500)
writer = Chem.SDWriter("PCAR.sdf")
writer.write(mol)
writer.close()
```
```
mk_prepare_ligand.py -i PCAR.sdf -o PCAR.pdbqt
```

**CAR:**
```python
from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "C[N+](C)(C)C[C@H](O)CC([O-])=O"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=2500)
writer = Chem.SDWriter("CAR.sdf")
writer.write(mol)
writer.close()
```
```
mk_prepare_ligand.py -i CAR.sdf -o CAR.pdbqt
```

**FAD:**
```python
from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP([O-])(=O)OP([O-])(=O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1O)N1C=NC3=C1N=CN=C3N)C1=NC(=O)NC(=O)C1=N2"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=2500)
writer = Chem.SDWriter("FAD.sdf")
writer.write(mol)
writer.close()
```
```
mk_prepare_ligand.py -i FAD.sdf -o FAD.pdbqt
```

## 2. Receptor preparation

Receptor `.pdbqt` files (`cBOU.pdbqt`, `mBOU.pdbqt`) were prepared and
independently charge-verified using the two scripts in this folder:

- **`protein_preparation.py`** -- cleans the input PDB, fixes missing
  heavy atoms (PDBFixer), protonates at physiological pH (PDB2PQR +
  PROPKA), and converts to PDBQT (AutoDockTools' `prepare_receptor4.py`,
  which assigns charges from a residue template library rather than
  re-deriving them from geometry).
  ```
  python3 protein_preparation.py cBOU.pdb
  python3 protein_preparation.py mBOU.pdb
  ```
- **`verify_pdbqt_charges.py`** -- sums the partial charges of every
  charged/titratable residue in the resulting PDBQT and flags any
  residue whose total deviates from its expected formal charge (e.g.
  +1.0 for Arg/Lys, -1.0 for Asp/Glu). This is run automatically as
  the last step of `protein_preparation.py`, and can also be run
  standalone on any existing PDBQT:
  ```
  python3 verify_pdbqt_charges.py cBOU_prepared.pdbqt
  ```

## 3. AutoDock Vina installation

See `AutoDockVina_setup.md` in this folder (AutoDock Vina v1.2.7).

## 4. Docking

```
vina --config config.txt --out FAD_out.pdbqt
vina --config config.txt --out PCAR_out.pdbqt
vina --config config.txt --out CAR_out.pdbqt
```
The `config.txt` used for each ligand (receptor, search-space
coordinates, exhaustiveness, etc.) is deposited with the corresponding
ligand's data on Zenodo repository.

Each multi-pose output is split into individual pose files:
```
vina_split --input FAD_out.pdbqt
vina_split --input PCAR_out.pdbqt
vina_split --input CAR_out.pdbqt
```
Individual pose `.pdbqt` files are then converted to `.pdb` (PyMOL or
OpenBabel, installed separately) for input to PLIP.

## 5. PLIP installation

See `PLIP_setup.md` in this folder (PLIP v3.0.0).

## 6. Interaction analysis

PLIP is run on each individual pose to identify and characterize the
protein-ligand interactions, then used to select the representative
docked pose:
```
plip -f poseXX.pdb -yvt
```
