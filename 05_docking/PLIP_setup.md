# PLIP v3.0.0 setup (Ubuntu)

Official sources used for this guide:
- GitHub: https://github.com/pharmai/plip
- PyPI: https://pypi.org/project/plip/3.0.0/

PLIP moved maintainers in 2020 (originally ssalentin/plip, now
pharmai/plip) -- use the pharmai/plip links above, not older ones you
may find elsewhere.

## Dependencies (needed for Options A-C below)

- Python >= 3.6.9
- OpenBabel >= 3.0.0, with Python bindings
- PyMOL >= 2.3.0 with Python bindings (optional, visualization only)
- ImageMagick >= 7.0 (optional)

OpenBabel is the dependency most people get stuck on. Two ways to get
it on Ubuntu:

```
# apt (example for Ubuntu 20.04)
apt-get update && apt-get install -y \
    libopenbabel-dev \
    libopenbabel6 \
    python3-openbabel \
    openbabel
```
or
```
# conda
conda install openbabel -c conda-forge
pip install openbabel
```
If PLIP later errors with `ValueError: [...] is not a recognised Open
Babel descriptor type`, the OpenBabel version and its Python bindings
don't match -- reinstall both together via the same method (both apt,
or both conda), not mixed.

## Option A -- pip (simplest, once OpenBabel is installed)

```
pip install plip==3.0.0
```

## Option B -- conda

```
conda install -c conda-forge plip=3.0.0
```

## Option C -- from source

```
git clone https://github.com/pharmai/plip.git
cd plip
pip install -e .
```
(editable mode, `-e`, is required for the C++ modules to compile
correctly)

## Option D -- containerized (Docker/Singularity, recommended by PLIP's
## own docs as the easiest route -- sidesteps the OpenBabel dependency
## entirely)

Docker:
```
docker run --rm \
    -v ${PWD}:/results \
    -w /results \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    pharmai/plip:latest -i <PDB_ID> -yv
```
Check Docker Hub (hub.docker.com/r/pharmai/plip) for a version-pinned
tag if you need exactly 3.0.0 rather than `latest`, for reproducibility.

Singularity (pre-built image under GitHub Releases):
```
./plip.simg -i <PDB_ID> -yv
```

## Verifying the install

```
plip --version
```
or, if running from a cloned source checkout without a system-wide
install:
```
python plip/plipcmd.py --version
```

## Running

```
plip -i <PDB_ID or path/to/complex.pdb> -yv
```
`-y` generates the PyMOL visualization session, `-v` runs in verbose
mode. See the docking scripts in this folder for how PLIP is invoked
on the selected docked poses in this project.
