# AutoDock Vina 1.2.7 setup (Ubuntu)

Official sources used for this guide:
- Releases: https://github.com/ccsb-scripps/AutoDock-Vina/releases
- Manual: https://vina.scripps.edu/manual/

Note: the vina.scripps.edu manual's Linux install section still refers
to the old 1.1.2 tarball release. For 1.2.x, the standalone `vina`
command-line binary is distributed via GitHub Releases instead (Option
A below); the pip/conda route (Option B) installs the separate Python
bindings, not the CLI binary.

Pick ONE option depending on what you need.

## Option A -- precompiled Linux binary (recommended for CLI docking)

This is what you need if you're running `vina` from the command line
(e.g. with a `config.txt`), which is the case for this project's
docking scripts.

```
wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64
chmod +x vina_1.2.7_linux_x86_64
mv vina_1.2.7_linux_x86_64 vina
./vina --version
```

Confirm the exact asset filename on the release page first --
https://github.com/ccsb-scripps/AutoDock-Vina/releases/tag/v1.2.7 --
in case the naming has changed; the pattern above matches the
convention used by recent 1.2.x releases.

## Option B -- Python bindings via conda (only if scripting with the
## `vina` Python package, e.g. `import vina`)

```
conda create -n vina python=3
conda activate vina
conda config --env --add channels conda-forge
conda install -c conda-forge numpy swig boost-cpp libboost
pip install vina
```

This does NOT install the standalone CLI binary -- the Python
bindings and the CLI executable are separate, per the official docs.

## Option C -- build from source (advanced, not recommended for
## regular use)

The official documentation explicitly notes: "Building Vina from
source is NOT meant to be done by regular users." Only do this if
Options A/B don't fit your system (e.g. a non-x86_64 architecture).

Requires a C++ compiler and Boost installed first, then:
```
git clone https://github.com/ccsb-scripps/AutoDock-Vina.git
cd AutoDock-Vina/build/linux/release
# edit the Makefile to set the Boost paths/version for your system
make depend
make
```

## Verifying the install

```
./vina --help
```
should print the usage summary. Docking runs are driven by a
`config.txt` (receptor, ligand, search-space box, exhaustiveness,
etc.) -- see the docking scripts in this folder for how it's invoked
in this project.
