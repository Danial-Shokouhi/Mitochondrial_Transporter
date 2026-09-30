# GROMACS (CUDA-enabled) setup for MD simulations (Ubuntu)

Environment used for all production MD simulations in this project:
GROMACS 2026.2, built from source with CUDA GPU acceleration, Colvars,
and PLUMED support -- confirmed directly from the working installation's
own `gmx mdrun -version` output.

## 1. Install the NVIDIA CUDA Toolkit

Requires an NVIDIA GPU and the NVIDIA driver already installed
(`nvidia-smi` should already work before doing this). This
installation used **CUDA 13.2** (`nvcc` 13.2.78, driver/runtime 13.20).

```
# Detect your Ubuntu version and download the matching keyring package
distro=$(. /etc/os-release && echo "ubuntu${VERSION_ID/./}")
wget https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-2
```

Add CUDA to your PATH and library path, then reload your shell:
```
echo 'export PATH=/usr/local/cuda-13.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```
nvcc --version
```
Should report release 13.2, V13.2.78. If `nvcc` is not found, the
PATH export above didn't take effect -- open a new terminal and retry.

## 2. Install build tools and dependencies

```
sudo apt update
sudo apt install -y build-essential cmake git wget
sudo apt install -y libfftw3-dev libboost-all-dev
```

This build used **gcc-13/g++-13** explicitly as the host compiler.
Install it if it isn't already the system default:
```
sudo apt install -y gcc-13 g++-13
```

## 3. Install PLUMED (required before configuring GROMACS with PLUMED support)

GROMACS 2026.2 supports linking PLUMED natively via a CMake flag
(rather than the older patch-based method used by earlier GROMACS
versions), but PLUMED itself must already be installed and
discoverable by CMake first. Follow PLUMED's own installation
instructions for your system (https://www.plumed.org/doc) before
proceeding to Step 5 -- this is not reproduced here since it depends
on which PLUMED version and install method you use.

## 4. Download and extract GROMACS source

This project used **GROMACS 2026.2**.

```
mkdir -p ~/gromacs-build
cd ~/gromacs-build
wget https://ftp.gromacs.org/gromacs/gromacs-2026.2.tar.gz
tar xfz gromacs-2026.2.tar.gz
cd gromacs-2026.2
```

## 5. Configure with CUDA, Colvars, and PLUMED support

```
mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
    -DCMAKE_C_COMPILER=gcc-13 \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DGMX_GPU=CUDA \
    -DGMX_MPI=OFF \
    -DGMX_OPENMP=ON \
    -DGMX_USE_COLVARS=internal \
    -DGMX_USE_PLUMED=ON
```

`-DGMX_USE_COLVARS=internal` enables GROMACS' built-in Colvars module.
`-DGMX_USE_PLUMED=ON` requires PLUMED to already be installed and
findable by CMake (Step 3) -- if this step fails to detect PLUMED,
check `PLUMED_ROOT` or your PLUMED install's CMake config location.

## 6. Compile and install

```
make -j$(nproc)
sudo make install
```

`-j$(nproc)` uses all available CPU cores to compile in parallel --
compiling GROMACS from source takes a while regardless, plan for
this to run for tens of minutes depending on the machine.

## 7. Activate GROMACS

```
source /usr/local/gromacs/bin/GMXRC
```

Make this permanent (auto-loaded in every new terminal):
```
echo "source /usr/local/gromacs/bin/GMXRC" >> ~/.bashrc
```

## 8. Verify the build

```
gmx mdrun -version
```

Confirm the output matches (adjust expectations if your hardware/OS
differs, e.g. SIMD instruction set is CPU-dependent and auto-detected):
- **GROMACS version:** 2026.2
- **GPU support:** CUDA
- **CUDA compiler:** nvcc, release 13.2
- **Colvars support:** enabled
- **Plumed support:** enabled
- **CPU FFT library:** fftw (built-in, from `GMX_BUILD_OWN_FFTW=ON`)
- **GPU FFT library:** cuFFT

If GPU support does not show CUDA, or Colvars/PLUMED support show
disabled, double check that CUDA (Step 1) and PLUMED (Step 3) were
correctly installed and detectable *before* running `cmake` --
CMake detects these at configure time, so installing them afterward
requires re-running `cmake` (Step 5) before rebuilding (Step 6).

