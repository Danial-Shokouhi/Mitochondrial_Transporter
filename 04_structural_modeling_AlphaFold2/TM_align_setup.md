# TMalign setup

Pick ONE of the following -- they are alternative ways to get the same
`TMalign` binary, not sequential steps:

**Option A -- build from source (dynamically linked)**
```
wget https://zhanggroup.org/TM-align/TMalign.cpp
g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp
```

**Option B -- build from source (statically linked, no runtime library
dependency, more portable across machines)**
```
wget https://zhanggroup.org/TM-align/TMalign.cpp
g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
```

**Option C -- install via package manager (Debian/Ubuntu)**
```
apt install tm-align
```

## Directory setup

In the same directory as the `TMalign` binary:
- copy the model `.pdb` files to be compared (only the un-relaxed
  models -- exclude any file with an `r0`/`r1` suffix)
- copy the two reference structures, `1OKC.pdb` and `6GCI.pdb`

Confirm everything is in place before running the analysis:
```
pwd
ls
```

## Running the analysis

```
chmod +x TM_align.sh
./TM_align.sh "80models_*.pdb"
```
(substitute whatever glob pattern matches your model files, e.g.
`"40models_*.pdb"` for the m-state run)

Produces `TM_results.csv` (all models) plus `Sorted_by_1OKC.csv` and
`Sorted_by_6GCI.csv` (ranked by similarity to each reference state).
