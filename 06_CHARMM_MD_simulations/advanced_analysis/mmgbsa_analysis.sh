#!/bin/bash
# Runs complete MM-GBSA pipeline for all replicas:
#  FAD | P-carnitine | Carnitine 

set -e   # exit on first error

GMX="gmx"
gmx_MMPBSA="gmx_MMPBSA"
CONDA_ENV="mda"

# Replica definitions
# Format: "BASE_DIR  XTC_PATH  LIGAND_TAG  OUT_SUBDIR"
declare -a REPLICAS=(
    # FAD replicas
    ".../rep1 step7_production_fit.xtc  FAD  MMGBSA_rep1"
    ".../rep2 step7_production_fit.xtc  FAD  MMGBSA_rep2"
    ".../rep3 step7_production_fit.xtc  FAD  MMGBSA_rep3"
    # P-carnitine replicas
    ".../rep1 step7_production_fit.xtc  PCAR  MMGBSA_rep1"
    ".../rep2 step7_production_fit.xtc  PCAR  MMGBSA_rep2"
    ".../rep3 step7_production_fit.xtc  PCAR  MMGBSA_rep3"
    # Free carnitine (m-state) replicas
    ".../rep1 step7_production_fit.xtc  CAR  MMGBSA_rep1"
    ".../rep2 step7_production_fit.xtc  CAR  MMGBSA_rep2"
    ".../rep3 step7_production_fit.xtc  CAR  MMGBSA_rep3"
)

# MMGBSA input file
MMGBSA_INPUT='&general
   startframe   = 1,
   endframe     = 10001,
   interval     = 100,
   verbose      = 2,
   keep_files   = 2,
/
&gb
   igb          = 2,
   saltcon      = 0.150,
/
&decomp
   idecomp      = 2,
   dec_verbose  = 3,
   print_res    = "within 6",
/
'

# Function: run one replica
run_replica() {
    local BASE_DIR="$1"
    local XTC_REL="$2"
    local TAG="$3"
    local OUT="$4"

    echo ""
    echo "================================================================"
    echo "  Processing: $BASE_DIR"
    echo "  Ligand: $TAG  |  Output: $OUT"
    echo "================================================================"

    cd "$BASE_DIR"
    local XTC="$BASE_DIR/$XTC_REL"
    local TPR="$BASE_DIR/step7_production.tpr"
    local GRO
    # Find the GRO file (may be in subdirectory for FAD rep1)
    if [ -f "$BASE_DIR/step7_production.gro" ]; then
        GRO="$BASE_DIR/step7_production.gro"
    else
        GRO=$(find "$BASE_DIR" -name "step7_production.gro" | head -1)
    fi

    echo "  TPR: $TPR"
    echo "  XTC: $XTC"
    echo "  GRO: $GRO"

    #  Step 1: Make index 
    if [ ! -f "complex_mmpbsa.ndx" ]; then
        echo "  [1/4] Making index..."
        printf "1 | 13\nname 29 Protein_UNL\nq\n" | \
            $GMX make_ndx -f "$TPR" -o complex_mmpbsa.ndx 2>/dev/null
        echo "  Done: complex_mmpbsa.ndx"
    else
        echo "  [1/4] complex_mmpbsa.ndx exists — skipping"
    fi

    #  Step 2: Reduced TPR 
    if [ ! -f "complex_prot_UNL.tpr" ]; then
        echo "  [2/4] Making reduced TPR..."
        echo "29" | $GMX convert-tpr \
            -s "$TPR" \
            -n complex_mmpbsa.ndx \
            -o complex_prot_UNL.tpr 2>/dev/null
        echo "  Done: complex_prot_UNL.tpr"
    else
        echo "  [2/4] complex_prot_UNL.tpr exists — skipping"
    fi

    #  Step 3: Reference PDB 
    if [ ! -f "complex_prot_UNL.pdb" ]; then
        echo "  [3/4] Extracting reference PDB (frame 0)..."
        echo "29" | $GMX trjconv \
            -s "$TPR" \
            -f "$XTC" \
            -n complex_mmpbsa.ndx \
            -dump 0 \
            -o complex_prot_UNL.pdb 2>/dev/null
        echo "  Done: complex_prot_UNL.pdb"
    else
        echo "  [3/4] complex_prot_UNL.pdb exists — skipping"
    fi

    #  Step 4: Reduced topology via parmed 
    if [ ! -f "complex_prot_UNL.top" ]; then
        echo "  [4/4] Generating reduced topology via parmed..."
        conda run -n $CONDA_ENV python3 -c "
import parmed as pmd
full = pmd.load_file('topol.top', xyz='$GRO')
cx = full[':1-301']
cx.save('complex_prot_UNL.top', overwrite=True)
print('  parmed done:', len(cx.atoms), 'atoms')
"
    else
        echo "  [4/4] complex_prot_UNL.top exists — skipping"
    fi

    #  Step 5: Run gmx_MMPBSA 
    local H5="$BASE_DIR/$OUT/RESULTS_gmx_MMPBSA.h5"
    if [ -f "$H5" ]; then
        echo "  [5/5] $H5 exists — skipping gmx_MMPBSA"
    else
        echo "  [5/5] Running gmx_MMPBSA (this takes 30-90 min)..."
        rm -rf "$OUT" && mkdir -p "$OUT"
        echo "$MMGBSA_INPUT" > "$OUT/mmpbsa.in"

        $gmx_MMPBSA -O \
            -i "$OUT/mmpbsa.in" \
            -cs complex_prot_UNL.tpr \
            -ct "$XTC" \
            -ci complex_mmpbsa.ndx \
            -cg 1 13 \
            -cp complex_prot_UNL.top \
            -cr complex_prot_UNL.pdb \
            -o  "$OUT/FINAL_RESULTS_MMGBSA.dat" \
            -do "$OUT/FINAL_DECOMP_MMGBSA.dat" \
            -eo "$OUT/ENERGIES.csv" \
            -prefix "$OUT/_MMGBSA_" \
            -nogui 2>&1 | tee "$OUT/run.log"

        if [ -f "$H5" ]; then
            echo "  gmx_MMPBSA completed: $H5"
        else
            echo "  WARNING: H5 file not found after run — check $OUT/run.log"
        fi
    fi

    echo "  Replica done: $BASE_DIR / $OUT"
}

# Main loop
echo "Starting MM-GBSA pipeline for all replicas..."
echo "Time: $(date)"

for entry in "${REPLICAS[@]}"; do
    read -r BASE XTC TAG OUT <<< "$entry"
    run_replica "$BASE" "$XTC" "$TAG" "$OUT"
done

echo ""
echo "================================================================"
echo "ALL REPLICAS COMPLETE"
echo "Time: $(date)"
echo ""
echo "H5 files produced:"
for entry in "${REPLICAS[@]}"; do
    read -r BASE XTC TAG OUT <<< "$entry"
    H5="$BASE/$OUT/RESULTS_gmx_MMPBSA.h5"
    if [ -f "$H5" ]; then
        echo "  [OK]  $H5"
    else
        echo "  [MISSING] $H5"
    fi
done
echo ""
echo "Next step — parse and plot:"
echo "  conda activate mda"
echo "  python3 parse_mmgbsa_dat.py"
echo "================================================================"
