#!/bin/bash
# TM-align comparison of AlphaFold2 models against the c-state (1OKC)
# and m-state (6GCI) reference structures. See TM_align_setup.md for
# the one-time build/install steps and directory requirements.
#
# Usage:
#   ./run_TM_align.sh "<model_glob_pattern>"
# Examples:
#   ./run_TM_align.sh "80models_*.pdb"
#   ./run_TM_align.sh "40models_*.pdb"

MODEL_PATTERN="${1:?Usage: $0 <model_glob_pattern>, e.g. \"80models_*.pdb\"}"
REF_C="1OKC.pdb"
REF_M="6GCI.pdb"

echo "Model,TM_1OKC,TM_6GCI,RMSD_1OKC,RMSD_6GCI" > TM_results.csv

for pdb in $MODEL_PATTERN; do
    echo "Processing $pdb"
    out1=$(./TMalign "$pdb" "$REF_C")
    out2=$(./TMalign "$pdb" "$REF_M")

    # Second reported TM-score = normalized by the reference structure
    tm1=$(echo "$out1" | grep "TM-score=" | tail -1 | awk '{print $2}')
    tm2=$(echo "$out2" | grep "TM-score=" | tail -1 | awk '{print $2}')
    rmsd1=$(echo "$out1" | grep "Aligned length=" | awk -F'RMSD=' '{print $2}' | awk -F',' '{print $1}')
    rmsd2=$(echo "$out2" | grep "Aligned length=" | awk -F'RMSD=' '{print $2}' | awk -F',' '{print $1}')

    echo "$pdb,$tm1,$tm2,$rmsd1,$rmsd2" >> TM_results.csv
done

# Sort by similarity to c-state (1OKC)
{
    head -n1 TM_results.csv
    tail -n +2 TM_results.csv | sort -t, -k2,2gr
} > Sorted_by_1OKC.csv

# Sort by similarity to m-state (6GCI)
{
    head -n1 TM_results.csv
    tail -n +2 TM_results.csv | sort -t, -k3,3gr
} > Sorted_by_6GCI.csv

echo ""
echo "Finished."
echo ""
echo "Top c-state candidates:"
head Sorted_by_1OKC.csv
echo ""
echo "Top m-state candidates:"
head Sorted_by_6GCI.csv
