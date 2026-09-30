#!/bin/bash

set -e

echo "===== STEP 6.0: ENERGY MINIMIZATION ====="

gmx grompp \
-f step6.0_minimization.mdp \
-c step5_input.gro \
-r step5_input.gro \
-p topol.top \
-n index.ndx \
-o em.tpr

gmx mdrun \
-deffnm em \
-ntmpi 1 \
-ntomp 24 \
-v

echo "===== STEP 6.1 ====="

gmx grompp \
-f step6.1_equilibration.mdp \
-c em.gro \
-r em.gro \
-p topol.top \
-n index.ndx \
-o eq1.tpr

gmx mdrun \
-deffnm eq1 \
-ntmpi 1 \
-ntomp 24 \
-gpu_id 0 \
-nb gpu \
-pme gpu \
-pin on \
-dlb yes \
-v

echo "===== STEP 6.2 ====="

gmx grompp \
-f step6.2_equilibration.mdp \
-c eq1.gro \
-r eq1.gro \
-p topol.top \
-n index.ndx \
-o eq2.tpr

gmx mdrun \
-deffnm eq2 \
-ntmpi 1 \
-ntomp 24 \
-gpu_id 0 \
-nb gpu \
-pme gpu \
-pin on \
-dlb yes \
-v

echo "===== STEP 6.3 ====="

gmx grompp \
-f step6.3_equilibration.mdp \
-c eq2.gro \
-r eq2.gro \
-p topol.top \
-n index.ndx \
-o eq3.tpr

gmx mdrun \
-deffnm eq3 \
-ntmpi 1 \
-ntomp 24 \
-gpu_id 0 \
-nb gpu \
-pme gpu \
-pin on \
-dlb yes \
-v

echo "===== STEP 6.4 ====="

gmx grompp \
-f step6.4_equilibration.mdp \
-c eq3.gro \
-r eq3.gro \
-p topol.top \
-n index.ndx \
-o eq4.tpr

gmx mdrun \
-deffnm eq4 \
-ntmpi 1 \
-ntomp 24 \
-gpu_id 0 \
-nb gpu \
-pme gpu \
-pin on \
-dlb yes \
-v

echo "===== STEP 6.5 ====="

gmx grompp \
-f step6.5_equilibration.mdp \
-c eq4.gro \
-r eq4.gro \
-p topol.top \
-n index.ndx \
-o eq5.tpr

gmx mdrun \
-deffnm eq5 \
-ntmpi 1 \
-ntomp 24 \
-gpu_id 0 \
-nb gpu \
-pme gpu \
-pin on \
-dlb yes \
-v

echo "===== STEP 6.6 ====="

gmx grompp \
-f step6.6_equilibration.mdp \
-c eq5.gro \
-r eq5.gro \
-p topol.top \
-n index.ndx \
-o eq6.tpr

gmx mdrun \
-deffnm eq6 \
-ntmpi 1 \
-ntomp 24 \
-gpu_id 0 \
-nb gpu \
-pme gpu \
-pin on \
-dlb yes \
-v

echo "===== STEP 7: PRODUCTION MD ====="

gmx grompp \
-f step7_production.mdp \
-c eq6.gro \
-t eq6.cpt \
-p topol.top \
-n index.ndx \
-o step7_production.tpr \
-maxwarn 1

gmx mdrun \
-deffnm step7_production \
-ntmpi 1 \
-ntomp 24 \
-gpu_id 0 \
-nb gpu \
-pme gpu \
-pin on \
-dlb yes \
-v

echo "===== PRODUCTION MD FINISHED SUCCESSFULLY ====="
