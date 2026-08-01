#!/bin/bash
# Aggregate the v2 sweeps listed in sweep_roots_v2.txt and regenerate the
# per-game three-panel figures (MSE / method compute / selection size).
# Safe to rerun any time; uses only the recorded v2 roots.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --job-name=shapleig-aggregate-v2
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/agg_v2_%j.out

source ~/.bashrc
D=/hopper/groups/witterlab/rwitter/scratch/shapleig-order
PY=$D/.venv-repro/bin/python
export MPLBACKEND=Agg

roots=$(cat $D/sweep_roots_v2.txt | sort -u | tr '\n' ' ')
echo "[$(date)] aggregating roots: $roots"
$PY $D/repro_plots.py $roots
echo "[$(date)] done rc=$?"
