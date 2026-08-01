#!/bin/bash
# Aggregate the 360-run main sweep + 90-run hybrid sweep and render the
# three-panel figures. Rerunnable; overwrites hybrid_*.png/svg.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:20:00
#SBATCH --job-name=shapleig-agg
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/agg_%j.out
source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order
export MPLBACKEND=Agg
.venv-repro/bin/python repro_plots.py \
  shapleig-repo/multirun/2026-07-31/17-22-55 \
  shapleig-repo/multirun/2026-07-31/19-52-24/
echo "[$(date)] agg done rc=$?"
