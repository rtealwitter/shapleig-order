#!/bin/bash
# Full aligned replication sweep: 3 DV games x 4 acquisitions x 30 seeds,
# 512 iterations each, refit every iteration, exhaustive candidate scoring —
# the paper's Figure 3 setup plus our PairedExtremes arm. Hydra/joblib runs
# 100 jobs in parallel; resubmitting reruns the whole sweep (hydra makes a
# new multirun dir), so only resubmit after wiping or on total failure.
# Afterwards, aggregate and plot.
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=200gb
#SBATCH --time=06:00:00
#SBATCH --job-name=shapleig-hybrid-full
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/hybrid_full_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname) starting full sweep"
$PY -m xac.experiments.cli --config-path conf --config-name repro_dv10_hybrid
rc=$?
echo "[$(date)] sweep done rc=$rc"

latest=$(ls -td multirun/*/* | head -1)
