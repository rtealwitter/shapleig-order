#!/bin/bash
# Cheap smoke test for the 2026-08-14 readout-reuse change to Hybrid
# (experiment_runner.py): confirms Hybrid's readout no longer pays a
# separate fit, and PairedExtremes is unaffected.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-smoke-hybrid-readout-reuse
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_hybrid_readout_reuse_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting hybrid readout-reuse smoke test"
$PY ../smoke_hybrid_readout_reuse.py
echo "[$(date)] rc=$?"
