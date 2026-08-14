#!/bin/bash
# Multi-seed MSE comparison for the incremental-Cholesky change: real
# resnet_14 HybridPairedEIG + AcceleratedFitConfig runs, incremental
# Cholesky forced off vs. on, N_SEEDS seeds each in parallel via joblib.
# Bit-identical archives turned out to be the wrong bar (see
# validate_incremental_cholesky.py's docstring) -- this checks the bar
# that actually matters: aggregate MSE performance across seeds.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --job-name=shapleig-validate-cholesky-mse
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/validate_cholesky_mse_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting incremental Cholesky multi-seed MSE validation"
$PY ../validate_incremental_cholesky_mse.py
echo "[$(date)] rc=$?"
