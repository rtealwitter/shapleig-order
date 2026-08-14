#!/bin/bash
# Multi-seed MSE comparison for compute_AKZZA_fast: real resnet_14
# HybridPairedEIG runs, compute_AKZZA_new (reference) vs compute_AKZZA_fast
# (monkeypatched in), N_SEEDS seeds each in parallel via joblib. Bit-
# identical archives is the wrong bar (see validate_akzza_fast_smoke.py's
# docstring) -- this checks the bar that actually matters: aggregate MSE
# performance across seeds.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --job-name=shapleig-validate-akzza-fast-mse
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/validate_akzza_fast_mse_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting compute_AKZZA_fast multi-seed MSE validation"
$PY ../validate_akzza_fast_mse.py
echo "[$(date)] rc=$?"
