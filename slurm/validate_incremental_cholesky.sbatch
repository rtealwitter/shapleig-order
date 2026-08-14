#!/bin/bash
# Correctness check: runs a real HybridPairedEIG smoke sweep end to end
# with ShapleyApplication._cholesky_maybe_incremental wrapped to compare
# every incremental-update call against an independent full recompute,
# asserting they agree, before trusting the incremental Cholesky update on
# real archives.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-validate-cholesky
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/validate_cholesky_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting incremental Cholesky validation (forced-full vs incremental)"
$PY ../validate_incremental_cholesky.py
echo "[$(date)] rc=$?"
