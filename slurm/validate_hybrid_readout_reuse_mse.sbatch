#!/bin/bash
# Multi-seed MSE + timing comparison for the readout-reuse change to
# Hybrid: 8 fresh resnet_14 seeds under the NEW code, compared against the
# OLD-code production data already collected this session.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --job-name=shapleig-validate-hybrid-readout-reuse-mse
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/validate_hybrid_readout_reuse_mse_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting hybrid readout-reuse multi-seed MSE validation"
$PY ../validate_hybrid_readout_reuse_mse.py
echo "[$(date)] rc=$?"
