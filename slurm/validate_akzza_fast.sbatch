#!/bin/bash
# Correctness + timing battery: compute_AKZZA_fast vs compute_AKZZA_new
# across p=2..32, 4 seeds each, torch.allclose gate before compute_AKZZA_fast
# is ever run inside a real experiment (HANDOFF_AKZZA.md validation protocol,
# stage 1).
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-validate-akzza-fast
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/validate_akzza_fast_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting compute_AKZZA_fast correctness+timing battery"
$PY ../validate_akzza_fast.py
echo "[$(date)] rc=$?"
