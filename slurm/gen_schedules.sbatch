#!/bin/bash
# Generate fixed paired-extremes schedules for p=9,14,16 (p=10 exists).
# Resumable: reruns skip schedules already on disk. p=16 runs a dense greedy
# over 2^16 coalitions for 600 steps, the slow part (~1h).
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=06:00:00
#SBATCH --job-name=shapleig-gen-schedules
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/gen_schedules_%j.out

source ~/.bashrc
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=8

echo "[$(date)] node $(hostname) generating schedules"
$PY /hopper/groups/witterlab/rwitter/scratch/shapleig-order/greedy/gen_schedules.py
echo "[$(date)] done rc=$?"
ls -la /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/data/
