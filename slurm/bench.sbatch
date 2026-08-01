#!/bin/bash
# ShaplEIG baselines benchmark on shapiq games: precompute value tables, run
# the three methods over 8 seeds, plot. Each stage skips outputs that already
# exist, so a failure resumes by resubmitting this same script.
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --job-name=shapleig-bench
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/bench_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/bench
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv/bin/python
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname)"
echo "[$(date)] stage 1: game tables"
$PY prep_games.py || exit 1
echo "[$(date)] stage 2: benchmark"
$PY benchmark.py || exit 1
echo "[$(date)] stage 3: plots"
$PY plot_results.py || exit 1
echo "[$(date)] done"
