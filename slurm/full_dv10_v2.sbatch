#!/bin/bash
# Full aligned replication sweep, restructured-refit semantics: 3 DV games
# (p=10) x 5 acquisitions (ShaplEIG, Hybrid, PairedExtremes, GP+Leverage,
# GP+Random) x 30 seeds, 512 iterations, exhaustive candidates. Method fits
# happen only where selection needs them; metrics readouts use separate
# eval fits (eval_fit_duration). Hydra makes a fresh multirun dir per submit,
# so only resubmit after a total failure. Appends its root to
# sweep_roots_v2.txt for the aggregate job.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
#SBATCH --job-name=shapleig-full-dv10-v2
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/full_dv10_v2_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname) starting dv10 sweep (450 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_dv10
rc=$?
echo "[$(date)] sweep done rc=$rc"
latest=$(ls -td multirun/*/* | head -1)
echo "$PWD/$latest" >> /hopper/groups/witterlab/rwitter/scratch/shapleig-order/sweep_roots_v2.txt
echo "[$(date)] root recorded: $latest"
