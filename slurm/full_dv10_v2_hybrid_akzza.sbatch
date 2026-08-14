#!/bin/bash
# AKZZA-accelerated Hybrid companion to full_dv10_v2.sbatch: 3 DV games
# (p=10) x 1 acquisition (HybridPairedEIG only) x 30 seeds, 512 iterations,
# exhaustive candidates, exact MLM fit (not AcceleratedFitConfig -- see
# repro_all_dv10_hybrid_akzza.yaml's comment and HANDOFF_AKZZA.md's
# 2026-08-14 investigation for why), application.use_akzza_fast=true so
# compute_AEA_new uses compute_AKZZA_fast instead of compute_AKZZA_new.
# This replaces the old exact-fit Hybrid arm's data (repro_plots.py drops
# the old HybridPairedEIG runs from the pre-existing sweep roots and
# HybridPairedEIG+fast entirely once this lands). Same resource sizing as
# the proven-safe full_dv10_v2.sbatch (only the acquisition/application
# differ, so eval_fit_duration -- the dominant per-run cost, blackbox-model
# retraining, independent of acquisition strategy -- is unchanged). Appends
# its root to sweep_roots_hybrid_akzza.txt.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
#SBATCH --job-name=shapleig-full-dv10-hybrid-akzza
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/full_dv10_hybrid_akzza_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname) starting dv10 Hybrid+AKZZA sweep (90 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_dv10_hybrid_akzza \
    hydra.sweep.dir='multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}-'"$SLURM_JOB_ID"
rc=$?
echo "[$(date)] sweep done rc=$rc"

latest=$(find multirun -mindepth 2 -maxdepth 2 -type d -name "*-${SLURM_JOB_ID}" 2>/dev/null | head -1)
if [ -z "$latest" ]; then
    echo "[$(date)] WARNING: could not find this job's own multirun dir (looked for *-${SLURM_JOB_ID}); not recording a root."
else
    echo "$PWD/$latest" >> /hopper/groups/witterlab/rwitter/scratch/shapleig-order/sweep_roots_hybrid_akzza.txt
    echo "[$(date)] root recorded: $latest"
fi
