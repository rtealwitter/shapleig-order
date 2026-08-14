#!/bin/bash
# Fast-fit companion to full_dv10_v2.sbatch: 3 DV games (p=10) x 1
# acquisition (PairedExtremes only as of 2026-08-14 -- HybridPairedEIG
# moved to full_dv10_v2_hybrid_akzza.sbatch, see repro_all_dv10_fastfit.yaml's
# comment) x 30 seeds, 512 iterations, exhaustive candidates,
# AcceleratedFitConfig (CG/Lanczos + inducing-point transplant + odd-part
# folding, src/xac/surrogates/fast_fit.py) instead of the exact MLM fit. The
# exact-fit counterpart for PairedExtremes (plus ShaplEIG/GP+Leverage/
# GP+Random) comes from the unmodified full_dv10_v2.sbatch; repro_plots.py
# distinguishes the two by the "+fast" suffix it adds when
# surrogate.fit_config._target_ is AcceleratedFitConfig. Appends its root
# to sweep_roots_v2.txt.
#
# mem=500gb (up from 200gb, which matches the exact baseline's own
# allocation): a single-worker diagnostic (diag_memory.py) showed
# fit_accelerated's own memory use is bounded, not leaking, so the earlier
# OOM (twice, at ~22min, t~450-500) was aggregate peak across the 90
# joblib workers, not a per-call leak -- each accelerated fit builds a
# second (auxiliary) model on top of the real one, raising per-call peak
# footprint over the baseline's single-model fit, and with all workers'
# archives growing in lockstep with iteration index, their peaks land in
# near-synchrony late in the run.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=500gb
#SBATCH --time=08:00:00
#SBATCH --job-name=shapleig-full-dv10-fastfit
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/full_dv10_fastfit_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

# `ls -td multirun/*/* | head -1`, and even the before/after snapshot-diff
# that replaced it, both infer this run's own directory from what's new in
# the shared multirun/ tree -- and that inference still collides when two
# jobs are submitted within the same wall-clock second (confirmed: the
# 267757 dv10 resubmit's own log recorded vit_9's directory as its root,
# because vit_9's directory also appeared inside dv10's before/after
# window). Suffixing hydra's own sweep dir with this job's SLURM_JOB_ID
# makes the directory name unique by construction, so there is nothing
# left to infer or race.
echo "[$(date)] node $(hostname) starting dv10 fast-fit sweep (90 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_dv10_fastfit \
    hydra.sweep.dir='multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}-'"$SLURM_JOB_ID"
rc=$?
echo "[$(date)] sweep done rc=$rc"

latest=$(find multirun -mindepth 2 -maxdepth 2 -type d -name "*-${SLURM_JOB_ID}" 2>/dev/null | head -1)
if [ -z "$latest" ]; then
    echo "[$(date)] WARNING: could not find this job's own multirun dir (looked for *-${SLURM_JOB_ID}); not recording a root."
else
    echo "$PWD/$latest" >> /hopper/groups/witterlab/rwitter/scratch/shapleig-order/sweep_roots_v2.txt
    echo "[$(date)] root recorded: $latest"
fi
