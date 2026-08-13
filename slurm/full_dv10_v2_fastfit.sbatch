#!/bin/bash
# Fast-fit companion to full_dv10_v2.sbatch: 3 DV games (p=10) x 2
# acquisitions (PairedExtremes, HybridPairedEIG) x 30 seeds, 512
# iterations, exhaustive candidates, AcceleratedFitConfig (CG/Lanczos +
# inducing-point transplant + odd-part folding, src/xac/surrogates/
# fast_fit.py) instead of the exact MLM fit. The exact-fit counterparts for
# these two arms (plus ShaplEIG/GP+Leverage/GP+Random) come from the
# unmodified full_dv10_v2.sbatch; repro_plots.py distinguishes the two by
# the "+fast" suffix it adds when surrogate.fit_config._target_ is
# AcceleratedFitConfig. Appends its root to sweep_roots_v2.txt.
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

# `ls -td multirun/*/* | head -1` picks the most-recently-*created*
# directory across the whole shared multirun/ tree, so it silently grabs
# another concurrently-running job's fresh directory instead of this run's
# own if one was created after this job's -- confirmed in practice (a
# crashed dv10 run recorded a directory that actually held a mix of dv10
# and vit_16 data from a job running at the same time). Snapshotting
# before/after and diffing identifies *this* invocation's own directory
# regardless of what else is running concurrently.
before=$(mktemp)
find multirun -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort > "$before"

echo "[$(date)] node $(hostname) starting dv10 fast-fit sweep (180 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_dv10_fastfit
rc=$?
echo "[$(date)] sweep done rc=$rc"

after=$(mktemp)
find multirun -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort > "$after"
latest=$(comm -13 "$before" "$after" | head -1)
rm -f "$before" "$after"
if [ -z "$latest" ]; then
    echo "[$(date)] WARNING: could not identify this run's own multirun dir; not recording a root."
else
    echo "$PWD/$latest" >> /hopper/groups/witterlab/rwitter/scratch/shapleig-order/sweep_roots_v2.txt
    echo "[$(date)] root recorded: $latest"
fi
