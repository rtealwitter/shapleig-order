#!/bin/bash
# Fast-fit companion to full_vit9_v2.sbatch: vit_9 (p=9) x 2 acquisitions
# (PairedExtremes, HybridPairedEIG) x 30 seeds, 490 iterations, exhaustive
# candidates, AcceleratedFitConfig instead of the exact MLM fit. See
# full_dv10_v2_fastfit.sbatch for the arm-matching note. Appends its root
# to sweep_roots_v2.txt.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=150gb
#SBATCH --time=05:00:00
#SBATCH --job-name=shapleig-full-vit9-fastfit
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/full_vit9_fastfit_%j.out

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
# own if one was created after this job's -- confirmed in practice on the
# dv10 companion script. Snapshot before/after and diff to identify this
# invocation's own directory regardless of what else is running.
before=$(mktemp)
find multirun -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort > "$before"

echo "[$(date)] node $(hostname) starting vit_9 fast-fit sweep (60 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_vit9_fastfit
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
