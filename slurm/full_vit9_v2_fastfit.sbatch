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

# `ls -td multirun/*/* | head -1`, and even the before/after snapshot-diff
# that replaced it, both infer this run's own directory from what's new in
# the shared multirun/ tree -- and that inference still collides when two
# jobs are submitted within the same wall-clock second (confirmed on the
# dv10 companion script: two sibling jobs' directories both appeared
# inside the same before/after window). Suffixing hydra's own sweep dir
# with this job's SLURM_JOB_ID makes the directory name unique by
# construction, so there is nothing left to infer or race.
echo "[$(date)] node $(hostname) starting vit_9 fast-fit sweep (60 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_vit9_fastfit \
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
