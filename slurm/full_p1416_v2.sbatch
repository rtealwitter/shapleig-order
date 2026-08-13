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
#SBATCH --cpus-per-task=64
#SBATCH --mem=480gb
#SBATCH --time=48:00:00
#SBATCH --job-name=shapleig-full-p1416-v2
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/full_p1416_v2_%j.out

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
# fastfit companion scripts. Suffixing hydra's own sweep dir with this
# job's SLURM_JOB_ID makes the directory name unique by construction, so
# there is nothing left to infer or race.
echo "[$(date)] node $(hostname) starting resnet_14+vit_16 sweep (300 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_p1416 \
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
