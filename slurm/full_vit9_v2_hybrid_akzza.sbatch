#!/bin/bash
# AKZZA-accelerated Hybrid companion to full_vit9_v2.sbatch: vit_9 (p=9)
# x 1 acquisition (HybridPairedEIG only) x 30 seeds, 490 iterations,
# exhaustive candidates, exact MLM fit. See full_dv10_v2_hybrid_akzza.sbatch
# for the full rationale. Appends its root to sweep_roots_hybrid_akzza.txt.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=200gb
#SBATCH --time=06:00:00
#SBATCH --job-name=shapleig-full-vit9-hybrid-akzza
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/full_vit9_hybrid_akzza_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname) starting vit_9 Hybrid+AKZZA sweep (30 jobs)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_vit9_hybrid_akzza \
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
