#!/bin/bash
# Cheap smoke test for the new repro_all_*_hybrid_akzza.yaml configs before
# launching the multi-hour production sweeps -- confirms
# application.use_akzza_fast is actually wired through hydra instantiation
# for all three config families.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-smoke-hybrid-akzza-config
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_hybrid_akzza_config_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting hybrid_akzza config smoke test"
$PY ../smoke_hybrid_akzza_config.py
echo "[$(date)] rc=$?"
