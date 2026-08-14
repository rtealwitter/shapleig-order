#!/bin/bash
# Cheap correctness smoke test: runs a real HybridPairedEIG smoke sweep
# twice (compute_AKZZA_new vs compute_AKZZA_fast monkeypatched in),
# confirming no crashes/shape errors/non-finite outputs before the
# expensive multi-seed MSE comparison. Does NOT assert archive match --
# see validate_akzza_fast_smoke.py's docstring for why.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-validate-akzza-fast-smoke
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/validate_akzza_fast_smoke_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting compute_AKZZA_fast smoke test"
$PY ../validate_akzza_fast_smoke.py
echo "[$(date)] rc=$?"
