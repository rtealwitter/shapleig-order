#!/bin/bash
# Targeted check of the InducingPointKernel path in fast_fit.py, which the
# main smoke_fastfit.sbatch never exercises (archive there tops out at
# t=59 < the default inducing_points=64 threshold). Same dv10 smoke config,
# but with inducing_points overridden to 16 via the hydra CLI so
# InducingPointKernel activates well before the run ends.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --job-name=shapleig-smoke-inducing
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_inducing_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname) fastfit with inducing_points=16 forced"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_smoke_dv10_fastfit \
    surrogate.fit_config.inducing_points=16
rc=$?
echo "[$(date)] rc=$rc"
root=$(ls -td multirun/*/* | head -1)
echo "[$(date)] root: $root"

echo "=== inducing activations ==="
grep -c "inducing=16" "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_inducing_${SLURM_JOB_ID}.out" || true

$PY - "$root" <<'EOF'
import glob, json, os, sys
import numpy as np
import yaml

root = sys.argv[1]
for mpath in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
    run_dir = os.path.dirname(mpath)
    cfg = yaml.safe_load(open(os.path.join(run_dir, ".hydra", "config.yaml")))
    met = json.load(open(mpath))
    acq = cfg["acquisition"]["_target_"].split(".")[-1]
    mse = met.get("mse") or []
    print(f"{acq:20s} seed={cfg['meta']['seed']} final_mse={mse[-1]:.3e}" if mse else f"{acq}: no mse")
EOF
echo "[$(date)] done"
