#!/bin/bash
# Check AcceleratedFitConfig at production defaults (inducing_points=64,
# use_iterative=True, fold_odd_part=True) once the archive actually grows
# past the inducing-point threshold -- the two smoke_fastfit* jobs so far
# only reached t<=59 (< 64, so InducingPointKernel never activated at
# defaults) or forced an unrealistically small inducing_points=16. One
# game, one seed, PairedExtremes only, 300 iterations (archive ~311) so
# InducingPointKernel genuinely activates at m=64 well before the end.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-smoke-scale
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_scale_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] exact fit, 300 iterations, PairedExtremes, seed 1"
$PY -m xac.experiments.cli --config-path conf --config-name repro_scale_dv10_exact
rc=$?
echo "[$(date)] exact rc=$rc"
root_exact=$(ls -td multirun/*/* | head -1)

echo "[$(date)] fast fit, 300 iterations, PairedExtremes, seed 1, defaults (inducing_points=64)"
$PY -m xac.experiments.cli --config-path conf --config-name repro_scale_dv10_fastfit
rc=$?
echo "[$(date)] fast rc=$rc"
root_fast=$(ls -td multirun/*/* | head -1)

echo "[$(date)] roots: exact=$root_exact fast=$root_fast"
echo "=== inducing activations in fast log ==="
grep -c "inducing=64" "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_scale_${SLURM_JOB_ID}.out" || true
echo "=== optimizer failure warnings in fast log ==="
grep -c "OptimizationStatus.FAILURE" "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_scale_${SLURM_JOB_ID}.out" || true

$PY - "$root_exact" "$root_fast" <<'EOF'
import glob, json, os, sys

for label, root in zip(("exact", "fast"), sys.argv[1:]):
    for mpath in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        met = json.load(open(mpath))
        mse = met.get("mse") or []
        hp = sum(t for t in (met.get("hp_fit_duration") or []) if t > 0)
        ev = sum(t for t in (met.get("eval_fit_duration") or []) if t > 0)
        print(f"{label:6s} final_mse={mse[-1]:.4e} n_points={len(mse)} "
              f"hp_fit_total={hp:.2f}s eval_fit_total={ev:.2f}s")
EOF
echo "[$(date)] done"
