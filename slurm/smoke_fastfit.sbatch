#!/bin/bash
# Smoke test + validation of the new AcceleratedFitConfig fitting routine
# (fast_fit.py: iterative CG/Lanczos, inducing-point transplant, odd-part
# folding) against the exact baseline, on dvbsgb_10 (p=10), 2 seeds, 48
# iterations, exhaustive candidates -- small enough to run on debug.
#
# Runs (a) the unmodified repro_all_smoke_dv10 (exact fit, all 5 arms,
# baseline for comparison) and (b) the new repro_all_smoke_dv10_fastfit
# (PairedExtremes + HybridPairedEIG, AcceleratedFitConfig), then prints a
# side-by-side MSE/timing comparison for the two arms that exist in both.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-smoke-fastfit
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_fastfit_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname) baseline: dv10 exact fit, all arms"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_smoke_dv10
echo "[$(date)] baseline rc=$?"
root_exact=$(ls -td multirun/*/* | head -1)

echo "[$(date)] fastfit: dv10 AcceleratedFitConfig, PairedExtremes + HybridPairedEIG"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_smoke_dv10_fastfit
echo "[$(date)] fastfit rc=$?"
root_fast=$(ls -td multirun/*/* | head -1)

echo "[$(date)] roots: exact=$root_exact fast=$root_fast"

$PY - "$root_exact" "$root_fast" <<'EOF'
import glob, json, os, sys
import numpy as np

root_exact, root_fast = sys.argv[1], sys.argv[2]

def load(root):
    runs = []
    for mpath in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        met = json.load(open(mpath))
        runs.append(met)
    return runs

exact = load(root_exact)
fast = load(root_fast)

print(f"\ncollected {len(exact)} exact runs, {len(fast)} fast-fit runs\n")

FAST_ARMS = ["xac.acquisition_functions.PairedExtremes",
             "xac.acquisition_functions.HybridPairedEIG"]

def summarize(runs, acq):
    sel = [r for r in runs if r.get("acquisition") == acq]
    if not sel:
        return None
    mse_last = np.mean([r["mse"][-1] for r in sel if r.get("mse")])
    hp = np.mean([sum(t for t in (r.get("hp_fit_duration") or []) if t > 0) for r in sel])
    ev = np.mean([sum(t for t in (r.get("eval_fit_duration") or []) if t > 0) for r in sel])
    n_fits = np.mean([sum(1 for t in (r.get("eval_fit_duration") or []) if t > 0)
                       + sum(1 for t in (r.get("hp_fit_duration") or []) if t > 0)
                       for r in sel])
    return dict(n=len(sel), mse=mse_last, hp_s=hp, eval_s=ev, n_fits=n_fits)

ok = True
for acq in FAST_ARMS:
    e = summarize(exact, acq)
    f = summarize(fast, acq)
    print(f"=== {acq.split('.')[-1]} ===")
    print(f"  exact    : n={e['n'] if e else 0} final_mse={e['mse']:.3e} "
          f"hp_fit={e['hp_s']:.2f}s eval_fit={e['eval_s']:.2f}s n_fits={e['n_fits']:.0f}" if e else "  exact    : MISSING")
    print(f"  fast-fit : n={f['n'] if f else 0} final_mse={f['mse']:.3e} "
          f"hp_fit={f['hp_s']:.2f}s eval_fit={f['eval_s']:.2f}s n_fits={f['n_fits']:.0f}" if f else "  fast-fit : MISSING")
    if e is None or f is None:
        print("  FAIL: one side missing runs")
        ok = False
        continue
    ratio = f["mse"] / e["mse"] if e["mse"] > 0 else float("nan")
    speedup = (e["hp_s"] + e["eval_s"]) / max(f["hp_s"] + f["eval_s"], 1e-9)
    print(f"  mse ratio (fast/exact) = {ratio:.2f}   fit-time speedup = {speedup:.2f}x")
    if not (0.1 <= ratio <= 10.0):
        print(f"  WARNING: MSE ratio {ratio:.2f} outside [0.1, 10] -- accuracy may have degraded")
        ok = False
    print()

print("SMOKETEST " + ("PASS" if ok else "CHECK (see warnings above)"))
EOF
echo "[$(date)] smoke_fastfit summary done"
