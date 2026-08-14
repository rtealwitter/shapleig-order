"""Multi-seed MSE + timing comparison for compute_AKZZA_fast (see
applications.py's compute_AKZZA_fast docstring): the bar that actually
matters, per the lesson already learned validating the incremental-
Cholesky change -- bit-identical archive trajectories are the wrong bar
for anything touching HybridPairedEIG's greedy selection, since any
nonzero floating-point difference eventually flips a near-tied EIG
comparison. Aggregate MSE across seeds, matching how the published figure
itself validates arms against each other, is what's checked here.

Unlike the Cholesky work, compute_AKZZA_new is called by EVERY arm
(ShaplEIG, exact Hybrid, fast Hybrid) with no AcceleratedFitConfig gate --
this run uses HybridPairedEIG + AcceleratedFitConfig only for consistency
with the existing validated config (see HANDOFF_AKZZA.md), not because the
change is gated to it; compute_AKZZA_fast's correctness does not depend on
fit_config.

Runs N_SEEDS real resnet_14 HybridPairedEIG experiments at full iteration
count, once with compute_AKZZA_new (reference) and once with
compute_AKZZA_fast substituted in, in parallel via joblib, and compares
mean+/-SEM MSE and total wall time per condition.
"""

import sys
import time

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from joblib import Parallel, delayed

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"
GAME = "resnet_14"
N_SEEDS = 8
ITERATIONS = 512
N_JOBS = 16


def run_once(seed: int, use_fast: bool):
    from xac.applications.applications import ShapleyApplication
    from xac.experimental_designs.experiment_runner import run_experiment
    from xac.utils.random_utils import set_seed

    # joblib (loky backend) reuses worker processes across tasks -- restore
    # the original method in `finally` regardless of which branch ran, or
    # it leaks into a later, unrelated task on the same reused worker.
    original_method = ShapleyApplication.compute_AKZZA_new
    try:
        if use_fast:
            ShapleyApplication.compute_AKZZA_new = ShapleyApplication.compute_AKZZA_fast

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            cfg = compose(
                config_name="repro_all_p1416_fastfit",
                overrides=[
                    f"meta.seed={seed}",
                    f"blackbox.name={GAME}",
                    "acquisition._target_=xac.acquisition_functions.HybridPairedEIG",
                    f"experimental_design.iterations={ITERATIONS}",
                ],
            )

        set_seed(cfg.meta.seed)

        application = hydra.utils.instantiate(cfg.application)
        surrogate_cfg = hydra.utils.instantiate(cfg.surrogate)
        acquisition_fn = hydra.utils.instantiate(cfg.acquisition)
        acquisition_optimizer = hydra.utils.instantiate(cfg.acquisition_optimizer)
        blackbox_fn = hydra.utils.instantiate(cfg.blackbox)
        ed_cfg = hydra.utils.instantiate(cfg.experimental_design)
        meta_cfg = hydra.utils.instantiate(cfg.meta)

        import tempfile
        run_dir = tempfile.mkdtemp()

        start = time.perf_counter()
        (
            (prop_posts, prop_posts_noisy, test_predictions, test_labels,
             prop_gt, sv_approximations),
            (archive_x, archive_y),
            (hp_fit_durations, prop_post_durations, acq_fun_durations,
             eval_fit_durations, peak_rss_mb),
        ) = run_experiment(
            application=application,
            blackbox_fn=blackbox_fn,
            surrogate_cfg=surrogate_cfg,
            acquisition_fn=acquisition_fn,
            acquisition_optimizer=acquisition_optimizer,
            ed_cfg=ed_cfg,
            meta_cfg=meta_cfg,
            run_dir=run_dir,
        )
        elapsed = time.perf_counter() - start
    finally:
        ShapleyApplication.compute_AKZZA_new = original_method

    from xac.utils.metrics import compute_mse
    mse_trace = np.array([compute_mse(p, prop_gt) for p in prop_posts])
    total_selection_time = float(np.nansum(hp_fit_durations)) + float(np.nansum(acq_fun_durations))
    return {
        "seed": seed,
        "use_fast": use_fast,
        "elapsed": elapsed,
        "total_selection_time": total_selection_time,
        "mse_final": float(mse_trace[-1]),
        "mse_trace_len": len(mse_trace),
        "archive_size": int(archive_x.shape[0]),
    }


def main():
    seeds = list(range(1, N_SEEDS + 1))
    jobs = [(s, False) for s in seeds] + [(s, True) for s in seeds]
    print(f"Running {len(jobs)} jobs ({N_SEEDS} seeds x 2 conditions) on {GAME}, "
          f"{ITERATIONS} iterations, n_jobs={N_JOBS}...", flush=True)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_once)(seed, use_fast) for seed, use_fast in jobs
    )

    ref = [r for r in results if not r["use_fast"]]
    fast = [r for r in results if r["use_fast"]]

    mse_ref = np.array([r["mse_final"] for r in ref])
    mse_fast = np.array([r["mse_final"] for r in fast])
    time_ref = np.array([r["total_selection_time"] for r in ref])
    time_fast = np.array([r["total_selection_time"] for r in fast])
    wall_ref = np.array([r["elapsed"] for r in ref])
    wall_fast = np.array([r["elapsed"] for r in fast])

    def mean_sem(a):
        return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

    mr, sr = mean_sem(mse_ref)
    mf, sf = mean_sem(mse_fast)
    tr, str_ = mean_sem(time_ref)
    tf, stf = mean_sem(time_fast)
    wr, swr = mean_sem(wall_ref)
    wf, swf = mean_sem(wall_fast)

    print(f"\n=== {GAME} HybridPairedEIG, {N_SEEDS} seeds, {ITERATIONS} iterations ===", flush=True)
    print(f"final MSE:            ref={mr:.4e} +/- {sr:.4e}   fast={mf:.4e} +/- {sf:.4e}", flush=True)
    print(f"  |mean diff| / SEM(ref+fast combined): "
          f"{abs(mr - mf) / np.sqrt(sr**2 + sf**2):.2f} (overlapping CIs if < ~2)", flush=True)
    print(f"total selection time (hp_fit+acq_fun): ref={tr:.1f}s +/- {str_:.1f}   "
          f"fast={tf:.1f}s +/- {stf:.1f}   speedup={tr/tf:.2f}x", flush=True)
    print(f"wall clock per run:   ref={wr:.1f}s +/- {swr:.1f}   fast={wf:.1f}s +/- {swf:.1f}", flush=True)

    print("\nper-seed detail:", flush=True)
    for s in seeds:
        rr = next(r for r in ref if r["seed"] == s)
        rf = next(r for r in fast if r["seed"] == s)
        print(f"  seed {s:2d}: mse_ref={rr['mse_final']:.4e} mse_fast={rf['mse_final']:.4e}  "
              f"sel_time_ref={rr['total_selection_time']:6.1f}s sel_time_fast={rf['total_selection_time']:6.1f}s "
              f"archive_ref={rr['archive_size']} archive_fast={rf['archive_size']}", flush=True)


if __name__ == "__main__":
    main()
