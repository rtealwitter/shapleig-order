"""Multi-seed MSE comparison for the incremental-Cholesky change (see
applications.py's _cholesky_maybe_incremental): bit-identical archive
trajectories turned out to be the wrong bar (HybridPairedEIG's greedy
selection is sensitive enough that ANY nonzero floating-point difference
eventually flips a near-tied decision and the trajectory branches -- this
is inherent to the algorithm, not specific to this change, and the
existing fast-fit path already accepts different-but-valid trajectories
from using different hyperparameters). The bar that actually matters,
matching how the published figure itself validates arms against each
other, is aggregate MSE performance across seeds.

Runs N_SEEDS real resnet_14 HybridPairedEIG + AcceleratedFitConfig
experiments (same config as the real fastfit sweep) at full iteration
count, once with the incremental Cholesky update forced off (matching
pre-change behavior) and once on, in parallel via joblib, and compares
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


def run_once(seed: int, force_full: bool):
    from xac.applications.applications import ShapleyApplication
    from xac.experimental_designs.experiment_runner import run_experiment
    from xac.utils.random_utils import set_seed

    # joblib (loky backend) reuses worker processes across tasks, so a
    # monkeypatch applied for one force_full=True task would otherwise leak
    # into a later force_full=False task on the same reused worker --
    # always restore the original method afterward, not just on the
    # force_full branch.
    original_method = ShapleyApplication._cholesky_maybe_incremental
    try:
        if force_full:
            from linear_operator.utils.cholesky import psd_safe_cholesky

            def always_full(self, surrogate, C, is_no_refit_step):
                L = psd_safe_cholesky(C.to_dense())
                object.__setattr__(self, "_L_cache", L)
                object.__setattr__(self, "_L_cache_chain_len", 0)
                return L

            ShapleyApplication._cholesky_maybe_incremental = always_full

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
        ShapleyApplication._cholesky_maybe_incremental = original_method

    from xac.utils.metrics import compute_mse
    mse_trace = np.array([compute_mse(p, prop_gt) for p in prop_posts])
    total_selection_time = float(np.nansum(hp_fit_durations)) + float(np.nansum(acq_fun_durations))
    return {
        "seed": seed,
        "force_full": force_full,
        "elapsed": elapsed,
        "total_selection_time": total_selection_time,
        "mse_final": float(mse_trace[-1]),
        "mse_trace_len": len(mse_trace),
        "archive_size": int(archive_x.shape[0]),
    }


def main():
    seeds = list(range(1, N_SEEDS + 1))
    jobs = [(s, True) for s in seeds] + [(s, False) for s in seeds]
    print(f"Running {len(jobs)} jobs ({N_SEEDS} seeds x 2 conditions) on {GAME}, "
          f"{ITERATIONS} iterations, n_jobs={N_JOBS}...", flush=True)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_once)(seed, force_full) for seed, force_full in jobs
    )

    full = [r for r in results if r["force_full"]]
    inc = [r for r in results if not r["force_full"]]

    mse_full = np.array([r["mse_final"] for r in full])
    mse_inc = np.array([r["mse_final"] for r in inc])
    time_full = np.array([r["total_selection_time"] for r in full])
    time_inc = np.array([r["total_selection_time"] for r in inc])
    wall_full = np.array([r["elapsed"] for r in full])
    wall_inc = np.array([r["elapsed"] for r in inc])

    def mean_sem(a):
        return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

    mf, sf = mean_sem(mse_full)
    mi, si = mean_sem(mse_inc)
    tf, stf = mean_sem(time_full)
    ti, sti = mean_sem(time_inc)
    wf, swf = mean_sem(wall_full)
    wi, swi = mean_sem(wall_inc)

    print(f"\n=== {GAME} HybridPairedEIG, {N_SEEDS} seeds, {ITERATIONS} iterations ===", flush=True)
    print(f"final MSE:            full={mf:.4e} +/- {sf:.4e}   incremental={mi:.4e} +/- {si:.4e}", flush=True)
    print(f"  |mean diff| / SEM(full+inc combined): "
          f"{abs(mf - mi) / np.sqrt(sf**2 + si**2):.2f} (overlapping CIs if < ~2)", flush=True)
    print(f"total selection time (hp_fit+acq_fun): full={tf:.1f}s +/- {stf:.1f}   "
          f"incremental={ti:.1f}s +/- {sti:.1f}   speedup={tf/ti:.2f}x", flush=True)
    print(f"wall clock per run:   full={wf:.1f}s +/- {swf:.1f}   incremental={wi:.1f}s +/- {swi:.1f}", flush=True)

    print("\nper-seed detail:", flush=True)
    for s in seeds:
        rf = next(r for r in full if r["seed"] == s)
        ri = next(r for r in inc if r["seed"] == s)
        print(f"  seed {s:2d}: mse_full={rf['mse_final']:.4e} mse_inc={ri['mse_final']:.4e}  "
              f"sel_time_full={rf['total_selection_time']:6.1f}s sel_time_inc={ri['total_selection_time']:6.1f}s "
              f"archive_full={rf['archive_size']} archive_inc={ri['archive_size']}", flush=True)


if __name__ == "__main__":
    main()
