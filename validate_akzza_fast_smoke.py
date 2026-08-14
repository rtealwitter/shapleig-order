"""Cheap correctness smoke test for compute_AKZZA_fast (applications.py):
runs a real HybridPairedEIG smoke sweep end to end with
ShapleyApplication.compute_AKZZA_new monkeypatched to compute_AKZZA_fast,
confirming no crashes/shape errors/NaNs on a real archive before spending
the ~1hr multi-seed MSE comparison.

Per the lesson already learned validating the incremental-Cholesky change
(see validate_incremental_cholesky_mse.py's docstring): bit-identical
archive trajectories are the WRONG bar here. compute_AKZZA_fast reorders
floating-point summation (global prefix/suffix instead of per-player
rebuild), so it can differ from compute_AKZZA_new by float64 noise
(confirmed: worst relative diff 2.4e-14 across p=2..32 in
validate_akzza_fast.py's direct allclose battery) -- and HybridPairedEIG's
greedy selection is sensitive enough that any nonzero difference will
eventually flip a near-tied EIG comparison and branch the trajectory. This
script therefore does NOT assert archive_x matches between the two runs;
it only asserts both runs complete cleanly with finite, sane outputs. The
actual correctness gate is the multi-seed MSE comparison in
validate_akzza_fast_mse.py.
"""

import sys

import hydra
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"


def run_once(use_fast: bool):
    from xac.applications.applications import ShapleyApplication
    from xac.experimental_designs.experiment_runner import run_experiment

    original = ShapleyApplication.compute_AKZZA_new
    try:
        if use_fast:
            ShapleyApplication.compute_AKZZA_new = ShapleyApplication.compute_AKZZA_fast

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            cfg = compose(
                config_name="repro_all_smoke_dv10_fastfit",
                overrides=[
                    "meta.seed=1",
                    "blackbox.name=dvbsgb_10",
                    "acquisition._target_=xac.acquisition_functions.HybridPairedEIG",
                    "experimental_design.iterations=450",
                ],
            )

        from xac.utils.random_utils import set_seed
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
        prop_post_means = torch.stack([p.mean for p in prop_posts])
        from xac.utils.metrics import compute_mse
        import numpy as np
        mse_trace = np.array([compute_mse(p, prop_gt) for p in prop_posts])
        return archive_x, archive_y, prop_post_means, mse_trace
    finally:
        ShapleyApplication.compute_AKZZA_new = original


def main():
    print("Run 1: compute_AKZZA_new (reference)...", flush=True)
    ax_ref, ay_ref, pm_ref, mse_ref = run_once(use_fast=False)

    print("Run 2: compute_AKZZA_fast (candidate)...", flush=True)
    ax_fast, ay_fast, pm_fast, mse_fast = run_once(use_fast=True)

    print(f"\narchive_x shapes: ref={tuple(ax_ref.shape)} fast={tuple(ax_fast.shape)}", flush=True)

    ok = True
    if ax_ref.shape != ax_fast.shape:
        print("FAILED: archive sizes differ", flush=True)
        ok = False

    for name, arr in [("archive_y (ref)", ay_ref), ("archive_y (fast)", ay_fast),
                       ("prop_post_means (ref)", pm_ref), ("prop_post_means (fast)", pm_fast),
                       ("mse_trace (ref)", torch.from_numpy(mse_ref)),
                       ("mse_trace (fast)", torch.from_numpy(mse_fast))]:
        if not torch.isfinite(arr).all():
            print(f"FAILED: {name} contains non-finite values", flush=True)
            ok = False

    n_match = (ax_ref == ax_fast).all(dim=-1).sum().item() if ax_ref.shape == ax_fast.shape else 0
    total = ax_ref.shape[0]
    print(f"archive points identical: {n_match}/{total} "
          f"(divergence expected once float64 noise flips a near-tied EIG comparison -- not a failure signal)",
          flush=True)
    print(f"mse_final: ref={mse_ref[-1]:.4e}  fast={mse_fast[-1]:.4e}  "
          f"ratio={mse_fast[-1]/mse_ref[-1]:.3f}", flush=True)

    if not ok:
        sys.exit(1)
    print("\nPASSED: both runs completed cleanly with finite outputs", flush=True)


if __name__ == "__main__":
    main()
