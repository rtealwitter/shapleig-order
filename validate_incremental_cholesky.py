"""Correctness check for ShapleyApplication._cholesky_maybe_incremental
(applications.py): runs the SAME real HybridPairedEIG smoke sweep twice
with the SAME seed -- once with the incremental Cholesky update forced
off (always full recompute, the pre-change behavior) and once with it on
(the new default for AcceleratedFitConfig) -- and compares the actual
archive selections and MSE trace end to end. This is the bar that
matters: not just "the intermediate L matrix is numerically close" but
"the acquisition never picks a different point and the reported metrics
don't move."
"""

import sys

import hydra
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"


def run_once(force_full: bool):
    from xac.applications.applications import ShapleyApplication
    from xac.experimental_designs.experiment_runner import run_experiment

    if force_full:
        original = ShapleyApplication._cholesky_maybe_incremental
        from linear_operator.utils.cholesky import psd_safe_cholesky

        def always_full(self, surrogate, C, is_no_refit_step):
            L = psd_safe_cholesky(C.to_dense())
            object.__setattr__(self, "_L_cache", L)
            return L

        ShapleyApplication._cholesky_maybe_incremental = always_full
    else:
        original = None

    try:
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
        set_seed(cfg.meta.seed)  # cli.py's main() does this; the two run_once()
        # calls share one process, so without re-seeding here Run 2 inherits
        # whatever random state Run 1 left behind instead of starting fresh.

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
        return archive_x, archive_y, prop_post_means
    finally:
        if force_full:
            ShapleyApplication._cholesky_maybe_incremental = original


def main():
    import os
    sanity = os.environ.get("SANITY_SELF_COMPARE") == "1"

    print("Run 1: incremental Cholesky FORCED OFF (always full recompute)...", flush=True)
    ax_full, ay_full, pm_full = run_once(force_full=True)

    if sanity:
        print("Run 2: SAME (forced off again) -- sanity check for baseline determinism...", flush=True)
        ax_inc, ay_inc, pm_inc = run_once(force_full=True)
    else:
        print("Run 2: incremental Cholesky ON (new default for AcceleratedFitConfig)...", flush=True)
        ax_inc, ay_inc, pm_inc = run_once(force_full=False)

    print(f"\narchive_x shapes: full={tuple(ax_full.shape)} incremental={tuple(ax_inc.shape)}", flush=True)

    if ax_full.shape != ax_inc.shape:
        print("FAILED: archive sizes differ", flush=True)
        sys.exit(1)

    x_match = (ax_full == ax_inc).all().item()
    y_diff = (ay_full - ay_inc).abs().max().item()
    print(f"archive_x identical: {x_match}", flush=True)
    print(f"archive_y max abs diff: {y_diff:.3e}", flush=True)

    pm_reldiff = ((pm_full - pm_inc).abs() / (pm_full.abs() + 1e-30)).max().item()
    print(f"prop_post_means max relative diff: {pm_reldiff:.3e}", flush=True)

    if not x_match:
        n_diff = (ax_full != ax_inc).any(dim=-1).sum().item()
        print(f"FAILED: {n_diff}/{ax_full.shape[0]} selected points differ between the two runs", flush=True)
        sys.exit(1)
    if y_diff > 1e-8:
        print(f"FAILED: archive_y differs by more than float64 noise ({y_diff:.3e})", flush=True)
        sys.exit(1)

    print("\nPASSED: identical point selections and archive values with incremental Cholesky on vs. off", flush=True)


if __name__ == "__main__":
    main()
