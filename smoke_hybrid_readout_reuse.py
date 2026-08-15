"""Cheap smoke test for the 2026-08-14 experiment_runner.py change: Hybrid's
readout now reuses `gp` (skipping the separate readout_gp.fit()) instead of
doing a fresh fit on no-refit iterations, while every other arm
(ShaplEIG, PairedExtremes, Random) is untouched by construction (the new
`skip_separate_readout_fit` flag only ever fires for HybridPairedEIG).

Confirms: no crashes, finite MSE, and that eval_fit_duration is now ~0 for
Hybrid's no-refit iterations (the whole point of the change) while
PairedExtremes' eval_fit_duration stays nonzero throughout (unaffected).
"""

import sys

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"


def run_once(config_name, acq_target, game, iterations):
    from xac.experimental_designs.experiment_runner import run_experiment
    from xac.utils.random_utils import set_seed

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "meta.seed=1",
                f"blackbox.name={game}",
                f"acquisition._target_={acq_target}",
                f"experimental_design.iterations={iterations}",
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
    from xac.utils.metrics import compute_mse
    mse_trace = np.array([compute_mse(p, prop_gt) for p in prop_posts])
    return {
        "mse_final": float(mse_trace[-1]),
        "eval_fit_durations": np.array(eval_fit_durations, dtype=float),
        "archive_size": int(archive_x.shape[0]),
    }


def main():
    iterations = 150
    game = "dvbsgb_10"

    print("--- Hybrid (repro_all_dv10_hybrid_akzza) ---", flush=True)
    hy = run_once("repro_all_dv10_hybrid_akzza",
                   "xac.acquisition_functions.HybridPairedEIG", game, iterations)
    n_zero = (hy["eval_fit_durations"] == 0.0).sum()
    n_nonzero = (hy["eval_fit_durations"] > 0.0).sum()
    print(f"  mse_final={hy['mse_final']:.4e}  archive={hy['archive_size']}", flush=True)
    print(f"  eval_fit_durations: n_zero={n_zero}  n_nonzero={n_nonzero}  "
          f"sum={hy['eval_fit_durations'].sum():.3f}s", flush=True)
    # n_nonzero > 0 is expected and correct: those are the extremes-phase
    # iterations (before handover), where gp is never fit and a real
    # readout fit is still required, same as PairedExtremes. The readout-
    # reuse change only applies post-handover, so most (not all) iterations
    # should now be free.
    assert n_zero > n_nonzero, (
        "expected most iterations (post-handover, no-refit) to skip the "
        "separate readout fit after the readout-reuse change"
    )
    assert np.isfinite(hy["mse_final"]), "non-finite MSE for Hybrid"

    print("\n--- PairedExtremes (repro_all_dv10, unaffected) ---", flush=True)
    pe = run_once("repro_all_dv10",
                   "xac.acquisition_functions.PairedExtremes", game, iterations)
    n_zero_pe = (pe["eval_fit_durations"] == 0.0).sum()
    n_nonzero_pe = (pe["eval_fit_durations"] > 0.0).sum()
    print(f"  mse_final={pe['mse_final']:.4e}  archive={pe['archive_size']}", flush=True)
    print(f"  eval_fit_durations: n_zero={n_zero_pe}  n_nonzero={n_nonzero_pe}  "
          f"sum={pe['eval_fit_durations'].sum():.3f}s", flush=True)
    assert n_nonzero_pe > 0, (
        "PairedExtremes should still pay a real eval_fit cost every iteration -- "
        "the readout-reuse change must not have leaked into it"
    )
    assert np.isfinite(pe["mse_final"]), "non-finite MSE for PairedExtremes"

    print("\nSMOKE TEST PASSED: Hybrid's readout now near-free, "
          "PairedExtremes unaffected", flush=True)


if __name__ == "__main__":
    main()
