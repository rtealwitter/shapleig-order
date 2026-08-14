"""Instrumented before/after on a real resnet_14 HybridPairedEIG smoke run,
comparing exact vs AcceleratedFitConfig, to find why the existing
sweep data (sweep_roots_v2.txt) shows the "fast" arm with a HIGHER
acq_fun_duration total than exact on resnet_14 (133s vs 103s over 512
iterations) and vit_9 -- i.e. the regression is in the main ACQUISITION
path, not just hyperparameter fitting, which is surprising since
AcceleratedFitConfig is supposed to only change how (lengthscale,
outputscale) get fit. Candidate mechanism: _cholesky_maybe_incremental /
_solve_against_cached_cholesky (both gated to AcceleratedFitConfig) may be
falling back to full recomputation often on resnet_14/vit_9's specific
archive-growth pattern, paying both the failed-incremental-attempt AND the
full-recompute cost on those calls.
"""

import sys
import time
from collections import defaultdict

import hydra
from hydra import compose, initialize_config_dir

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"


def timed(store, name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            store[name].append(time.perf_counter() - start)
            return out
        return wrapper
    return decorator


CONFIG_FAMILY = {
    "resnet_14": "p1416", "vit_16": "p1416",
    "vit_9": "vit9",
    "dvbsgb_10": "dv10", "dvbsrf_10": "dv10", "dvchgb_10": "dv10",
}


def run_once(game: str, fast: bool, iterations: int):
    from xac.applications.applications import ShapleyApplication
    from xac.surrogates.gp_surrogate import GPSurrogate
    from xac.surrogates.fast_fit import fit_accelerated as orig_fit_accelerated
    import xac.surrogates.fast_fit as fast_fit_mod
    import xac.experimental_designs.experiment_runner as experiment_runner_mod
    from xac.utils.random_utils import set_seed

    store = defaultdict(list)
    fallback_counts = defaultdict(int)

    names = [
        "compute_AEA_new", "compute_ASigmaW_new", "compute_A_KZW_new",
        "_cholesky_maybe_incremental", "_solve_against_cached_cholesky",
        "compute_AKZZA_fast", "get_K_XX_noisy",
    ]
    originals = {n: getattr(ShapleyApplication, n) for n in names}
    for n in names:
        setattr(ShapleyApplication, n, timed(store, n)(originals[n]))

    orig_gp_fit = GPSurrogate.fit
    GPSurrogate.fit = timed(store, "GPSurrogate.fit")(orig_gp_fit)

    fast_fit_mod.fit_accelerated = timed(store, "fit_accelerated")(orig_fit_accelerated)
    # gp_surrogate.py imports fit_accelerated locally inside fit(), so no
    # separate patch needed there -- the local import picks up the module
    # attribute we just replaced.

    orig_readout = experiment_runner_mod._readout_property_posterior
    experiment_runner_mod._readout_property_posterior = timed(
        store, "_readout_property_posterior"
    )(orig_readout)

    # Track how often the incremental path actually fires vs falls back to
    # full factorization, by wrapping psd_safe_cholesky calls made from
    # inside _cholesky_maybe_incremental's fallback branch.
    from linear_operator.utils.cholesky import psd_safe_cholesky as orig_psd_safe_cholesky
    import xac.applications.applications as apps_mod

    def counting_psd_safe_cholesky(*args, **kwargs):
        fallback_counts["full_cholesky_calls"] += 1
        return orig_psd_safe_cholesky(*args, **kwargs)

    apps_mod.psd_safe_cholesky = counting_psd_safe_cholesky

    try:
        overrides = [
            "meta.seed=1",
            f"blackbox.name={game}",
            "acquisition._target_=xac.acquisition_functions.HybridPairedEIG",
            f"experimental_design.iterations={iterations}",
        ]
        family = CONFIG_FAMILY[game]
        config_name = f"repro_all_{family}_fastfit" if fast else f"repro_all_{family}"

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            cfg = compose(config_name=config_name, overrides=overrides)

        set_seed(cfg.meta.seed)

        from xac.experimental_designs.experiment_runner import run_experiment

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
        run_experiment(
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
        for n in names:
            setattr(ShapleyApplication, n, originals[n])
        apps_mod.psd_safe_cholesky = orig_psd_safe_cholesky
        GPSurrogate.fit = orig_gp_fit
        fast_fit_mod.fit_accelerated = orig_fit_accelerated
        experiment_runner_mod._readout_property_posterior = orig_readout

    return elapsed, store, fallback_counts


def report(label, elapsed, store, fallback_counts):
    print(f"\n=== {label}: wall={elapsed:.1f}s ===", flush=True)
    for name, times in sorted(store.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {name:35s} n={len(times):5d}  sum={sum(times):8.2f}s  "
              f"mean={sum(times)/len(times)*1000:8.2f}ms", flush=True)
    print(f"  full (fallback) Cholesky factorizations: {fallback_counts['full_cholesky_calls']}", flush=True)


def main():
    iterations = 150
    for game in ["resnet_14", "vit_9", "dvbsgb_10"]:
        print(f"\n\n########## {game} ##########", flush=True)
        elapsed_ex, store_ex, fb_ex = run_once(game, fast=False, iterations=iterations)
        report(f"{game} EXACT", elapsed_ex, store_ex, fb_ex)

        elapsed_fa, store_fa, fb_fa = run_once(game, fast=True, iterations=iterations)
        report(f"{game} FAST", elapsed_fa, store_fa, fb_fa)

        print(f"\n--- {game} summary: wall exact={elapsed_ex:.1f}s fast={elapsed_fa:.1f}s "
              f"(speedup={elapsed_ex/elapsed_fa:.2f}x)", flush=True)
        for key in ["compute_AEA_new", "GPSurrogate.fit", "fit_accelerated",
                    "_readout_property_posterior"]:
            v_ex = sum(store_ex.get(key, []))
            v_fa = sum(store_fa.get(key, []))
            print(f"    {key:28s} total: exact={v_ex:7.2f}s fast={v_fa:7.2f}s  "
                  f"(n_ex={len(store_ex.get(key, []))} n_fa={len(store_fa.get(key, []))})",
                  flush=True)


if __name__ == "__main__":
    main()
