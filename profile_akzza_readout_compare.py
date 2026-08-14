"""Direct before/after timing on a real HybridPairedEIG smoke run,
comparing compute_AKZZA_new (reference) vs compute_AKZZA_fast (now the
live default in compute_AEA_new) -- specifically including
_readout_property_posterior, since profile_acq_fun.py's own finding was
that most compute_AKZZA_new calls come from the readout path (a fresh
surrogate fit every iteration, cache-miss every time), not the main
acquisition path's no-refit-cached AKA. The multi-seed MSE validation's
"total_selection_time" metric (hp_fit_durations + acq_fun_durations)
doesn't include readout time, so it under-counted this speedup -- this
script measures the readout path directly instead.
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


def run_once(use_fast: bool, iterations: int):
    from xac.applications.applications import ShapleyApplication
    import xac.experimental_designs.experiment_runner as experiment_runner_mod
    from xac.acquisition_functions.acquisition_functions import EIGFunctionProperty
    from xac.utils.random_utils import set_seed

    store = defaultdict(list)

    orig_akzza_new = ShapleyApplication.compute_AKZZA_new
    orig_akzza_fast = ShapleyApplication.compute_AKZZA_fast
    orig_readout = experiment_runner_mod._readout_property_posterior
    orig_call = EIGFunctionProperty.__call__

    ShapleyApplication.compute_AKZZA_new = timed(store, "compute_AKZZA_new")(orig_akzza_new)
    ShapleyApplication.compute_AKZZA_fast = timed(store, "compute_AKZZA_fast")(orig_akzza_fast)
    experiment_runner_mod._readout_property_posterior = timed(store, "_readout_property_posterior")(orig_readout)
    EIGFunctionProperty.__call__ = timed(store, "EIGFunctionProperty.__call__")(orig_call)

    if not use_fast:
        # Reference condition: force compute_AEA_new's internal call back to
        # compute_AKZZA_new by swapping compute_AKZZA_fast to alias it.
        ShapleyApplication.compute_AKZZA_fast = ShapleyApplication.compute_AKZZA_new

    try:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            cfg = compose(
                config_name="repro_all_p1416_fastfit",
                overrides=[
                    "meta.seed=1",
                    "blackbox.name=resnet_14",
                    "acquisition._target_=xac.acquisition_functions.HybridPairedEIG",
                    f"experimental_design.iterations={iterations}",
                ],
            )

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
        ShapleyApplication.compute_AKZZA_new = orig_akzza_new
        ShapleyApplication.compute_AKZZA_fast = orig_akzza_fast
        experiment_runner_mod._readout_property_posterior = orig_readout
        EIGFunctionProperty.__call__ = orig_call

    return elapsed, store


def report(label, elapsed, store):
    print(f"\n=== {label}: wall={elapsed:.1f}s ===", flush=True)
    for name, times in sorted(store.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {name:35s} n={len(times):5d}  sum={sum(times):8.2f}s  "
              f"mean={sum(times)/len(times)*1000:8.2f}ms", flush=True)


def main():
    iterations = 150
    print(f"Running resnet_14 HybridPairedEIG, {iterations} iterations, "
          f"reference (compute_AKZZA_new) vs fast (compute_AKZZA_fast)...", flush=True)

    elapsed_ref, store_ref = run_once(use_fast=False, iterations=iterations)
    report("REFERENCE (compute_AKZZA_new)", elapsed_ref, store_ref)

    elapsed_fast, store_fast = run_once(use_fast=True, iterations=iterations)
    report("FAST (compute_AKZZA_fast)", elapsed_fast, store_fast)

    print(f"\n=== summary ===", flush=True)
    print(f"wall clock:  ref={elapsed_ref:.1f}s  fast={elapsed_fast:.1f}s  "
          f"speedup={elapsed_ref/elapsed_fast:.2f}x", flush=True)
    readout_ref = sum(store_ref.get("_readout_property_posterior", []))
    readout_fast = sum(store_fast.get("_readout_property_posterior", []))
    print(f"readout sum: ref={readout_ref:.1f}s  fast={readout_fast:.1f}s  "
          f"speedup={readout_ref/readout_fast:.2f}x" if readout_fast else "", flush=True)


if __name__ == "__main__":
    main()
