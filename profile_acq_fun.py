"""Direct timing breakdown of EIGFunctionProperty.__call__'s sub-steps on a
real HybridPairedEIG smoke run: two targeted fixes (incremental Cholesky
in compute_AEA_new, cached-factor reuse in compute_ASigmaW_new) both
measured safe but delivered ~1.02x real speedup, far less than the
Cholesky-factorization theory predicted, so this instruments the actual
call graph to find what's really dominating acq_fun_duration instead of
guessing at a third fix.
"""

import sys
import time
from collections import defaultdict

import hydra
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"

_TIMES = defaultdict(list)


def timed(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            _TIMES[name].append(time.perf_counter() - start)
            return out
        return wrapper
    return decorator


def instrument():
    from xac.applications.applications import ShapleyApplication
    from xac.acquisition_functions.acquisition_functions import EIGFunctionProperty

    for name in [
        "compute_AEA_new", "compute_ASigmaW_new", "compute_A_KZW_new",
        "_cholesky_maybe_incremental", "_solve_against_cached_cholesky",
        "compute_AKZZA_new", "get_K_XX_noisy",
    ]:
        orig = getattr(ShapleyApplication, name)
        setattr(ShapleyApplication, name, timed(name)(orig))

    orig_call = EIGFunctionProperty.__call__
    EIGFunctionProperty.__call__ = timed("EIGFunctionProperty.__call__")(orig_call)


def main():
    instrument()

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="repro_all_p1416_fastfit",
            overrides=[
                "meta.seed=1",
                "blackbox.name=resnet_14",
                "acquisition._target_=xac.acquisition_functions.HybridPairedEIG",
                "experimental_design.iterations=150",
            ],
        )

    from xac.utils.random_utils import set_seed
    from xac.experimental_designs.experiment_runner import run_experiment

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

    print(f"Running resnet_14 HybridPairedEIG, {ed_cfg.iterations} iterations, instrumented...", flush=True)
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

    print("\n=== timing breakdown (sum over all calls) ===", flush=True)
    for name, times in sorted(_TIMES.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {name:40s} n={len(times):5d}  sum={sum(times):8.2f}s  "
              f"mean={sum(times)/len(times)*1000:8.2f}ms", flush=True)


if __name__ == "__main__":
    main()
