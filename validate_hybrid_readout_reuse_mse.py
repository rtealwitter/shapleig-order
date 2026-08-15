"""Multi-seed MSE + timing comparison for the 2026-08-14 readout-reuse
change to Hybrid (experiment_runner.py): runs N_SEEDS fresh resnet_14
HybridPairedEIG + AKZZA runs under the NEW code (readout reuses gp
post-handover) and compares against the OLD-code production data already
collected this session (multirun root in OLD_ROOT, from before this
change), which used a separate readout fit on every no-refit iteration.

Bit-identical archives is the wrong bar (see HANDOFF_AKZZA.md's earlier
lesson) -- this checks the bar that matters: aggregate MSE and timing
across seeds.
"""

import glob
import json
import os
import sys
import time

import hydra
import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir
from joblib import Parallel, delayed

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"
GAME = "resnet_14"
N_SEEDS = 8
ITERATIONS = 512
N_JOBS = 16

OLD_ROOT = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/multirun/2026-08-14/08-39-01-268943"


def load_old(game, seeds):
    out = {}
    for mpath in glob.glob(os.path.join(OLD_ROOT, "**", "metrics.json"), recursive=True):
        run_dir = os.path.dirname(mpath)
        cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        if cfg["blackbox"]["name"] != game:
            continue
        seed = cfg["meta"]["seed"]
        if seed not in seeds:
            continue
        met = json.load(open(mpath))
        mse = np.array(met["mse"], dtype=float)
        hp = np.nansum(met.get("hp_fit_duration", []))
        acq = np.nansum(met.get("acq_fun_duration", []))
        ef = np.nansum(met.get("eval_fit_duration", []))
        pp = np.nansum(met.get("prop_post_duration", []))
        out[seed] = {
            "mse_final": float(mse[-1]),
            "hp_fit": float(hp), "acq_fun": float(acq),
            "eval_fit": float(ef), "prop_post": float(pp),
            "archive_size": int(met.get("archive_sizes", [0])[-1]) if met.get("archive_sizes") else None,
        }
    return out


def run_new(seed):
    from xac.utils.random_utils import set_seed

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="repro_all_p1416_hybrid_akzza",
            overrides=[
                f"meta.seed={seed}",
                f"blackbox.name={GAME}",
                f"experimental_design.iterations={ITERATIONS}",
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

    from xac.utils.metrics import compute_mse
    mse_trace = np.array([compute_mse(p, prop_gt) for p in prop_posts])
    return {
        "seed": seed,
        "elapsed": elapsed,
        "mse_final": float(mse_trace[-1]),
        "hp_fit": float(np.nansum(hp_fit_durations)),
        "acq_fun": float(np.nansum(acq_fun_durations)),
        "eval_fit": float(np.nansum(eval_fit_durations)),
        "prop_post": float(np.nansum(prop_post_durations)),
        "archive_size": int(archive_x.shape[0]),
    }


def main():
    seeds = list(range(1, N_SEEDS + 1))

    print(f"Loading OLD-code data for {N_SEEDS} seeds from {OLD_ROOT} ...", flush=True)
    old = load_old(GAME, set(seeds))
    missing = [s for s in seeds if s not in old]
    if missing:
        print(f"WARNING: missing old data for seeds {missing}", flush=True)

    print(f"Running {N_SEEDS} NEW-code seeds on {GAME}, {ITERATIONS} iterations, "
          f"n_jobs={N_JOBS} ...", flush=True)
    new_results = Parallel(n_jobs=N_JOBS)(delayed(run_new)(s) for s in seeds)
    new = {r["seed"]: r for r in new_results}

    mse_old = np.array([old[s]["mse_final"] for s in seeds if s in old])
    mse_new = np.array([new[s]["mse_final"] for s in seeds if s in old])
    ef_old = np.array([old[s]["eval_fit"] for s in seeds if s in old])
    ef_new = np.array([new[s]["eval_fit"] for s in seeds if s in old])
    total_old = np.array([old[s]["hp_fit"] + old[s]["acq_fun"] + old[s]["eval_fit"]
                           for s in seeds if s in old])
    total_new = np.array([new[s]["hp_fit"] + new[s]["acq_fun"] + new[s]["eval_fit"]
                           for s in seeds if s in old])

    def mean_sem(a):
        return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

    mo, so = mean_sem(mse_old)
    mn, sn = mean_sem(mse_new)
    eo, seo = mean_sem(ef_old)
    en, sen = mean_sem(ef_new)
    to, sto = mean_sem(total_old)
    tn, stn = mean_sem(total_new)

    print(f"\n=== {GAME} HybridPairedEIG+AKZZA, {N_SEEDS} seeds, {ITERATIONS} iterations ===", flush=True)
    print(f"final MSE:        old={mo:.4e} +/- {so:.4e}   new={mn:.4e} +/- {sn:.4e}", flush=True)
    print(f"  |mean diff| / SEM(old+new combined): "
          f"{abs(mo - mn) / np.sqrt(so**2 + sn**2):.2f} (overlapping CIs if < ~2)", flush=True)
    print(f"eval_fit (readout) time: old={eo:.1f}s +/- {seo:.1f}   new={en:.1f}s +/- {sen:.1f}   "
          f"speedup={eo/en if en > 0 else float('inf'):.1f}x", flush=True)
    print(f"total (hp_fit+acq_fun+eval_fit): old={to:.1f}s +/- {sto:.1f}   new={tn:.1f}s +/- {stn:.1f}   "
          f"speedup={to/tn:.2f}x", flush=True)

    print("\nper-seed detail:", flush=True)
    for s in seeds:
        if s not in old:
            continue
        print(f"  seed {s:2d}: mse_old={old[s]['mse_final']:.4e} mse_new={new[s]['mse_final']:.4e}  "
              f"eval_fit_old={old[s]['eval_fit']:6.1f}s eval_fit_new={new[s]['eval_fit']:6.1f}s  "
              f"archive_old={old[s]['archive_size']} archive_new={new[s]['archive_size']}", flush=True)


if __name__ == "__main__":
    main()
