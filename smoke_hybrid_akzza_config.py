"""Cheap smoke test for the new repro_all_*_hybrid_akzza.yaml configs
before launching the multi-hour production sweeps: confirms
application.use_akzza_fast is wired through hydra instantiation correctly
(compute_AEA_new actually dispatches to compute_AKZZA_fast, not silently
falling back to compute_AKZZA_new), and that a short real run completes
without crashing, for each of the three config families.
"""

import sys

import hydra
from hydra import compose, initialize_config_dir

sys.path.insert(0, "src")

CONFIG_DIR = "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo/src/xac/experiments/conf"


def run_once(config_name: str, game: str, iterations: int = 30):
    from xac.applications.applications import ShapleyApplication
    from xac.experimental_designs.experiment_runner import run_experiment
    from xac.utils.random_utils import set_seed

    calls = {"fast": 0, "new": 0}
    orig_fast = ShapleyApplication.compute_AKZZA_fast
    orig_new = ShapleyApplication.compute_AKZZA_new

    def counting_fast(self, surrogate):
        calls["fast"] += 1
        return orig_fast(self, surrogate)

    def counting_new(self, surrogate):
        calls["new"] += 1
        return orig_new(self, surrogate)

    ShapleyApplication.compute_AKZZA_fast = counting_fast
    ShapleyApplication.compute_AKZZA_new = counting_new

    try:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            cfg = compose(
                config_name=config_name,
                overrides=[
                    "meta.seed=1",
                    f"blackbox.name={game}",
                    f"experimental_design.iterations={iterations}",
                ],
            )

        set_seed(cfg.meta.seed)

        application = hydra.utils.instantiate(cfg.application)
        assert application.use_akzza_fast is True, (
            f"use_akzza_fast not wired through for {config_name}: "
            f"got {application.use_akzza_fast}"
        )

        surrogate_cfg = hydra.utils.instantiate(cfg.surrogate)
        acquisition_fn = hydra.utils.instantiate(cfg.acquisition)
        acquisition_optimizer = hydra.utils.instantiate(cfg.acquisition_optimizer)
        blackbox_fn = hydra.utils.instantiate(cfg.blackbox)
        ed_cfg = hydra.utils.instantiate(cfg.experimental_design)
        meta_cfg = hydra.utils.instantiate(cfg.meta)

        import tempfile
        run_dir = tempfile.mkdtemp()

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
    finally:
        ShapleyApplication.compute_AKZZA_fast = orig_fast
        ShapleyApplication.compute_AKZZA_new = orig_new

    return calls


def main():
    targets = [
        ("repro_all_dv10_hybrid_akzza", "dvbsgb_10"),
        ("repro_all_vit9_hybrid_akzza", "vit_9"),
        ("repro_all_p1416_hybrid_akzza", "resnet_14"),
    ]
    failed = False
    for config_name, game in targets:
        print(f"--- {config_name} ({game}) ---", flush=True)
        try:
            calls = run_once(config_name, game)
            ok = calls["fast"] > 0 and calls["new"] == 0
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: compute_AKZZA_fast calls={calls['fast']}  "
                  f"compute_AKZZA_new calls={calls['new']}", flush=True)
            failed = failed or not ok
        except Exception:
            import traceback
            traceback.print_exc()
            print("  FAIL: exception during run", flush=True)
            failed = True

    if failed:
        print("\nSMOKE TEST FAILED", flush=True)
        sys.exit(1)
    print("\nSMOKE TEST PASSED: all three configs wire use_akzza_fast "
          "correctly and complete a short run cleanly", flush=True)


if __name__ == "__main__":
    main()
