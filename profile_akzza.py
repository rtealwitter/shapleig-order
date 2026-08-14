"""cProfile breakdown of ShapleyApplication.compute_AKZZA_new in isolation,
called directly on a real fitted surrogate (no candidate set, no
blackbox), to see whether its ~25ms/call cost (measured via
profile_acq_fun.py) is dominated by tensor-op dispatch count (many small
ops in the nested Python loop) or something else, before attempting any
rewrite of this shared, correctness-critical Shapley-computation code.
"""

import cProfile
import io
import pstats
import sys
import time

import torch

sys.path.insert(0, "src")

from xac.applications.applications import ShapiqShapleyApplication
from xac.surrogates import (AcceleratedFitConfig, ConstantNoiseConfig,
                             GPSurrogate, GPSurrogateConfig,
                             HammingKernelConfig)

torch.set_default_dtype(torch.float64)


def build_application_and_surrogate(p, t=50):
    cfg = GPSurrogateConfig(
        kernel_config=HammingKernelConfig(min_lengthscale=1e-6),
        noise_config=ConstantNoiseConfig(noise_level=1e-6),
        fit_config=AcceleratedFitConfig(),
    )
    baseline_config = torch.zeros(p, dtype=torch.float64)
    candidate_config = torch.ones(p, dtype=torch.float64)
    X = torch.randint(0, 2, (t, p), dtype=torch.float64)
    Y = torch.randn(t, 1, dtype=torch.float64)
    gp = GPSurrogate(
        X, Y, config=cfg, cat_dims=[], log_trafo_dims=[], bounds=None,
        shapley_configs=(baseline_config, candidate_config),
    )
    gp.fit()

    app = ShapiqShapleyApplication.__new__(ShapiqShapleyApplication)
    object.__setattr__(app, "amount_players", p)
    return app, gp


def main():
    for p in [10, 14, 16]:
        app, gp = build_application_and_surrogate(p)

        # warmup (first call may pay extra one-time costs)
        _ = app.compute_AKZZA_new(gp)

        reps = 20
        start = time.perf_counter()
        for _ in range(reps):
            _ = app.compute_AKZZA_new(gp)
        elapsed = (time.perf_counter() - start) / reps
        print(f"\n=== p={p}: {elapsed*1000:.2f} ms/call (mean of {reps}) ===", flush=True)

        pr = cProfile.Profile()
        pr.enable()
        for _ in range(reps):
            _ = app.compute_AKZZA_new(gp)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats(40)
        print(s.getvalue(), flush=True)

        # line-level profiling of compute_AKZZA_new itself, if available
        try:
            import line_profiler
            lp = line_profiler.LineProfiler()
            lp.add_function(ShapiqShapleyApplication.compute_AKZZA_new)
            wrapped = lp(app.compute_AKZZA_new)
            for _ in range(reps):
                wrapped(gp)
            s2 = io.StringIO()
            lp.print_stats(stream=s2)
            print(s2.getvalue(), flush=True)
        except ImportError:
            print("line_profiler not installed; skipping line-level breakdown", flush=True)


if __name__ == "__main__":
    main()
