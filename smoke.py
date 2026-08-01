# Fast end-to-end smoke test: gp_core self-tests, then the full benchmark path
# on a synthetic d=8 game with 2 seeds and a short grid.
import numpy as np
import benchmark as B
from gp_core import self_test, fixed_schedule, membership, shapley_A

self_test()

d = 8
N = 1 << d
rng = np.random.default_rng(0)
Mem = membership(d)
w = rng.normal(size=d)
f_gt = Mem @ w + 0.3 * np.sin(Mem @ rng.normal(size=d))
tab = {"d": d,
       "values_seeds": np.array([f_gt + 0.01 * np.random.default_rng(s).normal(size=N)
                                 for s in range(2)]),
       "values_gt": f_gt,
       "phi_star": shapley_A(d, Mem) @ f_gt}

B.N_SEEDS = 2
B.GRIDS[8] = [6, 8, 12, 16, 24]
sched = fixed_schedule(d, 24)
sizes = [bin(m).count("1") for m in sched]
print("schedule sizes:", sizes)
assert sched[0] == 0 and sched[1] == N - 1, "schedule starts empty/full"
res = B.run_game("smoke", tab, sched)
for meth in ("uniform", "shapleig", "paired"):
    errs = [res[meth][str(s)]["err"][-1] for s in range(2)]
    print(f"{meth}: final errs {np.round(errs, 4)} times "
          f"{np.round([res[meth][str(s)]['time'][-1] for s in range(2)], 3)}")
    assert all(np.isfinite(errs)), meth
print("noise floor:", np.round(res["noise_floor"], 4))
print("SMOKE OK")
