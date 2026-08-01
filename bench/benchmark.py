# Three baselines on the precomputed shapiq games, 8 seeds each:
#   A "uniform"  — coalitions sampled uniformly by size, one GP fit at each m
#   B "shapleig" — EIG greedy selection with lengthscale refit every round
#   C "paired"   — the fixed-kernel greedy schedule (extremes in complement
#                  pairs), one GP fit at each m; player order permuted per seed
# Error = relative L2 distance to the exact Shapley vector of the ground-truth
# table; time = method compute (fit / selection / estimate), excluding game
# evaluations (those are the budget axis m) and excluding C's offline schedule.
# Resumable per game: writes results_<game>.json.
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "greedy"))
from gp_core import (membership, shapley_A, AH_matrix, fit_gp, shapley_estimate,
                     eig_scores, fixed_schedule, self_test)

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(HERE, "games")
N_SEEDS = 8
GRIDS = {8: [6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
         12: [6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         14: [6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]}


def rel_err(phi_hat, phi_star):
    return float(np.linalg.norm(phi_hat - phi_star) / np.linalg.norm(phi_star))


def masks_to_bool(masks, d):
    return ((np.asarray(masks)[:, None] >> np.arange(d)) & 1).astype(bool)


def fit_and_estimate(E_bool, y, Mem, AHm):
    t0 = time.perf_counter()
    betas, sig2, mu, sc, _, _ = fit_gp(E_bool, y)
    phi = shapley_estimate(betas, sig2, Mem, AHm, E_bool, y, mu, sc)
    return phi, time.perf_counter() - t0, betas


def uniform_design(d, m, rng):
    chosen = []
    seen = set()
    while len(chosen) < m:
        s = int(rng.integers(0, d + 1))
        T = frozenset(rng.choice(d, size=s, replace=False).tolist())
        mask = sum(1 << j for j in T)
        if mask not in seen:
            seen.add(mask)
            chosen.append(mask)
    return np.array(chosen)


def permute_masks(masks, perm):
    d = len(perm)
    out = np.zeros_like(masks)
    for j in range(d):
        out |= (((masks >> j) & 1) << int(perm[j]))
    return out


def run_game(name, tab, schedule):
    d = int(tab["d"])
    Mem = membership(d)
    AHm = AH_matrix(d, Mem)
    phi_star = tab["phi_star"]
    grid = [m for m in GRIDS[d] if m <= (1 << d)]
    res = {meth: {str(s): {"m": [], "err": [], "time": []} for s in range(N_SEEDS)}
           for meth in ("uniform", "shapleig", "paired")}
    res["noise_floor"] = [rel_err(shapley_A(d, Mem) @ tab["values_seeds"][s], phi_star)
                          for s in range(N_SEEDS)]
    for s in range(N_SEEDS):
        values = tab["values_seeds"][s]
        rng = np.random.default_rng(1000 + s)

        # A: uniform by size, fit once per checkpoint (design nested over m)
        design = uniform_design(d, grid[-1], rng)
        for m in grid:
            E = masks_to_bool(design[:m], d)
            phi, dt, _ = fit_and_estimate(E, values[design[:m]], Mem, AHm)
            r = res["uniform"][str(s)]
            r["m"].append(m); r["err"].append(rel_err(phi, phi_star)); r["time"].append(dt)

        # C: fixed-kernel paired schedule, player order permuted per seed
        perm = rng.permutation(d)
        sched = permute_masks(schedule, perm)
        for m in grid:
            E = masks_to_bool(sched[:m], d)
            phi, dt, betas_c = fit_and_estimate(E, values[sched[:m]], Mem, AHm)
            r = res["paired"][str(s)]
            r["m"].append(m); r["err"].append(rel_err(phi, phi_star)); r["time"].append(dt)
        res["paired"][str(s)]["betas_final"] = betas_c.tolist()

        # B: ShaplEIG — EIG greedy + refit every round
        init = [0, (1 << d) - 1]
        while len(init) < 4:
            c = int(rng.integers(1, (1 << d) - 1))
            if c not in init:
                init.append(c)
        masks = list(init)
        t_cum = 0.0
        z_warm, w_warm = None, None
        betas = np.full(d, 0.5)
        sig2 = 1e-4
        gi = 0
        for m_now in range(len(init), grid[-1] + 1):
            E = masks_to_bool(masks, d)
            y = values[np.array(masks)]
            t0 = time.perf_counter()
            if m_now >= 6:
                betas, sig2, mu, sc, z_warm, w_warm = fit_gp(
                    E, y, z0=z_warm, w0=w_warm, maxiter=30)
            else:
                mu, sc = y.mean(), max(y.std(), 1e-12)
            while gi < len(grid) and grid[gi] == m_now:
                phi = shapley_estimate(betas, sig2, Mem, AHm, E, y, mu, sc)
                t_cum += time.perf_counter() - t0
                r = res["shapleig"][str(s)]
                r["m"].append(m_now); r["err"].append(rel_err(phi, phi_star))
                r["time"].append(t_cum)
                t0 = time.perf_counter()
                gi += 1
            if m_now < grid[-1]:
                scores = eig_scores(betas, sig2, Mem, AHm, E)
                masks.append(int(np.argmax(scores)))
            t_cum += time.perf_counter() - t0
        res["shapleig"][str(s)]["betas_final"] = betas.tolist()
        print(f"  {name} seed {s} done", flush=True)
    return res


def main():
    self_test()
    for fname in sorted(os.listdir(GAMES_DIR)):
        if not fname.endswith(".npz"):
            continue
        name = fname[:-4]
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        os.makedirs(os.path.join(HERE, "schedules"), exist_ok=True)
        out = os.path.join(HERE, "results", f"results_{name}.json")
        if os.path.exists(out):
            print(f"skip {name} (results exist)")
            continue
        tab = np.load(os.path.join(GAMES_DIR, fname))
        d = int(tab["d"])
        sched_path = os.path.join(HERE, "schedules", f"schedule_d{d}.npy")
        if os.path.exists(sched_path):
            schedule = np.load(sched_path)
        else:
            t0 = time.time()
            schedule = fixed_schedule(d, max(GRIDS[d]))
            np.save(sched_path, schedule)
            print(f"schedule d={d}: {time.time() - t0:.1f}s (offline, cached)")
        print(f"=== {name} (d={d}) ===", flush=True)
        t0 = time.time()
        res = run_game(name, tab, schedule)
        res["d"] = d
        with open(out, "w") as f:
            json.dump(res, f)
        print(f"{name}: total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
