# Aggregate the replication sweeps (possibly several multirun roots) into one
# mega-figure: one row per game, four columns —
#   (1) MSE (mean +/- SEM over seeds) vs total evaluations
#   (2) method compute to reach each budget vs evaluations: cumulative
#       selection compute (method fits + acquisition) plus the readout at
#       that budget (the evaluation fit + posterior read a user stopping
#       there would pay). For GP+Leverage the recorded per-budget duration
#       IS that total (the method restarts from scratch every budget), so
#       it is not summed.
#   (3) compute-error tradeoff: the same two traces against each other,
#       one point per budget, increasing budget moving right and down
#   (4) selection-size trace: mean min(|T|, p-|T|) per iteration
# Usage: python repro_plots.py <multirun_root> [<multirun_root2> ...]
import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

ARMS = {
    "xac.acquisition_functions.EIGFunctionProperty":
        ("ShaplEIG (EIG, refit every iter)", "#b45309"),
    "xac.acquisition_functions.HybridPairedEIG":
        ("Hybrid: extremes then EIG, geometric refits", "#1d4ed8"),
    "xac.acquisition_functions.PairedExtremes":
        ("Paired extremes (fixed order)", "#0d9488"),
    "xac.acquisition_functions.LeverageGPSampler":
        ("GP + Leverage Score Sampling", "#7c3aed"),
    "xac.acquisition_functions.Random":
        ("GP + Random", "#be185d"),
}
LEVGP = "xac.acquisition_functions.LeverageGPSampler"
GAME_ORDER = ["vit_9", "dvbsgb_10", "dvbsrf_10", "dvchgb_10",
              "resnet_14", "vit_16"]  # rows in increasing p
GAME_TITLES = {"dvbsrf_10": "DV; RF; Bike Sharing (p=10)",
               "dvbsgb_10": "DV; GB; Bike Sharing (p=10)",
               "dvchgb_10": "DV; GB; Cal. Housing (p=10)",
               "vit_9": "LE; ViT; 9 patches (p=9)",
               "resnet_14": "LE; ResNet-18; 14 superpix. (p=14)",
               "vit_16": "LE; ViT; 16 patches (p=16)"}
INK = "#243330"
MUTED = "#6b7975"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e3e9e7", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "svg.fonttype": "none",
})


def collect(roots):
    runs = []
    for root in roots:
        for mpath in glob.glob(os.path.join(root, "**", "metrics.json"),
                               recursive=True):
            run_dir = os.path.dirname(mpath)
            cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
            if not os.path.exists(cfg_path):
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            with open(mpath) as f:
                met = json.load(f)
            acq = cfg["acquisition"]["_target_"]

            def arr(key):
                v = met.get(key)
                return np.array(v, dtype=float) if v else np.zeros(0)

            if acq == LEVGP:
                # per-budget totals; the whole method reruns every budget
                total_time = arr("acq_fun_duration")
            else:
                def padsum(*keys):
                    vs = [arr(k) for k in keys]
                    L = max((len(v) for v in vs), default=0)
                    return sum(np.pad(v, (0, L - len(v))) for v in vs) \
                        if L else np.zeros(0)

                cumsel = np.cumsum(padsum("hp_fit_duration",
                                          "acq_fun_duration"))
                readout = padsum("eval_fit_duration", "prop_post_duration")
                if len(cumsel) and len(readout):
                    L = len(readout)
                    cum_ext = np.concatenate(
                        [cumsel, np.full(max(0, L - len(cumsel)), cumsel[-1])]
                    )[:L]
                    total_time = cum_ext + readout
                else:
                    total_time = np.zeros(0)

            runs.append({
                "game": cfg["blackbox"]["name"],
                "acq": acq,
                "seed": cfg["meta"]["seed"],
                "init": int(met.get("initial_design_size", 0)),
                "mse": np.array(met["mse"], dtype=float),
                "sizes": np.array(met.get("archive_sizes", []), dtype=float),
                "time": total_time,
            })
    return runs


def main(roots):
    runs = collect(roots)
    print(f"collected {len(runs)} runs from {len(roots)} root(s)")
    present = {r["game"] for r in runs}
    games = [g for g in GAME_ORDER if g in present]
    games += sorted(present - set(games))
    R = len(games)

    fig, axes = plt.subplots(R, 4, figsize=(19.5, 3.8 * R),
                             layout="constrained")
    axes = np.atleast_2d(axes)

    for gi, game in enumerate(games):
        axE, axT, axP, axS = axes[gi]
        p = int(game.rsplit("_", 1)[1])
        game_runs = [r for r in runs if r["game"] == game]
        init = max(r["init"] for r in game_runs) or (p + 1)
        bottom = gi == R - 1

        for acq, (label, color) in ARMS.items():
            sel = [r for r in game_runs if r["acq"] == acq]
            if not sel:
                continue
            # error vs evaluations
            L = min(len(r["mse"]) for r in sel)
            M = np.stack([r["mse"][:L] for r in sel])
            x = init + np.arange(L)
            mean = M.mean(0)
            sem = M.std(0, ddof=1) / np.sqrt(M.shape[0])
            axE.plot(x, mean, color=color, linewidth=1.8, label=label)
            axE.fill_between(x, mean - sem, mean + sem, color=color,
                             alpha=0.2, linewidth=0)
            # method compute vs evaluations
            ts = [r["time"] for r in sel if len(r["time"]) > 4]
            Tmean = None
            if ts:
                Lt = min(len(t) for t in ts)
                T = np.stack([t[:Lt] for t in ts])
                Tmean = T.mean(0)
                axT.plot(init + np.arange(Lt), Tmean, color=color,
                         linewidth=1.8)
                axT.fill_between(init + np.arange(Lt),
                                 np.percentile(T, 20, axis=0),
                                 np.percentile(T, 80, axis=0), color=color,
                                 alpha=0.15, linewidth=0)
            # compute-error tradeoff (one point per budget)
            if Tmean is not None:
                Lp = min(len(Tmean), len(mean))
                axP.plot(Tmean[:Lp], mean[:Lp], color=color, linewidth=1.6)
                axP.plot(Tmean[Lp - 1], mean[Lp - 1], "o", color=color,
                         markersize=4)
            # selection-size trace
            szs = [r["sizes"] for r in sel if len(r["sizes"]) > init]
            if szs:
                Ls = min(len(s) for s in szs)
                S = np.stack([s[:Ls] for s in szs])[:, init:]
                dist = np.minimum(S, p - S)
                it = np.arange(dist.shape[1])
                axS.plot(it, dist.mean(0), color=color, linewidth=1.5)
                axS.fill_between(it, np.percentile(dist, 20, axis=0),
                                 np.percentile(dist, 80, axis=0),
                                 color=color, alpha=0.13, linewidth=0)

        n_seeds = max(len({r["seed"] for r in game_runs if r["acq"] == a})
                      for a in ARMS)
        axE.set_yscale("log")
        axE.set_ylabel(f"{GAME_TITLES.get(game, game)}\n\n"
                       f"MSE (mean ± SEM, {n_seeds} seeds)")
        axT.set_yscale("log")
        axT.set_ylabel("method compute (s)")
        axP.set_xscale("log")
        axP.set_yscale("log")
        axP.set_ylabel("MSE")
        axS.set_ylabel("min(|T|, p−|T|)")
        axS.set_ylim(-0.3, p / 2 + 0.3)
        for ax in (axE, axT, axP, axS):
            ax.grid(True)
        if bottom:
            axE.set_xlabel("evaluations (incl. init design)")
            axT.set_xlabel("evaluations")
            axP.set_xlabel("method compute (s)")
            axS.set_xlabel("iteration (after init design)")
        if gi == 0:
            axE.set_title("error by evaluations", fontsize=11)
            axT.set_title("compute by evaluations", fontsize=11)
            axP.set_title("compute-error tradeoff (dot = final budget)",
                          fontsize=11)
            axS.set_title("distance of selected coalition to size extreme",
                          fontsize=11)
        print(f"plotted {game}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=5,
               frameon=False, fontsize=9.5)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"all_games.{ext}"), dpi=150)
    plt.close(fig)

    print("\nMSE (mean over seeds) and method time at selected budgets:")
    for game in games:
        game_runs = [r for r in runs if r["game"] == game]
        init = max(r["init"] for r in game_runs)
        final = init + min(len(r["mse"]) for r in game_runs) - 1
        for target in (32, 64, 128, 256, final):
            row = [f"{game} m={target}:"]
            for acq, (label, _) in ARMS.items():
                sel = [r for r in game_runs if r["acq"] == acq]
                if not sel:
                    continue
                idx = target - init
                vals = [r["mse"][idx] for r in sel if len(r["mse"]) > idx]
                ts = [r["time"][min(idx, len(r["time"]) - 1)]
                      for r in sel if len(r["time"]) > 4]
                tag = label.split(" (")[0].split(":")[0]
                if vals:
                    t = f"/{np.mean(ts):.0f}s" if ts else ""
                    row.append(f"{tag}={np.mean(vals):.2e}{t}")
            print("  " + "  ".join(row))


if __name__ == "__main__":
    roots = sys.argv[1:] or [os.path.join(HERE, "shapleig-repo", "multirun")]
    main(roots)
