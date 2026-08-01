# Plots for the three-baselines benchmark: per game, error vs m and method
# compute time vs m, median with 20-80% band over the 8 seeds.
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

METHODS = [("uniform", "#7c3aed", "Uniform by size + one fit"),
           ("shapleig", "#b45309", "ShaplEIG (EIG + refit)"),
           ("paired", "#0d9488", "Paired extremes + one fit")]
INK = "#243330"
MUTED = "#6b7975"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e3e9e7", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "svg.fonttype": "none",
})


def quantiles(res, method, key):
    seeds = [k for k in res[method] if k.isdigit()]
    M = np.array(res[method][seeds[0]]["m"])
    vals = np.array([res[method][s][key] for s in seeds])
    lo, med, hi = np.percentile(vals, [20, 50, 80], axis=0)
    return M, lo, med, hi


def main():
    for path in sorted(glob.glob(os.path.join(HERE, "results", "results_*.json"))):
        name = os.path.basename(path)[len("results_"):-len(".json")]
        with open(path) as f:
            res = json.load(f)
        d = res["d"]
        fig, (axE, axT) = plt.subplots(1, 2, figsize=(9.6, 4.1), layout="constrained")
        for method, color, label in METHODS:
            M, lo, med, hi = quantiles(res, method, "err")
            axE.fill_between(M, lo, hi, color=color, alpha=0.15, linewidth=0)
            axE.plot(M, med, color=color, linewidth=2, marker="o", markersize=4.5,
                     label=label)
            Mt, lot, medt, hit = quantiles(res, method, "time")
            axT.fill_between(Mt, lot, hit, color=color, alpha=0.15, linewidth=0)
            axT.plot(Mt, medt, color=color, linewidth=2, marker="o", markersize=4.5,
                     label=label)
        floor = np.median(res["noise_floor"])
        axE.axhline(floor, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        axE.text(axE.get_xlim()[0], floor, " game-noise floor", color=MUTED,
                 fontsize=8.5, va="bottom", ha="left")
        for ax, ylab in ((axE, "relative $\\ell_2$ error of $\\hat{\\phi}$"),
                         (axT, "method compute time (s)")):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("evaluations $m$")
            ax.set_ylabel(ylab)
            ax.grid(True, which="major", axis="both")
        axE.legend(frameon=False, fontsize=8.5, loc="lower left")
        fig.suptitle(f"{name}  (d = {d}, 8 seeds, median with 20–80% band)",
                     fontsize=11)
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(HERE, "..", "figures", "bench",
                                     f"bench_{name}.{ext}"), dpi=180)
        plt.close(fig)
        print(f"plotted {name}")

    # console summary: error at the largest common m, plus fitted-lengthscale spread
    print("\nsummary (median rel. error at largest m):")
    for path in sorted(glob.glob(os.path.join(HERE, "results", "results_*.json"))):
        name = os.path.basename(path)[len("results_"):-len(".json")]
        with open(path) as f:
            res = json.load(f)
        row = [name]
        for method, _, _ in METHODS:
            _, _, med, _ = quantiles(res, method, "err")
            row.append(f"{method}={med[-1]:.4f}")
        row.append(f"floor={np.median(res['noise_floor']):.4f}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
