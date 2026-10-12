"""All result figures for the report and the video, made from the CSV files in results/tables/.

Owner: U.D.L.L De Silva (Member 2)

Each method always has the same colour AND marker shape, so the figures can be read in
black-and-white print and by colour-blind readers:
    ILP = blue square, Greedy = orange triangle, SA = aqua circle.

Run from the repository root (after run_comparison and run_heuristic_analysis):
    python -m src.plots
"""

import json
import math

import matplotlib.pyplot as plt
import pandas as pd

from src.data_prep import REPO_ROOT, load_customers, sample_instance, depot_location, make_instance

TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"

STYLE = {
    "ILP": {"color": "#2a78d6", "marker": "s", "label": "ILP (CBC)"},
    "Greedy": {"color": "#eb6834", "marker": "^", "label": "Greedy"},
    "SA": {"color": "#1baf7a", "marker": "o", "label": "Simulated annealing"},
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved results/figures/{name}")


def draw_routes(ax, instance, routes, colour, title):
    """Draw every route of a solution in one colour; the depot is a black square."""
    lats, lons = instance["lats"], instance["lons"]
    for route in routes:
        ax.plot([lons[n] for n in route], [lats[n] for n in route], "-", color=colour, linewidth=1.4, alpha=0.9)
    ax.scatter(lons[1:], lats[1:], s=18, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(lons[0], lats[0], marker="s", s=70, color=INK, zorder=4, label="Depot")
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_aspect(1 / math.cos(math.radians(lats[0])))


def best_routes(runs, method, distance_column, instance=None):
    """Routes of the shortest run of a method (optionally on one instance)."""
    rows = runs[runs["method"] == method]
    if instance is not None:
        rows = rows[rows["instance"] == instance]
    best = rows.loc[rows[distance_column].idxmin()]
    return json.loads(best["routes"]), best


# ----------------------------------------------------------------------------------------------
# E1 / E2 figures (data from src/run_comparison.py)
# ----------------------------------------------------------------------------------------------

def plot_greedy_vs_optimal_routes(e1_runs, customers):
    """Greedy plan vs the proven optimal plan on the E1 instance where greedy is worst."""
    summary = pd.read_csv(TABLES / "e1_summary.csv")
    worst = summary.loc[summary["greedy_gap_to_opt_pct"].idxmax()]
    n, seed = int(worst["n_customers"]), int(worst["instance"].split("seed")[1])
    instance = sample_instance(customers, n=n, seed=seed)
    greedy_routes, greedy = best_routes(e1_runs, "Greedy", "distance_km", instance=worst["instance"])
    ilp_routes, ilp = best_routes(e1_runs, "ILP", "distance_km", instance=worst["instance"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    draw_routes(axes[0], instance, greedy_routes, STYLE["Greedy"]["color"],
                f"Greedy: {greedy['distance_km']:.1f} km")
    draw_routes(axes[1], instance, ilp_routes, STYLE["ILP"]["color"],
                f"Proven optimum (ILP; SA finds the same): {ilp['distance_km']:.1f} km")
    fig.suptitle(f"Instance {worst['instance']}: greedy is {worst['greedy_gap_to_opt_pct']:.1f} % longer "
                 f"than the optimum", color=INK)
    save(fig, "f3_greedy_vs_optimal_routes.png")


def plot_runtime_vs_size(e1_runs, e2_runs):
    """Runtime against number of customers, log scale. Hollow ILP markers = stopped by the time limit."""
    runs = pd.concat([e1_runs, e2_runs])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for method in ("ILP", "Greedy", "SA"):
        per_n = runs[runs["method"] == method].groupby("n_customers")["runtime_s"].mean()
        style = STYLE[method]
        ax.plot(per_n.index, per_n.values, color=style["color"], linewidth=2, alpha=0.6)
        if method == "ILP":
            proven = runs[(runs["method"] == "ILP")].groupby("n_customers")["status"].apply(
                lambda s: (s == "optimal").all())
            ax.scatter(per_n.index[proven.values], per_n.values[proven.values], marker="s", s=45,
                       color=style["color"], zorder=3, label="ILP, proven optimal")
            ax.scatter(per_n.index[~proven.values], per_n.values[~proven.values], marker="s", s=45,
                       facecolor="white", edgecolor=style["color"], linewidth=1.8, zorder=3,
                       label="ILP, stopped at 300 s limit")
        else:
            ax.scatter(per_n.index, per_n.values, marker=style["marker"], s=45, color=style["color"],
                       zorder=3, label=style["label"] + (" (mean of seeds)" if method == "SA" else ""))
    ax.axhline(300, color=MUTED, linestyle="--", linewidth=1)
    ax.text(runs["n_customers"].min(), 300, "300 s time limit", va="bottom", ha="left", color=MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Number of customers (n)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Runtime grows explosively for the exact method", color=INK)
    ax.legend(loc="center right")
    save(fig, "f4_runtime_vs_size.png")


def plot_distance_vs_size(e2_summary):
    """E2: total distance of each method as the instance grows."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    series = {"ILP": "ilp_km", "Greedy": "greedy_km", "SA": "sa_mean_km"}
    for method, column in series.items():
        style = STYLE[method]
        label = {"ILP": "ILP best plan after 300 s", "Greedy": "Greedy", "SA": "SA (mean of 5 seeds)"}[method]
        ax.plot(e2_summary["n_customers"], e2_summary[column], marker=style["marker"], markersize=7,
                color=style["color"], label=label)
        last = e2_summary.iloc[-1]
        ax.annotate(f"{last[column]:.0f} km", (last["n_customers"], last[column]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Number of customers (n)")
    ax.set_ylabel("Total distance (km)")
    ax.set_title("E2: plan quality when the ILP can no longer prove optimality", color=INK)
    ax.legend(loc="upper left")
    ax.set_xlim(right=e2_summary["n_customers"].max() * 1.12)
    save(fig, "f5_distance_vs_size.png")


def plot_gap_to_optimum(e1_summary):
    """E1: how far greedy and SA are from the proven optimum on each small instance."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(e1_summary))
    # small sideways offset so a marker at 0 % never hides the other method's marker
    for offset, (method, column) in ((-0.15, ("Greedy", "greedy_gap_to_opt_pct")),
                                     (0.15, ("SA", "sa_mean_gap_to_opt_pct"))):
        style = STYLE[method]
        label = style["label"] + (" (mean of 10 seeds)" if method == "SA" else "")
        ax.scatter([i + offset for i in x], e1_summary[column].clip(lower=0), marker=style["marker"], s=55,
                   color=style["color"], zorder=3, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(e1_summary["instance"], rotation=45, ha="right")
    ax.set_ylabel("Gap to proven optimum (%)")
    ax.set_title("E1: SA reached the proven optimum in all 120 runs; greedy did not", color=INK)
    ax.legend(loc="upper left")
    save(fig, "f6_gap_to_optimum.png")


# ----------------------------------------------------------------------------------------------
# E3 / E4 / E5 figures (data from src/run_heuristic_analysis.py)
# ----------------------------------------------------------------------------------------------

def plot_sa_convergence(history):
    """E4: current and best distance over the iterations (top) and the temperature (bottom)."""
    run = history[(history["instance"] == "all_87_Q1000") & (history["sa_seed"] == 0)]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    top.plot(run["iteration"], run["current_distance"], color=STYLE["SA"]["color"], linewidth=1.2, alpha=0.75,
             label="Current solution")
    top.plot(run["iteration"], run["best_distance"], color=INK, linewidth=2, label="Best so far")
    top.annotate(f"start (greedy): {run['current_distance'].iloc[0]:.0f} km",
                 (0, run["current_distance"].iloc[0]), xytext=(4, -8), textcoords="offset points",
                 color=INK, fontsize=9, va="top", ha="left",
                 bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"})
    top.annotate(f"final best: {run['best_distance'].iloc[-1]:.0f} km",
                 (run["iteration"].iloc[-1], run["best_distance"].iloc[-1]), xytext=(-4, 10),
                 textcoords="offset points", color=INK, fontsize=9, ha="right")
    top.set_ylabel("Total distance (km)")
    top.set_title("E4: simulated annealing on all 87 customers (seed 0, Q = 1,000 kg)", color=INK)
    top.legend(loc="upper right")
    bottom.plot(run["iteration"], run["temperature"], color=MUTED, linewidth=1.6)
    bottom.set_yscale("log")
    bottom.set_ylabel("Temperature T")
    bottom.set_xlabel("Iteration")
    bottom.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    save(fig, "f7_sa_convergence.png")


def plot_sa_seed_spread(e3_runs):
    """E4: final SA distance for each of the 10 seeds, one small panel per capacity."""
    sa = e3_runs[(e3_runs["experiment"] == "E3") & (e3_runs["method"] == "SA")]
    capacities = sorted(sa["capacity"].unique())
    fig, axes = plt.subplots(1, len(capacities), figsize=(10, 3.8))
    for ax, capacity in zip(axes, capacities):
        values = sa[sa["capacity"] == capacity]["distance"]
        ax.boxplot(values, widths=0.5, medianprops={"color": INK}, boxprops={"color": MUTED},
                   whiskerprops={"color": MUTED}, capprops={"color": MUTED}, showfliers=False)
        ax.scatter([1] * len(values), values, color=STYLE["SA"]["color"], edgecolor="white", s=40, zorder=3)
        ax.set_title(f"Q = {capacity:.0f} kg", color=INK, fontsize=10)
        ax.set_xticks([])
        spread = 100 * (values.max() - values.min()) / values.min()
        ax.set_xlabel(f"best {values.min():.0f} · worst {values.max():.0f}\n(spread {spread:.1f} %)", color=MUTED)
    axes[0].set_ylabel("SA final distance (km)")
    fig.suptitle("E4: variation of SA across 10 random seeds (all 87 customers)", color=INK)
    save(fig, "f8_sa_seed_spread.png")


def plot_capacity_sensitivity(e3_summary):
    """E3: van capacity vs total distance and vs number of vans (two panels, never a dual axis)."""
    data = e3_summary[e3_summary["experiment"] == "E3"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for method in ("Greedy", "SA"):
        rows = data[data["method"] == method]
        style = STYLE[method]
        label = style["label"] + (" (best of 10 seeds)" if method == "SA" else "")
        left.plot(rows["capacity"], rows["best_distance"], marker=style["marker"], markersize=7,
                  color=style["color"], label=label)
        right.plot(rows["capacity"], rows["routes_min"], marker=style["marker"], markersize=7,
                   color=style["color"], label=style["label"])
    lower = data[data["method"] == "SA"]
    right.plot(lower["capacity"], lower["min_routes_needed"], linestyle="--", color=MUTED, linewidth=1.2,
               label="Lower bound ⌈Σq / Q⌉")
    left.set_xlabel("Van capacity Q (kg)")
    left.set_ylabel("Total distance (km)")
    left.set_title("Distance", color=INK)
    left.legend()
    right.set_xlabel("Van capacity Q (kg)")
    right.set_ylabel("Number of vans used")
    right.set_title("Vans", color=INK)
    right.legend()
    for ax in (left, right):
        ax.set_xticks(sorted(data["capacity"].unique()))
    fig.suptitle("E3: effect of van capacity (all 87 customers)", color=INK)
    save(fig, "f9_capacity_sensitivity.png")


def plot_benchmark(e5_runs, optimum):
    """E5: every method's result on A-n32-k5 compared with the published optimum."""
    data = e5_runs[e5_runs["experiment"] == "E5"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rows = [("ILP", "ILP after 300 s"), ("Greedy", "Greedy"), ("SA", "SA, 10 seeds")]
    for y, (method, label) in enumerate(rows):
        values = data[data["method"] == method]["distance"].dropna()
        style = STYLE[method]
        if method == "ILP" and data[data["method"] == "ILP"]["status"].iloc[0] != "optimal":
            ax.scatter(values, [y] * len(values), marker="s", s=60, facecolor="white",
                       edgecolor=style["color"], linewidth=1.8, zorder=3)
        else:
            ax.scatter(values, [y] * len(values), marker=style["marker"], s=60, color=style["color"],
                       edgecolor="white", zorder=3, alpha=0.85)
        best = values.min()
        label = f"best {best:.0f}"
        if method == "SA":
            label += f" ({int((abs(values - optimum) < 1e-6).sum())} of {len(values)} runs at the optimum)"
        ax.annotate(label, (best, y), xytext=(6, -14), textcoords="offset points", ha="left", color=INK,
                    fontsize=9)
    ax.axvline(optimum, color=INK, linestyle="--", linewidth=1.2)
    ax.text(optimum, len(rows) - 0.45, f" published optimum {optimum:.0f}", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, label in rows])
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Total distance (rounded Euclidean units)")
    ax.set_title("E5: benchmark A-n32-k5 (31 customers, Q = 100)", color=INK)
    save(fig, "f10_benchmark.png")


def plot_sa_routes_all87(e3_runs, customers):
    """Greedy vs best SA plan on all 87 customers (Q = 1,000 kg)."""
    instance = make_instance(customers, depot_location(customers), name="all_87")
    greedy_routes, greedy = best_routes(e3_runs, "Greedy", "distance", instance="all_87_Q1000")
    sa_routes, sa = best_routes(e3_runs, "SA", "distance", instance="all_87_Q1000")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    draw_routes(axes[0], instance, greedy_routes, STYLE["Greedy"]["color"],
                f"Greedy: {greedy['distance']:.0f} km, {len(greedy_routes)} vans")
    draw_routes(axes[1], instance, sa_routes, STYLE["SA"]["color"],
                f"Simulated annealing (best seed): {sa['distance']:.0f} km, {len(sa_routes)} vans")
    fig.suptitle("All 87 customers, Q = 1,000 kg", color=INK)
    save(fig, "f11_routes_all87_greedy_vs_sa.png")


def main():
    customers = load_customers()

    if (TABLES / "e1_runs.csv").exists():
        e1_runs = pd.read_csv(TABLES / "e1_runs.csv")
        e2_runs = pd.read_csv(TABLES / "e2_runs.csv")
        plot_greedy_vs_optimal_routes(e1_runs, customers)
        plot_runtime_vs_size(e1_runs, e2_runs)
        plot_distance_vs_size(pd.read_csv(TABLES / "e2_summary.csv"))
        plot_gap_to_optimum(pd.read_csv(TABLES / "e1_summary.csv"))
    else:
        print("E1/E2 tables missing: run `python -m src.run_comparison` first")

    if (TABLES / "e3_e5_runs.csv").exists():
        runs = pd.read_csv(TABLES / "e3_e5_runs.csv")
        summary = pd.read_csv(TABLES / "e3_e5_summary.csv")
        optimum = summary.loc[summary["experiment"] == "E5", "published_optimum"].iloc[0]
        plot_sa_convergence(pd.read_csv(TABLES / "e4_sa_history.csv"))
        plot_sa_seed_spread(runs)
        plot_capacity_sensitivity(summary)
        plot_benchmark(runs, optimum)
        plot_sa_routes_all87(runs, customers)
    else:
        print("E3-E5 tables missing: run `python -m src.run_heuristic_analysis` first")


if __name__ == "__main__":
    main()
