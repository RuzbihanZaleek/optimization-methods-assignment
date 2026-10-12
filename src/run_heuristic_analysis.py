"""Experiments E3, E4 and E5: how the heuristic behaves.

Owner: U.D.L.L De Silva (Member 2)

E3 (capacity sensitivity): all 87 customers with Q in {600, 1000, 1500} kg, greedy + SA (10 seeds).
    How do van capacity, number of vans and total distance trade off?
E4 (SA robustness and convergence): the Q = 1000 kg runs of E3 give 10 seeds on the full instance.
    We also save the cost history of these runs to plot how SA converges.
E5 (external validation): benchmark A-n32-k5 (published optimum 784). Greedy, SA (10 seeds) and
    the ILP with the same 300 s time limit as E1/E2.

Run from the repository root (takes roughly 10-15 minutes):
    python -m src.run_heuristic_analysis
"""

import json
import os
import platform
import sys

import pandas as pd

from src.benchmark import load_benchmark
from src.data_prep import REPO_ROOT, depot_location, load_customers, make_instance
from src.evaluate import summarize
from src.exact_ilp import solve_cvrp_ilp
from src.heuristics import solve_greedy, solve_sa

TABLES = REPO_ROOT / "results" / "tables"

E3_CAPACITIES = [600, 1000, 1500]
SA_SEEDS = list(range(10))
ILP_TIME_LIMIT_S = 300   # same limit as in E1/E2


def result_row(experiment, instance, result, sa_seed=None):
    """One CSV row, scored by the shared judge in src/evaluate.py."""
    judged = summarize(result["routes"], instance) if result["routes"] else {}
    return {
        "experiment": experiment,
        "instance": instance["name"],
        "n_customers": len(instance["demand"]) - 1,
        "capacity": instance["capacity"],
        "method": result["method"],
        "sa_seed": sa_seed,
        "status": result["status"],
        "distance": judged.get("distance_km"),
        "runtime_s": result["runtime_s"],
        "n_routes": judged.get("n_routes"),
        "min_routes_needed": judged.get("min_routes_needed"),
        "max_load": judged.get("max_load_kg"),
        "max_volume_dm3": judged.get("max_volume_dm3"),
        "feasible": judged.get("feasible"),
        "ilp_lower_bound": result.get("best_bound"),
        "routes": json.dumps(result["routes"]),
    }


def history_rows(experiment, instance, result):
    """Cost history of one SA run, for the convergence plot."""
    return [{"experiment": experiment, "instance": instance["name"], "capacity": instance["capacity"],
             "sa_seed": result["seed"], "iteration": it, "temperature": temp,
             "current_distance": current, "best_distance": best}
            for it, temp, current, best in result["history"]]


def run_heuristics(experiment, instance, rows, histories):
    greedy = solve_greedy(instance)
    rows.append(result_row(experiment, instance, greedy))
    distances = []
    for seed in SA_SEEDS:
        sa = solve_sa(instance, seed=seed)
        rows.append(result_row(experiment, instance, sa, sa_seed=seed))
        histories += history_rows(experiment, instance, sa)
        distances.append(sa["distance_km"])
    print(f"  greedy {greedy['distance_km']:.2f} ({len(greedy['routes'])} routes) | "
          f"SA mean {sum(distances) / len(distances):.2f}, best {min(distances):.2f}", flush=True)


def run_e3_e4(rows, histories):
    customers = load_customers()
    for capacity in E3_CAPACITIES:
        instance = make_instance(customers, depot_location(customers), capacity=capacity,
                                 name=f"all_87_Q{capacity}")
        print(f"\nE3/E4 {instance['name']}", flush=True)
        run_heuristics("E3", instance, rows, histories)


def run_e5(rows, histories):
    instance, optimum = load_benchmark()
    print(f"\nE5 {instance['name']} (published optimum {optimum:.0f})", flush=True)
    ilp = solve_cvrp_ilp(instance, time_limit=ILP_TIME_LIMIT_S)
    rows.append(result_row("E5", instance, ilp))
    print(f"  ILP {ilp['status']} {ilp['distance_km'] or float('nan'):.2f} in {ilp['runtime_s']:.1f} s, "
          f"lower bound {ilp['best_bound']}", flush=True)
    run_heuristics("E5", instance, rows, histories)
    return optimum


def summarize_runs(runs, e5_optimum):
    """One row per (instance, method): mean, best, worst and spread of distance and runtime."""
    summary = (runs.groupby(["experiment", "instance", "capacity", "method"], sort=False)
               .agg(runs=("distance", "size"),
                    mean_distance=("distance", "mean"),
                    best_distance=("distance", "min"),
                    worst_distance=("distance", "max"),
                    std_distance=("distance", "std"),
                    mean_runtime_s=("runtime_s", "mean"),
                    routes_min=("n_routes", "min"),
                    routes_max=("n_routes", "max"),
                    min_routes_needed=("min_routes_needed", "first"),
                    max_volume_dm3=("max_volume_dm3", "max"),
                    all_feasible=("feasible", lambda s: bool(s.dropna().all())))
               .reset_index())
    is_e5 = summary["experiment"] == "E5"
    summary.loc[is_e5, "published_optimum"] = e5_optimum
    summary.loc[is_e5, "mean_gap_to_optimum_pct"] = 100 * (summary["mean_distance"] - e5_optimum) / e5_optimum
    summary.loc[is_e5, "best_gap_to_optimum_pct"] = 100 * (summary["best_distance"] - e5_optimum) / e5_optimum
    e5_sa = runs[(runs["experiment"] == "E5") & (runs["method"] == "SA")]
    summary.loc[is_e5 & (summary["method"] == "SA"), "runs_at_optimum"] = \
        int((abs(e5_sa["distance"] - e5_optimum) < 1e-6).sum())
    return summary


def write_environment():
    """Record the machine these experiments ran on (E1/E2 may have run on another computer)."""
    lines = [
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"logical CPUs: {os.cpu_count()}",
        f"python: {sys.version.split()[0]}",
        f"ILP time limit (E5): {ILP_TIME_LIMIT_S} s",
        f"SA seeds: {SA_SEEDS[0]}-{SA_SEEDS[-1]}",
    ]
    (TABLES / "environment_e3_e5.txt").write_text("\n".join(lines) + "\n")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    write_environment()
    rows, histories = [], []
    run_e3_e4(rows, histories)
    e5_optimum = run_e5(rows, histories)

    runs = pd.DataFrame(rows)
    assert runs["feasible"].dropna().all(), "an infeasible solution was produced"
    runs.to_csv(TABLES / "e3_e5_runs.csv", index=False)
    # E4 only needs the convergence history of the main case (all 87 customers, Q = 1,000 kg)
    history = pd.DataFrame(histories)
    history = history[history["instance"] == "all_87_Q1000"].round(4)
    history.to_csv(TABLES / "e4_sa_history.csv", index=False)
    summary = summarize_runs(runs, e5_optimum)
    summary.to_csv(TABLES / "e3_e5_summary.csv", index=False)

    print("\nSummary (saved to results/tables/e3_e5_summary.csv):")
    columns = ["instance", "method", "runs", "mean_distance", "best_distance", "worst_distance", "std_distance",
               "mean_runtime_s", "routes_min", "routes_max", "min_routes_needed", "max_volume_dm3",
               "mean_gap_to_optimum_pct", "runs_at_optimum"]
    print(summary[columns].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
