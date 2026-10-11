"""Experiments E1 and E2: exact ILP vs greedy vs simulated annealing.

Owner: M.Z.M. Ruzbihan (Member 1)

E1 (quality on small instances): n in {5, 8, 10, 12}, 3 random customer subsets per size.
    Every method solves the same instance. The ILP gives the proven optimum, so we can measure
    how far the heuristics are from it (optimality gap).
E2 (scalability): n in {15, 20, 30, 45, 60, 87}, one subset per size (87 = all customers).
    The ILP gets the same time limit; we record whether it can still prove optimality,
    its best plan, and its lower bound.

Fairness rules:
    - same instance, distance matrix, depot and Q for every method,
    - every result is re-scored by src/evaluate.py (no method grades its own work),
    - SA uses a fixed iteration budget and 10 (E1) or 5 (E2) fixed seeds,
    - all runs are sequential on the same machine (see results/tables/environment.txt).

Run from the repository root (takes roughly 30-45 minutes because of the ILP time limits):
    python -m src.run_comparison          # E1 and E2
    python -m src.run_comparison e1       # only E1
    python -m src.run_comparison e2       # only E2
"""

import json
import os
import platform
import statistics
import sys

import pandas as pd
import pulp

from src.data_prep import REPO_ROOT, VEHICLE_CAPACITY_KG, load_customers, sample_instance
from src.evaluate import summarize
from src.exact_ilp import solve_cvrp_ilp
from src.heuristics import SA_ITERATIONS, solve_greedy, solve_sa

TABLES = REPO_ROOT / "results" / "tables"

ILP_TIME_LIMIT_S = 300
E1_SIZES = [5, 8, 10, 12]
E1_SUBSET_SEEDS = [1, 2, 3]
E1_SA_SEEDS = list(range(10))
E2_SIZES = [15, 20, 30, 45, 60, 87]
E2_SUBSET_SEED = 1
E2_SA_SEEDS = list(range(5))


def result_row(experiment, instance, subset_seed, result, sa_seed=None):
    """One CSV row. Distance and feasibility come from the shared judge in src/evaluate.py."""
    judged = summarize(result["routes"], instance) if result["routes"] else {}
    return {
        "experiment": experiment,
        "instance": instance["name"],
        "n_customers": len(instance["customer_ids"]),
        "subset_seed": subset_seed,
        "capacity_kg": instance["capacity"],
        "method": result["method"],
        "sa_seed": sa_seed,
        "status": result["status"],
        "distance_km": judged.get("distance_km"),
        "runtime_s": result["runtime_s"],
        "n_routes": judged.get("n_routes"),
        "min_routes_needed": judged.get("min_routes_needed"),
        "max_load_kg": judged.get("max_load_kg"),
        "max_volume_dm3": judged.get("max_volume_dm3"),
        "feasible": judged.get("feasible"),   # empty when the ILP found no plan within the time limit
        "ilp_lower_bound_km": result.get("best_bound"),
        "routes": json.dumps(result["routes"]),
    }


def run_all_methods(experiment, instance, subset_seed, sa_seeds):
    """Solve one instance with ILP, greedy and SA (one run per SA seed). Returns CSV rows."""
    rows = []
    ilp = solve_cvrp_ilp(instance, time_limit=ILP_TIME_LIMIT_S)
    rows.append(result_row(experiment, instance, subset_seed, ilp))
    print(f"  ILP    : {ilp['status']:<11} {ilp['distance_km'] or float('nan'):9.2f} km  {ilp['runtime_s']:7.2f} s",
          flush=True)

    greedy = solve_greedy(instance)
    rows.append(result_row(experiment, instance, subset_seed, greedy))
    print(f"  Greedy : {greedy['distance_km']:21.2f} km  {greedy['runtime_s']:7.3f} s", flush=True)

    sa_distances = []
    for seed in sa_seeds:
        sa = solve_sa(instance, seed=seed)
        rows.append(result_row(experiment, instance, subset_seed, sa, sa_seed=seed))
        sa_distances.append(sa["distance_km"])
    print(f"  SA     : mean {statistics.mean(sa_distances):16.2f} km  (best {min(sa_distances):.2f}, "
          f"{len(sa_seeds)} seeds)", flush=True)
    return rows


def summarize_instances(runs):
    """One summary row per instance: ILP vs greedy vs SA (mean over seeds)."""
    summary = []
    for name, group in runs.groupby("instance", sort=False):
        ilp = group[group["method"] == "ILP"].iloc[0]
        greedy = group[group["method"] == "Greedy"].iloc[0]
        sa = group[group["method"] == "SA"]

        proven = ilp["status"] == "optimal"
        bound = ilp["ilp_lower_bound_km"]
        reference = ilp["distance_km"] if proven else None   # gaps to the optimum only when it is proven

        def gap(value, ref):
            return None if ref is None or pd.isna(ref) or pd.isna(value) else 100 * (value - ref) / ref

        summary.append({
            "experiment": ilp["experiment"],
            "instance": name,
            "n_customers": ilp["n_customers"],
            "capacity_kg": ilp["capacity_kg"],
            "min_routes_needed": greedy["min_routes_needed"],
            "ilp_status": ilp["status"],
            "ilp_km": ilp["distance_km"],
            "ilp_lower_bound_km": bound,
            "ilp_time_s": ilp["runtime_s"],
            "greedy_km": greedy["distance_km"],
            "greedy_time_s": greedy["runtime_s"],
            "sa_mean_km": sa["distance_km"].mean(),
            "sa_best_km": sa["distance_km"].min(),
            "sa_worst_km": sa["distance_km"].max(),
            "sa_std_km": sa["distance_km"].std(),
            "sa_mean_time_s": sa["runtime_s"].mean(),
            "greedy_gap_to_opt_pct": gap(greedy["distance_km"], reference),
            "sa_mean_gap_to_opt_pct": gap(sa["distance_km"].mean(), reference),
            "sa_runs_at_optimum": int((abs(sa["distance_km"] - reference) < 1e-6).sum()) if proven else None,
            # When the ILP is not proven optimal, the lower bound still limits how bad a plan can be:
            # the true optimum lies between the bound and the best plan found.
            "sa_best_gap_to_bound_pct": gap(sa["distance_km"].min(), bound),
            "sa_vs_ilp_best_pct": gap(sa["distance_km"].mean(), ilp["distance_km"]),
            "sa_routes": int(sa["n_routes"].iloc[0]),
            "all_feasible": bool(group["feasible"].dropna().all()),
        })
    return pd.DataFrame(summary)


def write_environment():
    """Record the machine and library versions, so the timings can be interpreted."""
    lines = [
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"logical CPUs: {os.cpu_count()}",
        f"python: {sys.version.split()[0]}",
        f"pulp: {pulp.__version__} (CBC solver bundled with PuLP)",
        f"pandas: {pd.__version__}",
        f"ILP time limit: {ILP_TIME_LIMIT_S} s",
        f"SA iterations per run: {SA_ITERATIONS}",
        f"capacity Q: {VEHICLE_CAPACITY_KG} kg",
    ]
    (TABLES / "environment.txt").write_text("\n".join(lines) + "\n")


def run_e1(customers):
    rows = []
    for n in E1_SIZES:
        for subset_seed in E1_SUBSET_SEEDS:
            instance = sample_instance(customers, n=n, seed=subset_seed)
            print(f"\nE1 {instance['name']} (Q = {instance['capacity']:.0f} kg)", flush=True)
            rows += run_all_methods("E1", instance, subset_seed, E1_SA_SEEDS)
    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "e1_runs.csv", index=False)
    summary = summarize_instances(runs)
    summary.to_csv(TABLES / "e1_summary.csv", index=False)
    return runs, summary


def run_e2(customers):
    rows = []
    for n in E2_SIZES:
        instance = sample_instance(customers, n=n, seed=E2_SUBSET_SEED)
        print(f"\nE2 {instance['name']} (Q = {instance['capacity']:.0f} kg)", flush=True)
        rows += run_all_methods("E2", instance, E2_SUBSET_SEED, E2_SA_SEEDS)
    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "e2_runs.csv", index=False)
    summary = summarize_instances(runs)
    summary.to_csv(TABLES / "e2_summary.csv", index=False)
    return runs, summary


def main():
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    TABLES.mkdir(parents=True, exist_ok=True)
    write_environment()
    customers = load_customers()

    columns = ["instance", "ilp_status", "ilp_km", "ilp_time_s", "greedy_km", "sa_mean_km", "sa_best_km",
               "sa_mean_time_s", "greedy_gap_to_opt_pct", "sa_mean_gap_to_opt_pct", "sa_runs_at_optimum",
               "sa_best_gap_to_bound_pct", "all_feasible"]
    for name, runner in (("e1", run_e1), ("e2", run_e2)):
        if which in (name, "all"):
            runs, summary = runner(customers)
            assert runs["feasible"].dropna().all(), f"{name}: an infeasible solution was produced"
            print(f"\n{name.upper()} summary (saved to results/tables/{name}_summary.csv):")
            print(summary[columns].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
