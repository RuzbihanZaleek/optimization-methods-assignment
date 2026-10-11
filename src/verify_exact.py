"""Independent check that the ILP in src/exact_ilp.py really finds the optimum.

Owner: M.Z.M. Ruzbihan (Member 1)

On small instances we can find the true optimum by exhaustive dynamic programming, without
any solver, and compare it with the ILP result. If the formulation had a mistake (for example
a missing constraint), the two numbers would differ.

How the dynamic programming works (customers are represented as subsets S):
    1. tour[S] = shortest single-van tour depot -> every customer in S -> depot,
                 found with the Held-Karp algorithm; only allowed if load(S) <= Q.
    2. best[S] = cheapest way to serve all of S with any number of vans
               = min over feasible van groups T that contain the first customer of S
                 of tour[T] + best[S without T].
    The answer is best[all customers]. Time grows like 3^n, so this is only for small n.

Run from the repository root (takes about one minute):
    python -m src.verify_exact
"""

import pandas as pd

from src.data_prep import REPO_ROOT, load_customers, sample_instance
from src.exact_ilp import solve_cvrp_ilp

RESULTS_PATH = REPO_ROOT / "results" / "tables" / "ilp_verification.csv"

# (number of customers, subset seed, capacity Q in kg)
CASES = [(8, 1, 1000), (8, 2, 1000), (10, 1, 1000), (10, 2, 1000), (10, 3, 600), (12, 2, 1000)]


def exhaustive_optimum(instance):
    """True optimal total distance by dynamic programming over customer subsets."""
    dist = instance["dist"]
    demand = instance["demand"]
    capacity = instance["capacity"]
    n = len(demand) - 1
    infinity = float("inf")

    # Customer k (0-based in the bit mask) is node k + 1 in the instance.
    # path[S][k] = shortest path from the depot that visits all of S and ends at customer k.
    path = [[infinity] * n for _ in range(1 << n)]
    for k in range(n):
        path[1 << k][k] = dist[0][k + 1]
    for subset in range(1, 1 << n):
        for k in range(n):
            if not (subset >> k) & 1 or path[subset][k] == infinity:
                continue
            for m in range(n):
                if (subset >> m) & 1:
                    continue
                bigger = subset | (1 << m)
                new_length = path[subset][k] + dist[k + 1][m + 1]
                if new_length < path[bigger][m]:
                    path[bigger][m] = new_length

    # tour[S]: close the path back to the depot, only if one van can carry all of S.
    tour = [infinity] * (1 << n)
    for subset in range(1, 1 << n):
        load = sum(demand[k + 1] for k in range(n) if (subset >> k) & 1)
        if load <= capacity + 1e-9:
            tour[subset] = min(path[subset][k] + dist[k + 1][0] for k in range(n) if (subset >> k) & 1)

    # best[S]: split S into van groups.
    best = [infinity] * (1 << n)
    best[0] = 0.0
    for subset in range(1, 1 << n):
        first = subset & -subset          # lowest customer in S, must be in some van group
        rest = subset ^ first
        group_rest = rest
        while True:                       # loop over every sub-group of `rest`
            group = group_rest | first
            if tour[group] < infinity:
                best[subset] = min(best[subset], tour[group] + best[subset ^ group])
            if group_rest == 0:
                break
            group_rest = (group_rest - 1) & rest
    return best[(1 << n) - 1]


def main():
    customers = load_customers()
    rows = []
    for n, seed, capacity in CASES:
        instance = sample_instance(customers, n=n, seed=seed, capacity=capacity)
        ilp = solve_cvrp_ilp(instance)
        exact = exhaustive_optimum(instance)
        match = ilp["status"] == "optimal" and abs(ilp["distance_km"] - exact) < 1e-4
        rows.append({
            "instance": instance["name"],
            "n_customers": n,
            "capacity_kg": capacity,
            "ilp_status": ilp["status"],
            "ilp_distance_km": round(ilp["distance_km"], 4),
            "exhaustive_distance_km": round(exact, 4),
            "ilp_routes": len(ilp["routes"]),
            "ilp_runtime_s": round(ilp["runtime_s"], 2),
            "match": match,
        })
        print(f"n={n:2d} seed={seed} Q={capacity:4d}: ILP {ilp['distance_km']:.4f} km ({ilp['status']}, "
              f"{len(ilp['routes'])} routes) | exhaustive {exact:.4f} km | match: {match}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH.relative_to(REPO_ROOT)}")
    all_match = all(row["match"] for row in rows)
    print("ALL MATCH: the ILP finds the true optimum." if all_match else "MISMATCH: check the ILP model.")
    assert all_match


if __name__ == "__main__":
    main()
