"""Exact method: CVRP as an Integer Linear Program, solved by CBC (branch and bound) via PuLP.

Owner: M.Z.M. Ruzbihan (Member 1)

Model (two-index formulation with Miller-Tucker-Zemlin load constraints):
    min  sum_{i != j} d_ij * x_ij                                    (1) total distance
    s.t. sum_{j != i} x_ij = 1                  for every customer i  (2) leave once
         sum_{i != j} x_ij = 1                  for every customer j  (3) enter once
         sum_j x_0j = sum_i x_i0                                      (4) vans return to depot
         u_j >= u_i + q_j - Q * (1 - x_ij)      for customers i != j  (5) MTZ load / no subtours
         q_i <= u_i <= Q,  x_ij in {0, 1}                             (6) capacity and domains

Run from the repository root:
    python -m src.exact_ilp
"""

import re
import tempfile
import time
from pathlib import Path

import pulp

from src.data_prep import load_customers, sample_instance

DEFAULT_TIME_LIMIT_S = 300


def build_model(instance):
    """Create the PuLP model. Returns (model, x, u)."""
    dist = instance["dist"]
    demand = instance["demand"]
    capacity = instance["capacity"]
    n = len(demand) - 1

    nodes = range(n + 1)           # V = {0, 1, ..., n}, node 0 is the depot
    customers = range(1, n + 1)    # C = {1, ..., n}
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    model = pulp.LpProblem("CVRP", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", arcs, cat="Binary")                    # x_ij: van drives i -> j
    u = {i: pulp.LpVariable(f"u_{i}", lowBound=demand[i], upBound=capacity)  # (6) q_i <= u_i <= Q
         for i in customers}

    # (1) Objective: total distance driven
    model += pulp.lpSum(dist[i][j] * x[(i, j)] for (i, j) in arcs)

    for i in customers:
        # (2) exactly one arc leaves customer i
        model += pulp.lpSum(x[(i, j)] for j in nodes if j != i) == 1, f"leave_{i}"
        # (3) exactly one arc enters customer i
        model += pulp.lpSum(x[(j, i)] for j in nodes if j != i) == 1, f"enter_{i}"

    # (4) every van that leaves the depot comes back
    model += (pulp.lpSum(x[(0, j)] for j in customers)
              == pulp.lpSum(x[(i, 0)] for i in customers)), "depot_balance"

    # (5) MTZ: if the van goes i -> j, the load after j is at least the load after i plus q_j.
    #     When x_ij = 0 the "- Q" term switches the constraint off.
    for i in customers:
        for j in customers:
            if i != j:
                model += u[j] >= u[i] + demand[j] - capacity * (1 - x[(i, j)]), f"mtz_{i}_{j}"

    return model, x, u


def extract_routes(x, n):
    """Follow the arcs with x_ij = 1 from the depot to turn the solution into routes."""
    next_node = {}
    first_stops = []
    for (i, j), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            if i == 0:
                first_stops.append(j)
            else:
                next_node[i] = j

    routes = []
    for start in sorted(first_stops):
        route = [0, start]
        while route[-1] != 0:
            route.append(next_node[route[-1]])
            if len(route) > n + 2:   # safety stop: should never happen if constraint (5) works
                raise RuntimeError("route does not return to the depot")
        routes.append(route)
    return routes


def read_cbc_log(log_path):
    """Read the solver's final result line, lower bound and gap from the CBC log (if present)."""
    text = Path(log_path).read_text(errors="ignore")
    result = re.search(r"Result - (.+)", text)
    lower_bound = re.search(r"Lower bound:\s+([-+0-9.eE]+)", text)
    return {
        "cbc_result": result.group(1).strip() if result else None,
        "best_bound": float(lower_bound.group(1)) if lower_bound else None,
    }


def solve_cvrp_ilp(instance, time_limit=DEFAULT_TIME_LIMIT_S, verbose=False):
    """Solve the instance with CBC. Returns a dictionary with routes, distance, status and runtime.

    status is "optimal" when CBC proved optimality, "feasible" when the time limit stopped it
    with a solution that is not proven optimal, and "no_solution" otherwise.
    """
    n = len(instance["demand"]) - 1
    model, x, _ = build_model(instance)

    log_path = Path(tempfile.gettempdir()) / f"cbc_{instance['name']}.log"
    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit, logPath=str(log_path))

    start = time.perf_counter()
    model.solve(solver)
    runtime = time.perf_counter() - start

    log = read_cbc_log(log_path)
    has_solution = model.sol_status in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible)
    proven_optimal = (model.sol_status == pulp.LpSolutionOptimal
                      and log["cbc_result"] is not None
                      and log["cbc_result"].startswith("Optimal"))

    if not has_solution:
        return {"method": "ILP", "status": "no_solution", "routes": [], "distance_km": None,
                "runtime_s": runtime, "best_bound": log["best_bound"], "gap_pct": None}

    objective = pulp.value(model.objective)
    best_bound = objective if proven_optimal else log["best_bound"]
    gap_pct = None
    if best_bound is not None and objective > 0:
        gap_pct = 100 * (objective - best_bound) / objective

    return {
        "method": "ILP",
        "status": "optimal" if proven_optimal else "feasible",
        "routes": extract_routes(x, n),
        "distance_km": objective,
        "runtime_s": runtime,
        "best_bound": best_bound,
        "gap_pct": gap_pct,
    }


def main():
    customers = load_customers()
    instance = sample_instance(customers, n=8, seed=1)
    print(f"Instance {instance['name']}: {len(instance['customer_ids'])} customers, "
          f"Q = {instance['capacity']:.0f} kg, total demand = {sum(instance['demand']):.1f} kg")

    model, _, _ = build_model(instance)
    print(f"Model size: {len(model.variables())} variables, {len(model.constraints)} constraints")

    result = solve_cvrp_ilp(instance)
    print(f"Status: {result['status']} | distance: {result['distance_km']:.2f} km | "
          f"runtime: {result['runtime_s']:.2f} s | gap: {result['gap_pct']:.2f} %")

    for k, route in enumerate(result["routes"], start=1):
        load = sum(instance["demand"][node] for node in route)
        km = sum(instance["dist"][a][b] for a, b in zip(route, route[1:]))
        kaggle_ids = [instance["customer_ids"][node - 1] for node in route[1:-1]]
        print(f"  Van {k}: nodes {route} | Kaggle customers {kaggle_ids} | "
              f"load {load:.1f} kg | {km:.2f} km")


if __name__ == "__main__":
    main()
