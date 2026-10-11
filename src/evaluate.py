"""Solution evaluation: route distance and feasibility checker.

Owner: Luminda De Silva (Member 2)

Every method (ILP, greedy, simulated annealing) is judged by the SAME functions in this file,
so the comparison is fair and no method grades its own work.

A solution is a list of routes. Each route is a list of node numbers that starts and ends at
the depot (node 0), for example [[0, 3, 1, 0], [0, 2, 0]]. Node numbers follow the instance
format created by src/data_prep.make_instance().

Run from the repository root (runs a small hand-made test):
    python -m src.evaluate
"""

import math

TOLERANCE = 1e-6   # allow tiny floating-point rounding when comparing loads with Q


def route_distance(route, dist):
    """Length of one route in km: sum of d_ij over consecutive stops."""
    return sum(dist[a][b] for a, b in zip(route, route[1:]))


def total_distance(routes, dist):
    """Objective value Z: total km driven by all vans."""
    return sum(route_distance(route, dist) for route in routes)


def route_load(route, demand):
    """Total order weight (kg) carried on one route. The depot has demand 0."""
    return sum(demand[node] for node in route)


def check_feasible(routes, instance):
    """Return a list of problems. An empty list means the solution is feasible.

    Checks the CVRP constraints:
      - every route starts and ends at the depot and does not pass through it in the middle,
      - every customer 1..n is visited exactly once (no missing, no duplicates, no unknown nodes),
      - no route carries more than the capacity Q.
    """
    demand = instance["demand"]
    capacity = instance["capacity"]
    n = len(demand) - 1
    problems = []

    visits = {}
    for k, route in enumerate(routes, start=1):
        if len(route) < 3:
            problems.append(f"route {k} serves no customer: {route}")
            continue
        if route[0] != 0 or route[-1] != 0:
            problems.append(f"route {k} does not start and end at the depot: {route}")
        for node in route[1:-1]:
            if node == 0:
                problems.append(f"route {k} returns to the depot in the middle: {route}")
            elif not 1 <= node <= n:
                problems.append(f"route {k} has an unknown node {node}")
            else:
                visits[node] = visits.get(node, 0) + 1

        load = sum(demand[node] for node in route if 0 <= node <= n)
        if load > capacity + TOLERANCE:
            problems.append(f"route {k} is overloaded: {load:.1f} kg > Q = {capacity:.1f} kg")

    for customer in range(1, n + 1):
        count = visits.get(customer, 0)
        if count == 0:
            problems.append(f"customer node {customer} is not visited")
        elif count > 1:
            problems.append(f"customer node {customer} is visited {count} times")

    return problems


def summarize(routes, instance):
    """Key numbers for the results tables, computed the same way for every method."""
    problems = check_feasible(routes, instance)
    loads = [route_load(route, instance["demand"]) for route in routes]
    summary = {
        "distance_km": float(total_distance(routes, instance["dist"])),
        "n_routes": len(routes),
        "min_routes_needed": math.ceil(sum(instance["demand"]) / instance["capacity"] - TOLERANCE),
        "max_load_kg": float(max(loads)) if loads else 0.0,
        "feasible": not problems,
        "n_problems": len(problems),
    }
    # Assumption A6: volume is not a model constraint, it is only reported after solving.
    if "volume" in instance:
        summary["max_volume_dm3"] = float(max(route_load(route, instance["volume"]) for route in routes)) if routes else 0.0
    return summary


def make_toy_instance():
    """A tiny instance whose answers can be checked by hand (straight-line distances).

        3 (0,4) ---- 2 (3,4)
        |            |
        0 (0,0) ---- 1 (3,0) ---- 4 (6,0)

    Sides are 3 and 4 units, so the route 0-1-2-3-0 has length 3 + 4 + 3 + 4 = 14,
    and the route 0-4-0 has length 6 + 6 = 12.
    """
    points = [(0, 0), (3, 0), (3, 4), (0, 4), (6, 0)]
    dist = [[math.dist(p, q) for q in points] for p in points]
    return {
        "name": "toy",
        "demand": [0, 40, 30, 20, 50],   # kg, node 0 is the depot
        "capacity": 100,
        "dist": dist,
    }


def main():
    toy = make_toy_instance()

    good = [[0, 1, 2, 3, 0], [0, 4, 0]]
    print("Feasible solution:", good)
    print("  distance:", total_distance(good, toy["dist"]), "(expected 26 = 14 + 12)")
    print("  problems:", check_feasible(good, toy))
    print("  summary:", summarize(good, toy))
    assert total_distance(good, toy["dist"]) == 26
    assert check_feasible(good, toy) == []

    bad_cases = {
        "overloaded (140 kg > 100 kg)": [[0, 1, 2, 3, 4, 0]],
        "customers 3 and 4 missing": [[0, 1, 2, 0]],
        "customer 2 visited twice": [[0, 1, 2, 0], [0, 2, 3, 4, 0]],
        "route does not start at depot": [[1, 2, 3, 0], [0, 4, 0]],
    }
    for label, routes in bad_cases.items():
        problems = check_feasible(routes, toy)
        print(f"\nInfeasible solution ({label}): {routes}")
        for problem in problems:
            print("  -", problem)
        assert problems, f"the checker missed: {label}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
