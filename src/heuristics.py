"""Heuristic method, part 1: capacity-aware nearest-neighbour construction (greedy).

Owner: Luminda De Silva (Member 2)

The greedy route builder:
    1. Start a new van at the depot with an empty load.
    2. Go to the NEAREST unvisited customer whose order still fits in the van (load + q_j <= Q).
    3. Repeat step 2 until no remaining order fits, then return to the depot.
    4. If customers are still unvisited, start another van (step 1).

It is fast and always feasible, but short-sighted: each step only looks at the next stop,
so it often makes long "return" trips at the end of a route. Simulated annealing (added next)
starts from this solution and improves it.

Run from the repository root:
    python -m src.heuristics
"""

import time

from src.evaluate import check_feasible, make_toy_instance, summarize


def nearest_neighbour(instance):
    """Build routes with the capacity-aware nearest-neighbour rule. Returns a list of routes."""
    dist = instance["dist"]
    demand = instance["demand"]
    capacity = instance["capacity"]
    n = len(demand) - 1

    too_big = [c for c in range(1, n + 1) if demand[c] > capacity]
    if too_big:
        raise ValueError(f"customer nodes {too_big} have an order larger than Q, no solution exists")

    unvisited = set(range(1, n + 1))
    routes = []
    while unvisited:
        route = [0]
        load = 0.0
        current = 0
        while True:
            fits = [c for c in unvisited if load + demand[c] <= capacity]
            if not fits:
                break
            # nearest customer that fits; ties broken by the smaller node number (deterministic)
            nearest = min(fits, key=lambda c: (dist[current][c], c))
            route.append(nearest)
            load += demand[nearest]
            unvisited.remove(nearest)
            current = nearest
        route.append(0)
        routes.append(route)
    return routes


def solve_greedy(instance):
    """Run the greedy heuristic and return results in the same format as the ILP."""
    start = time.perf_counter()
    routes = nearest_neighbour(instance)
    runtime = time.perf_counter() - start
    return {
        "method": "Greedy",
        "status": "heuristic",
        "routes": routes,
        "distance_km": summarize(routes, instance)["distance_km"],
        "runtime_s": runtime,
    }


def main():
    # 1. Hand-checkable toy example (see make_toy_instance in src/evaluate.py), Q = 100 kg.
    #    From 0 the nearest is 1 (3 km, load 40). From 1 the nearest is 4 (3 km, load 90).
    #    Nothing else fits, so return to 0: route [0, 1, 4, 0] = 3 + 3 + 6 = 12.
    #    New van: nearest is 3 (4 km, load 20), then 2 (3 km, load 50), back to 0 (5 km):
    #    route [0, 3, 2, 0] = 12. Total = 24.
    toy = make_toy_instance()
    toy_routes = nearest_neighbour(toy)
    print("Toy instance routes:", toy_routes, "| distance:", summarize(toy_routes, toy)["distance_km"],
          "(expected [[0, 1, 4, 0], [0, 3, 2, 0]] and 24.0)")
    assert toy_routes == [[0, 1, 4, 0], [0, 3, 2, 0]]
    assert check_feasible(toy_routes, toy) == []

    # 2. All 87 real customers, Q = 1,000 kg.
    from src.data_prep import depot_location, load_customers, make_instance
    customers = load_customers()
    full = make_instance(customers, depot_location(customers), name="all_87")
    result = solve_greedy(full)
    print(f"\nAll 87 customers: {result['distance_km']:.2f} km with {len(result['routes'])} vans "
          f"in {result['runtime_s'] * 1000:.1f} ms")
    print("Summary:", summarize(result["routes"], full))
    for k, route in enumerate(result["routes"], start=1):
        load = sum(full["demand"][node] for node in route)
        print(f"  Van {k:2d}: {len(route) - 2:2d} customers, load {load:6.1f} kg")
    assert check_feasible(result["routes"], full) == []
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
