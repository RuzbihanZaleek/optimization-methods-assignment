"""Heuristic method: greedy nearest-neighbour construction + Simulated Annealing improvement.

Owner: U.D.L.L De Silva (Member 2)

Part 1, greedy (nearest neighbour):
    1. Start a new van at the depot with an empty load.
    2. Go to the NEAREST unvisited customer whose order still fits in the van (load + q_j <= Q).
    3. Repeat step 2 until no remaining order fits, then return to the depot.
    4. If customers are still unvisited, start another van (step 1).
    Fast and always feasible, but short-sighted: each step only looks at the next stop.

Part 2, Simulated Annealing (SA), starting from the greedy solution:
    Repeat for a fixed number of iterations:
        - make one small random change ("move"): relocate, swap or 2-opt,
        - reject it if a van would be overloaded,
        - delta = new distance - current distance,
        - accept if delta <= 0, otherwise accept with probability exp(-delta / T),
        - lower the temperature: T = cooling * T,
        - remember the best solution ever seen.
    A high T early on lets the search accept worse moves and escape local optima;
    a low T at the end makes it behave like greedy hill-climbing.

Run from the repository root:
    python -m src.heuristics
"""

import math
import random
import time

from src.evaluate import check_feasible, make_toy_instance, route_distance, summarize

# Default SA settings. The temperature falls geometrically from T0 to T0 * FINAL_TEMP_RATIO.
# The iteration budget was chosen once in a pilot run on all 87 customers (5 seeds each):
# mean distance 2025 km at 50k, 1957 km at 100k, 1903 km at 300k, 1860 km at 1M iterations
# (about 9 s per run). It is then kept fixed for every experiment.
SA_ITERATIONS = 1_000_000
SA_FINAL_TEMP_RATIO = 0.001
SA_START_ACCEPT_PROB = 0.5     # T0 is set so an average worse move is accepted with this probability
HISTORY_EVERY = 500            # record the cost every this many iterations (for convergence plots)


# ----------------------------------------------------------------------------------------------
# Part 1: greedy construction
# ----------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------
# Part 2: simulated annealing
# ----------------------------------------------------------------------------------------------
# Inside SA a route is stored WITHOUT the depot, e.g. [3, 1] means 0 -> 3 -> 1 -> 0.
# This makes the moves simpler. The depot is added back before returning the solution.

def tour_length(customers, dist):
    """Distance of the route depot -> customers -> depot."""
    return route_distance([0] + customers + [0], dist)


def random_move(routes, rng):
    """Propose one random neighbour. Returns (move name, {route index: new customer list}).

    Only the routes that change are returned, so the change in distance can be computed
    from those routes alone. An empty new list means that route disappears (one van fewer).
    """
    move = rng.choice(("relocate", "swap", "2-opt"))

    if move == "relocate":
        # take one customer out of route a and insert it at any position of route b (b may equal a)
        a = rng.randrange(len(routes))
        b = rng.randrange(len(routes))
        i = rng.randrange(len(routes[a]))
        customer = routes[a][i]
        new_a = routes[a][:i] + routes[a][i + 1:]
        if a == b:
            j = rng.randrange(len(new_a) + 1)
            return move, {a: new_a[:j] + [customer] + new_a[j:]}
        j = rng.randrange(len(routes[b]) + 1)
        return move, {a: new_a, b: routes[b][:j] + [customer] + routes[b][j:]}

    if move == "swap":
        # exchange two customers (in the same route or in two different routes)
        a = rng.randrange(len(routes))
        b = rng.randrange(len(routes))
        i = rng.randrange(len(routes[a]))
        j = rng.randrange(len(routes[b]))
        if a == b:
            new_a = routes[a][:]
            new_a[i], new_a[j] = new_a[j], new_a[i]
            return move, {a: new_a}
        new_a, new_b = routes[a][:], routes[b][:]
        new_a[i], new_b[j] = routes[b][j], routes[a][i]
        return move, {a: new_a, b: new_b}

    # 2-opt: reverse the order of a segment inside one route (removes crossing lines)
    a = rng.randrange(len(routes))
    i, j = sorted(rng.sample(range(len(routes[a]) + 1), 2)) if len(routes[a]) > 1 else (0, 1)
    return move, {a: routes[a][:i] + routes[a][i:j][::-1] + routes[a][j:]}


def fits_capacity(changed, demand, capacity):
    """True if every changed route respects the capacity Q."""
    return all(sum(demand[c] for c in customers) <= capacity + 1e-9 for customers in changed.values())


def delta_of(routes, changed, dist):
    """Change in total distance if the changed routes replace the old ones."""
    old = sum(tour_length(routes[k], dist) for k in changed)
    new = sum(tour_length(customers, dist) for customers in changed.values() if customers)
    return new - old


def estimate_start_temperature(routes, instance, rng, samples=500):
    """Pick T0 so that an average worse move is accepted with probability SA_START_ACCEPT_PROB.

    From exp(-avg_delta / T0) = p we get T0 = -avg_delta / ln(p).
    """
    worse = []
    for _ in range(samples):
        _, changed = random_move(routes, rng)
        if fits_capacity(changed, instance["demand"], instance["capacity"]):
            delta = delta_of(routes, changed, instance["dist"])
            if delta > 0:
                worse.append(delta)
    average_worse = sum(worse) / len(worse) if worse else 1.0
    return -average_worse / math.log(SA_START_ACCEPT_PROB)


def simulated_annealing(instance, initial_routes, seed=0, iterations=SA_ITERATIONS,
                        final_temp_ratio=SA_FINAL_TEMP_RATIO):
    """Improve a feasible solution with SA. Returns (best routes, best distance, history, T0).

    history is a list of (iteration, temperature, current distance, best distance).
    """
    rng = random.Random(seed)   # own random generator: the same seed always gives the same result
    dist = instance["dist"]
    demand = instance["demand"]
    capacity = instance["capacity"]

    routes = [route[1:-1] for route in initial_routes]   # drop the depot inside SA
    current = sum(tour_length(r, dist) for r in routes)
    best_routes, best = [r[:] for r in routes], current

    start_temp = estimate_start_temperature(routes, instance, rng)
    cooling = final_temp_ratio ** (1 / iterations)        # so T ends at start_temp * final_temp_ratio
    temperature = start_temp
    history = [(0, temperature, current, best)]

    for iteration in range(1, iterations + 1):
        _, changed = random_move(routes, rng)

        if fits_capacity(changed, demand, capacity):
            delta = delta_of(routes, changed, dist)
            # Metropolis acceptance rule
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                for k, customers in changed.items():
                    routes[k] = customers
                routes = [r for r in routes if r]               # remove a van that became empty
                current += delta
                if current < best - 1e-9:
                    best_routes, best = [r[:] for r in routes], current

        temperature *= cooling
        if iteration % HISTORY_EVERY == 0:
            history.append((iteration, temperature, current, best))

    best_with_depot = [[0] + r + [0] for r in best_routes]
    return best_with_depot, best, history, start_temp


def solve_sa(instance, seed=0, iterations=SA_ITERATIONS):
    """Greedy start + SA. Returns results in the same format as the ILP and greedy methods.

    The runtime includes building the greedy start, because SA cannot run without it.
    """
    start = time.perf_counter()
    initial = nearest_neighbour(instance)
    routes, _, history, start_temp = simulated_annealing(instance, initial, seed=seed, iterations=iterations)
    runtime = time.perf_counter() - start
    return {
        "method": "SA",
        "status": "heuristic",
        "routes": routes,
        "distance_km": summarize(routes, instance)["distance_km"],   # recomputed by the shared judge
        "runtime_s": runtime,
        "seed": seed,
        "iterations": iterations,
        "start_temperature": start_temp,
        "history": history,
    }


def main():
    from src.data_prep import depot_location, load_customers, make_instance, sample_instance

    # 1. Greedy on the hand-checkable toy example (see make_toy_instance in src/evaluate.py), Q = 100 kg.
    #    From 0 the nearest is 1 (3 km, load 40). From 1 the nearest is 4 (3 km, load 90).
    #    Nothing else fits, so return to 0: route [0, 1, 4, 0] = 3 + 3 + 6 = 12.
    #    New van: nearest is 3 (4 km, load 20), then 2 (3 km, load 50), back to 0 (5 km):
    #    route [0, 3, 2, 0] = 12. Total = 24.
    toy = make_toy_instance()
    toy_routes = nearest_neighbour(toy)
    print("Toy instance greedy routes:", toy_routes, "| distance:", summarize(toy_routes, toy)["distance_km"],
          "(expected [[0, 1, 4, 0], [0, 3, 2, 0]] and 24.0)")
    assert toy_routes == [[0, 1, 4, 0], [0, 3, 2, 0]]
    assert check_feasible(toy_routes, toy) == []

    # 2. SA on a small real instance whose proven ILP optimum is 557.91 km (see notebook 01).
    small = sample_instance(load_customers(), n=10, seed=3, capacity=600)
    greedy_small = solve_greedy(small)
    sa_small = solve_sa(small, seed=0, iterations=20_000)
    print(f"\nInstance {small['name']} (Q = 600 kg, proven optimum 557.91 km):")
    print(f"  greedy {greedy_small['distance_km']:.2f} km | SA {sa_small['distance_km']:.2f} km "
          f"in {sa_small['runtime_s']:.2f} s")
    assert check_feasible(sa_small["routes"], small) == []
    assert sa_small["distance_km"] >= 557.91 - 0.01, "SA cannot beat a proven optimum: bug!"

    # 3. Greedy and SA on all 87 real customers, Q = 1,000 kg.
    customers = load_customers()
    full = make_instance(customers, depot_location(customers), name="all_87")
    greedy = solve_greedy(full)
    sa = solve_sa(full, seed=0)
    improvement = 100 * (greedy["distance_km"] - sa["distance_km"]) / greedy["distance_km"]
    print(f"\nAll 87 customers (Q = 1,000 kg):")
    print(f"  greedy: {greedy['distance_km']:.2f} km, {len(greedy['routes'])} vans")
    print(f"  SA:     {sa['distance_km']:.2f} km, {len(sa['routes'])} vans, {sa['runtime_s']:.1f} s, "
          f"T0 = {sa['start_temperature']:.2f}, {sa['iterations']} iterations")
    print(f"  SA improves the greedy solution by {improvement:.1f} %")
    print("  summary:", summarize(sa["routes"], full))
    assert check_feasible(sa["routes"], full) == []
    assert sa["distance_km"] <= greedy["distance_km"] + 1e-9

    # 4. Reproducibility: the same seed must give exactly the same answer.
    again = solve_sa(small, seed=0, iterations=20_000)
    print(f"\nSame seed, same result: {again['routes'] == sa_small['routes']}")
    assert again["routes"] == sa_small["routes"]

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
