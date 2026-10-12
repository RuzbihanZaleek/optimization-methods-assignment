"""Benchmark instance A-n32-k5 (Augerat et al., CVRPLIB) for external validation of our methods.

Owner: U.D.L.L De Silva (Member 2)

Why: on our own data the ILP cannot prove optimality beyond about 12 customers, so we cannot
measure how good SA is on larger instances. A-n32-k5 (31 customers, Q = 100) has a published
optimal cost of 784, so it gives an outside yardstick.

CVRPLIB convention: distances are Euclidean and rounded to the nearest integer ("EUC_2D").
We check that our loader follows this convention by scoring the published optimal routes:
they must cost exactly 784 with our own evaluate.total_distance().

Files (from the PyVRP/VRPLIB repository, which mirrors CVRPLIB):
    data/benchmark/A-n32-k5.vrp   instance
    data/benchmark/A-n32-k5.sol   published optimal routes and cost

Run from the repository root:
    python -m src.benchmark
"""

import math

from src.data_prep import REPO_ROOT
from src.evaluate import check_feasible, total_distance

BENCHMARK_DIR = REPO_ROOT / "data" / "benchmark"


def euc_2d(p, q):
    """CVRPLIB/TSPLIB EUC_2D distance: Euclidean distance rounded to the nearest integer."""
    return float(int(math.dist(p, q) + 0.5))


def load_vrplib_instance(path):
    """Read a CVRPLIB .vrp file into our instance format (node 0 = depot)."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    capacity = None
    coords = {}
    demand = {}
    section = None
    for line in lines:
        if line.startswith("CAPACITY"):
            capacity = float(line.split(":")[1])
        elif line.startswith("NODE_COORD_SECTION"):
            section = "coords"
        elif line.startswith("DEMAND_SECTION"):
            section = "demand"
        elif line.startswith(("DEPOT_SECTION", "EOF")):
            section = None
        elif section == "coords":
            node, x, y = line.split()
            coords[int(node)] = (float(x), float(y))
        elif section == "demand":
            node, q = line.split()
            demand[int(node)] = float(q)

    # The file numbers nodes 1..32 with the depot as node 1; we use 0..31 with the depot as 0.
    file_nodes = sorted(coords)
    points = [coords[node] for node in file_nodes]
    return {
        "name": path.stem,
        "customer_ids": file_nodes[1:],
        "xs": [p[0] for p in points],
        "ys": [p[1] for p in points],
        "demand": [demand[node] for node in file_nodes],
        "capacity": capacity,
        "dist": [[euc_2d(p, q) for q in points] for p in points],
    }


def load_solution(path):
    """Read a CVRPLIB .sol file. Customer numbers in .sol files already match our 1..n numbering."""
    routes = []
    cost = None
    for line in path.read_text().splitlines():
        if line.startswith("Route"):
            customers = [int(c) for c in line.split(":")[1].split()]
            routes.append([0] + customers + [0])
        elif line.startswith("Cost"):
            cost = float(line.split()[1])
    return routes, cost


def load_benchmark(name="A-n32-k5"):
    """Return (instance, published optimal cost)."""
    instance = load_vrplib_instance(BENCHMARK_DIR / f"{name}.vrp")
    _, cost = load_solution(BENCHMARK_DIR / f"{name}.sol")
    return instance, cost


def main():
    instance, published_cost = load_benchmark()
    n = len(instance["demand"]) - 1
    print(f"{instance['name']}: {n} customers, Q = {instance['capacity']:.0f}, "
          f"total demand {sum(instance['demand']):.0f}, published optimum {published_cost:.0f}")

    routes, _ = load_solution(BENCHMARK_DIR / f"{instance['name']}.sol")
    our_cost = total_distance(routes, instance["dist"])
    problems = check_feasible(routes, instance)
    print(f"Published routes scored by our code: {our_cost:.0f} | feasible: {not problems}")
    assert problems == [], problems
    assert our_cost == published_cost, "distance convention does not match CVRPLIB"
    print("All checks passed: our loader and distance convention match CVRPLIB.")


if __name__ == "__main__":
    main()
