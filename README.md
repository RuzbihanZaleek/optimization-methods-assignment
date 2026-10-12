# Delivery Optimization with Exact and Heuristic Methods

IT50802 Optimization Methods, MSc in Artificial Intelligence (SLIIT), group programming assignment.

| Member | Student ID |
|---|---|
| M.Z.M. Ruzbihan | MS26911458 |
| U.D.L.L De Silva | MS26903224 |

## Problem

A courier delivers orders of known weight from one depot to 87 real customers.
Each van carries at most 1,000 kg. We decide which van visits which customers,
and in what order, so that the total distance driven is as small as possible.
This is the Capacitated Vehicle Routing Problem (CVRP), which is NP-hard.

We solve it with:

1. **Exact method:** Integer Linear Programming (two-index formulation with MTZ constraints),
   solved by the CBC branch-and-bound solver through PuLP.
2. **Heuristic:** capacity-aware nearest-neighbour construction improved by Simulated Annealing.

We compare solution quality, runtime, scalability and feasibility.

## Data and assumptions

Real data (from the Kaggle file, see `data/README.md`): customer coordinates, order weight and volume.

Assumptions, because the dataset does not contain this information:

| ID | Assumption |
|---|---|
| A1 | One depot at the centroid of all 87 customers (lat 24.9670, lon 55.1540) |
| A2 | Distances are great-circle (haversine) km, not road distances |
| A3 | Van capacity Q = 1,000 kg (600 and 1,500 kg tested in a sensitivity experiment) |
| A4 | Unlimited identical vans; the number of routes is an output |
| A5 | Time windows and priorities are not modelled (24 of 87 time slots are blank, priority is undocumented) |
| A6 | Order volume is checked after solving against a 3,000 dm³ van volume |
| A7 | Exact-method instances are random customer subsets drawn with fixed seeds |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then download the dataset as described in `data/README.md`.

## Run

All commands are run from the repository root with the virtual environment active.

```powershell
# Data and methods (each module runs its own checks)
python -m src.data_prep               # clean the data, build the distance matrix
python -m src.evaluate                # feasibility checker on a hand-checked toy example
python -m src.exact_ilp               # solve an 8-customer instance exactly with the ILP (CBC)
python -m src.verify_exact            # ILP vs exhaustive search on 6 small instances (~1 min)
python -m src.heuristics              # greedy and simulated annealing tests
python -m src.benchmark               # A-n32-k5 loader check (published optimum 784)

# Experiments and figures
python -m src.run_comparison          # E1 + E2: ILP vs greedy vs SA, quality and scalability (~1 hour)
python -m src.run_heuristic_analysis  # E3-E5: capacity, SA robustness, benchmark (~15 min)
python -m src.plots                   # all result figures -> results/figures/

# Notebooks and appendix
jupyter notebook notebooks/01_data_and_model.ipynb          # data tour, formulation and ILP demo
jupyter notebook notebooks/02_heuristic_and_results.ipynb   # heuristic demo and all results
python scripts/make_code_appendix.py                        # plain-text code appendix -> report/code_appendix.txt
```

In VS Code, choose the `.venv` Python as the notebook kernel, otherwise `pulp` will not be found.
PuLP is pinned to 2.9.0 because it bundles the CBC solver; PuLP 4.0 no longer does.

## Reproducibility

- All random choices use fixed seeds (customer subsets: seeds 1-3; simulated annealing: seeds 0-9),
  so distances are identical on every run and every computer.
- E1/E2 were run on M1's laptop and E3-E5 on M2's laptop (see `results/tables/environment*.txt`).
  The heuristic distances were reproduced exactly on both machines; runtimes and time-limited ILP
  results depend on the machine.

## Repository layout

```
data/              dataset instructions (raw and processed files are not committed)
data/benchmark/    CVRPLIB benchmark A-n32-k5 (instance and published optimal solution)
src/               Python modules: data preparation, ILP, heuristics, checker, experiments, plots
scripts/           code-appendix generator
notebooks/         01: data and exact model, 02: heuristic and results
results/tables/    CSV outputs of every experiment
results/figures/   PNG figures used in the report and video
report/            plain-text code appendix (and the report)
```
