"""Data preparation for the delivery optimization (CVRP) project.

Owner: M.Z.M. Ruzbihan (Member 1)

This module turns the raw Kaggle workbook into the inputs of the optimization model:
    - a clean table of the 87 real customers,
    - one depot (assumption A1: centroid of all customers),
    - a haversine distance matrix d_ij in km (assumption A2),
    - an "instance" dictionary that both the exact and the heuristic method use.

Run from the repository root:
    python -m src.data_prep
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "Customer Data.xlsx"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "customers_clean.csv"

EARTH_RADIUS_KM = 6371.0
VEHICLE_CAPACITY_KG = 1000.0   # assumption A3: payload of a light commercial van
VAN_VOLUME_DM3 = 3000.0        # assumption A6: only checked after solving

COLUMN_NAMES = {
    "CUSTOMER ID": "customer_id",
    "LATITUDE": "lat",
    "LONGITUDE": "lon",
    "PRIORITY": "priority",
    "OPERATING SLOTS": "time_slot",
    "Order size (weight in kg)": "weight_kg",
    "Order size (volume in dm^3)": "volume_dm3",
}


def load_raw(path=RAW_PATH):
    """Read the workbook exactly as it is on Kaggle (no cleaning)."""
    return pd.read_excel(path, sheet_name="Customer Data")


def clean_customers(raw):
    """Keep only real customer rows and give every column a usable type.

    pandas reads 25,734 rows, but only 87 have a CUSTOMER ID. The other 25,647 rows hold only
    an OPERATING SLOTS value or are empty. A row is a real customer only if CUSTOMER ID is filled.
    """
    df = raw.rename(columns=COLUMN_NAMES)
    df = df[df["customer_id"].notna()].copy()

    df["customer_id"] = df["customer_id"].astype(int)
    df["priority"] = df["priority"].astype(int)
    # Remove floating-point noise such as 43.10000000000001
    df["weight_kg"] = df["weight_kg"].round(1)
    df["volume_dm3"] = df["volume_dm3"].round(1)

    df = df.sort_values("customer_id").reset_index(drop=True)

    # Checks that must hold for the data described in our report
    assert len(df) == 87, f"expected 87 customers, found {len(df)}"
    assert df["customer_id"].is_unique, "customer IDs are not unique"
    assert df[["lat", "lon", "weight_kg", "volume_dm3"]].notna().all().all(), "missing numeric values"
    assert (df["weight_kg"] > 0).all(), "every order must have a positive weight"
    return df


def depot_location(customers):
    """Assumption A1: the depot is the centroid (mean lat, mean lon) of ALL customers.

    It is computed from all 87 customers, so every subset instance uses the same depot.
    """
    return float(customers["lat"].mean()), float(customers["lon"].mean())


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on the Earth, in km (assumption A2)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def build_distance_matrix(lats, lons):
    """d[i][j] = haversine distance from node i to node j. Node 0 is the depot."""
    n_nodes = len(lats)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist[i, j] = haversine_km(lats[i], lons[i], lats[j], lons[j])
    return dist


def make_instance(customers, depot, capacity=VEHICLE_CAPACITY_KG, name="instance"):
    """Build the model inputs for a set of customers.

    Node 0 is the depot and nodes 1..n are the customers in the order of `customers`.
    Routes produced by both methods use these node numbers, e.g. [[0, 3, 1, 0], [0, 2, 0]].
    """
    lats = [depot[0]] + customers["lat"].tolist()
    lons = [depot[1]] + customers["lon"].tolist()
    return {
        "name": name,
        "customer_ids": customers["customer_id"].tolist(),   # original Kaggle IDs of nodes 1..n
        "lats": lats,
        "lons": lons,
        "demand": [0.0] + customers["weight_kg"].tolist(),   # q_i, depot has no demand
        "volume": [0.0] + customers["volume_dm3"].tolist(),
        "capacity": capacity,                                 # Q
        "dist": build_distance_matrix(lats, lons),            # d_ij
    }


def sample_instance(customers, n, seed, capacity=VEHICLE_CAPACITY_KG):
    """Assumption A7: a reproducible random subset of n customers (same seed -> same subset)."""
    rng = np.random.default_rng(seed)
    chosen_rows = np.sort(rng.choice(len(customers), size=n, replace=False))
    subset = customers.iloc[chosen_rows].reset_index(drop=True)
    return make_instance(subset, depot_location(customers), capacity, name=f"n{n}_seed{seed}")


def load_customers():
    """Load the cleaned customer table, creating it from the raw workbook if needed."""
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    customers = clean_customers(load_raw())
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(PROCESSED_PATH, index=False)
    return customers


def main():
    raw = load_raw()
    print(f"Raw sheet: {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"Rows without a CUSTOMER ID (filler rows): {raw['CUSTOMER ID'].isna().sum()}")

    customers = clean_customers(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(PROCESSED_PATH, index=False)
    print(f"\nClean customers: {len(customers)} rows -> saved to {PROCESSED_PATH.relative_to(REPO_ROOT)}")
    print(customers.head().to_string(index=False))

    print("\nSummary of the real data:")
    print(customers[["lat", "lon", "weight_kg", "volume_dm3"]].describe().round(3).to_string())
    print("Priority counts:", customers["priority"].value_counts().sort_index().to_dict())
    print("Customers without a time slot:", customers["time_slot"].isna().sum())
    total_weight = customers["weight_kg"].sum()
    print(f"Total weight: {total_weight:.1f} kg | total volume: {customers['volume_dm3'].sum():.1f} dm3")
    print(f"Minimum number of vans at Q = {VEHICLE_CAPACITY_KG:.0f} kg: {math.ceil(total_weight / VEHICLE_CAPACITY_KG)}")

    depot = depot_location(customers)
    print(f"\nDepot (assumed centroid): lat {depot[0]:.4f}, lon {depot[1]:.4f}")

    full = make_instance(customers, depot, name="all_87")
    dist = full["dist"]
    print(f"Distance matrix: {dist.shape[0]} x {dist.shape[1]} (depot + {len(full['customer_ids'])} customers)")
    print(f"  symmetric: {np.allclose(dist, dist.T)} | zero diagonal: {np.all(np.diag(dist) == 0)}")
    off_diagonal = dist[~np.eye(len(dist), dtype=bool)]
    print(f"  shortest {off_diagonal.min():.2f} km | longest {off_diagonal.max():.2f} km")
    print(f"  depot -> customer {full['customer_ids'][0]}: {dist[0, 1]:.2f} km")
    # Sanity check of the formula: 1 degree of latitude is about 111.19 km when R = 6371 km
    print(f"  check: 1 degree of latitude = {haversine_km(0, 0, 1, 0):.2f} km (expected 111.19)")

    small = sample_instance(customers, n=8, seed=1)
    again = sample_instance(customers, n=8, seed=1)
    print(f"\nSample instance {small['name']}: customers {small['customer_ids']}")
    print(f"  same seed gives same subset: {small['customer_ids'] == again['customer_ids']}")


if __name__ == "__main__":
    main()
