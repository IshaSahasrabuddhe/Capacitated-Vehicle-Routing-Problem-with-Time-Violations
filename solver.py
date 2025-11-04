import math
import random
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import io


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


# ---------------------------
# Data utils
# ---------------------------

def make_template_dataframe():
    return pd.DataFrame({
        "id": ["A", "B", "C"],
        "x": [10, -5, 15],
        "y": [4, -12, 8],
        "demand": [1, 2, 1],
        "tw_start": [0, 10, 5],
        "tw_end": [50, 30, 20],
        "service": [2, 3, 1],
    })


def parse_uploaded_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file.name if hasattr(file, "name") else file)
    required = {"id", "x", "y", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for opt in ["tw_start", "tw_end", "service"]:
        if opt not in df.columns:
            df[opt] = 0 if opt != "tw_end" else 999999

    df["id"] = df["id"].astype(str)
    for col in ["x", "y", "demand", "tw_start", "tw_end", "service"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def generate_random_instance(
    n_clients=15,
    n_vehicles=4,
    capacity=7,
    spread=10,           # smaller area = closer stops
    demand_min=1,
    demand_max=3,
    seed=42,
):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-spread, spread, size=n_clients)
    ys = rng.uniform(-spread, spread, size=n_clients)
    demands = rng.integers(demand_min, demand_max + 1, size=n_clients)

    # Wider time windows (30–45 minutes)
    tw_start = rng.integers(0, 40, size=n_clients)
    tw_end = tw_start + rng.integers(30, 45, size=n_clients)

    # Service time fixed to 1 minute
    #service = np.ones(n_clients, dtype=int)
    # Service time between 2 and 3 minutes (inclusive)
    service = rng.integers(2, 4, size=n_clients)


    df = pd.DataFrame({
        "id": [f"C{i+1}" for i in range(n_clients)],
        "x": xs,
        "y": ys,
        "demand": demands,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "service": service
    })
    return df


# ---------------------------
# Geometry helpers
# ---------------------------

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def total_distance(points: List[Tuple[float, float]]) -> float:
    return sum(euclid(points[i], points[i + 1]) for i in range(len(points) - 1))


# ---------------------------
# Time-window aware clustering
# ---------------------------

def tw_aware_clusters(df: pd.DataFrame, depot: Tuple[float, float],
                      n_vehicles: int, capacity: float) -> List[List[int]]:
    dx = df["x"].values - depot[0]
    dy = df["y"].values - depot[1]
    ang = np.arctan2(dy, dx)

    distances = np.sqrt(dx**2 + dy**2)
    tw_urgency = df["tw_end"].values / (distances + 1.0)
    order = np.lexsort((tw_urgency, ang))

    clusters = [[] for _ in range(n_vehicles)]
    loads = [0.0] * n_vehicles
    v = 0

    for idx in order:
        d = float(df.loc[idx, "demand"])
        if loads[v] + d > capacity and v < n_vehicles - 1:
            v += 1
        clusters[v].append(int(idx))
        loads[v] += d

    return clusters


# ---------------------------
# Schedule computation
# ---------------------------

def compute_schedule_for_route(route_idxs: List[int], depot: Tuple[float, float],
                               df: pd.DataFrame, speed: float = 1.0) -> Dict:
    arrivals, departures = [], []
    t = 0.0
    prev = depot
    lateness_count = total_lateness = max_lateness = 0.0

    for idx in route_idxs:
        cur = (float(df.loc[idx, "x"]), float(df.loc[idx, "y"]))
        travel = euclid(prev, cur) / max(speed, 1e-9)
        arrival = t + travel
        tw_s, tw_e = float(df.loc[idx, "tw_start"]), float(df.loc[idx, "tw_end"])

        arrival_eff = max(arrival, tw_s)
        lateness = max(0.0, arrival_eff - tw_e)

        if lateness > 0:
            lateness_count += 1
            total_lateness += lateness
            max_lateness = max(max_lateness, lateness)

        depart = arrival_eff + float(df.loc[idx, "service"])
        arrivals.append(arrival_eff)
        departures.append(depart)
        t = depart
        prev = cur

    return {
        "arrivals": arrivals,
        "departures": departures,
        "lateness_count": int(lateness_count),
        "total_lateness": float(total_lateness),
        "max_lateness": float(max_lateness),
        "feasible": lateness_count == 0
    }


# ---------------------------
# TW-prioritized insertion heuristic
# ---------------------------

def build_route_by_insertion_tw(df: pd.DataFrame, idxs: List[int],
                                depot: Tuple[float, float], speed: float = 1.0) -> List[int]:
    if not idxs:
        return []
    route, remaining = [], set(idxs)

    def urgency_score(i):
        dist = euclid(depot, (df.loc[i, "x"], df.loc[i, "y"]))
        tw_e = float(df.loc[i, "tw_end"])
        return tw_e / (dist + 1.0)

    first = min(remaining, key=urgency_score)
    route.append(first)
    remaining.remove(first)

    while remaining:
        best_choice = None
        remaining_sorted = sorted(remaining, key=urgency_score)

        for client in remaining_sorted:
            for pos in range(len(route) + 1):
                candidate = route[:pos] + [client] + route[pos:]
                pts = [depot] + [(float(df.loc[i, "x"]), float(df.loc[i, "y"])) for i in candidate] + [depot]
                dist = total_distance(pts)
                sched = compute_schedule_for_route(candidate, depot, df, speed)
                lateness_penalty = sched["total_lateness"] * 8000.0
                cost = dist + lateness_penalty

                if best_choice is None or cost < best_choice[2]:
                    best_choice = (client, pos, cost)
        client, pos, _ = best_choice
        route.insert(pos, client)
        remaining.remove(client)

    return route


# ---------------------------
# Local search (2-opt + Or-opt)
# ---------------------------

def two_opt_tw(route, df, depot, speed=1.0, max_iter=300, lateness_weight=40000.0):
    if len(route) <= 2:
        return route[:]

    def route_cost(r):
        pts = [depot] + [(float(df.loc[i, "x"]), float(df.loc[i, "y"])) for i in r] + [depot]
        dist = total_distance(pts)
        sched = compute_schedule_for_route(r, depot, df, speed)
        return dist + lateness_weight * sched["total_lateness"]

    best = route[:]
    best_cost = route_cost(best)
    n = len(route)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                if i == 0 and k == n - 1:
                    continue
                candidate = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                c_cost = route_cost(candidate)
                if c_cost < best_cost - 1e-6:
                    best, best_cost, improved = candidate, c_cost, True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def or_opt_tw(route, df, depot, speed=1.0, max_iter=100, lateness_weight=40000.0):
    if len(route) <= 2:
        return route[:]

    def route_cost(r):
        pts = [depot] + [(float(df.loc[i, "x"]), float(df.loc[i, "y"])) for i in r] + [depot]
        dist = total_distance(pts)
        sched = compute_schedule_for_route(r, depot, df, speed)
        return dist + lateness_weight * sched["total_lateness"]

    best = route[:]
    best_cost = route_cost(best)
    n = len(route)

    for _ in range(max_iter):
        improved = False
        for length in [1, 2]:
            if length >= n:
                continue
            for i in range(n - length + 1):
                seg = best[i:i + length]
                rem = best[:i] + best[i + length:]
                for j in range(len(rem) + 1):
                    if j == i:
                        continue
                    cand = rem[:j] + seg + rem[j:]
                    c_cost = route_cost(cand)
                    if c_cost < best_cost - 1e-6:
                        best, best_cost, improved = cand, c_cost, True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best


# ---------------------------
# Multi-phase route optimizer
# ---------------------------

def build_route_for_cluster_tw(df, idxs, depot, speed=1.0):
    if not idxs:
        return []
    route = build_route_by_insertion_tw(df, idxs, depot, speed)
    route = two_opt_tw(route, df, depot, speed)
    route = or_opt_tw(route, df, depot, speed)
    return route

# ---------------------------
# Redistribution helper: force-using empty vehicles
# ---------------------------
def redistribute_to_use_all_vehicles(routes: List[List[int]],
                                     df: pd.DataFrame,
                                     depot: Tuple[float, float],
                                     n_vehicles: int,
                                     capacity: float,
                                     speed: float = 1.0) -> List[List[int]]:
    """
    Iteratively create new routes on unused vehicles by extracting the most problematic
    client (highest lateness, or earliest tw_end) from the worst route, then rebuilding
    the two affected routes. Stop when we've used all vehicles or can't split further.
    """
    def route_lateness_per_client(route):
        # returns list of (client_idx, lateness, tw_end)
        if not route:
            return []
        sched = compute_schedule_for_route(route, depot, df, speed)
        arrivals = sched["arrivals"]  # arrival_eff for each client in route order
        res = []
        for pos, cli in enumerate(route):
            tw_e = float(df.loc[cli, "tw_end"])
            lateness = max(0.0, arrivals[pos] - tw_e)
            res.append((cli, lateness, tw_e))
        return res

    # copy to avoid mutating original reference
    routes = [r[:] for r in routes]
    used = sum(1 for r in routes if r)
    # ensure routes list has capacity for all vehicles
    if len(routes) < n_vehicles:
        routes += [[] for _ in range(n_vehicles - len(routes))]

    # set of empty vehicle indices available for splits
    def first_empty_index():
        for i, r in enumerate(routes):
            if not r:
                return i
        return None

    # loop: split until used == n_vehicles or can't split
    while used < n_vehicles:
        # choose route to split: route with largest total lateness (or largest total lateness_weighted)
        best_route_idx = None
        best_route_lateness = -1.0
        for i, r in enumerate(routes):
            if not r:
                continue
            sched = compute_schedule_for_route(r, depot, df, speed)
            if sched["total_lateness"] > best_route_lateness:
                best_route_lateness = sched["total_lateness"]
                best_route_idx = i

        # nothing to split
        if best_route_idx is None:
            break

        # compute per-client lateness in that route
        per_client = route_lateness_per_client(routes[best_route_idx])
        if not per_client:
            break

        # pick the client with largest lateness; fallback pick earliest tw_end (tightest window)
        per_client_sorted = sorted(per_client, key=lambda x: (-x[1], x[2]))
        cli_to_move, cli_lateness, _ = per_client_sorted[0]

        # If there is no lateness at all, still consider moving the *tightest* deadline client
        if cli_lateness <= 0:
            # find earliest tw_end client
            per_client_sorted = sorted(per_client, key=lambda x: (x[2], -x[1]))
            cli_to_move = per_client_sorted[0][0]

        # if the client demand > capacity (we cannot move into a single-vehicle), break
        if float(df.loc[cli_to_move, "demand"]) > capacity:
            # cannot place this client alone on a vehicle; try next candidate
            alt = None
            for c, laten, tw in per_client_sorted[1:]:
                if float(df.loc[c, "demand"]) <= capacity:
                    alt = c
                    break
            if alt is None:
                break
            cli_to_move = alt

        # find an empty vehicle
        empty_idx = first_empty_index()
        if empty_idx is None:
            break

        # remove client from original route
        orig_route = routes[best_route_idx]
        if cli_to_move not in orig_route:
            # safety check
            break
        new_orig = [c for c in orig_route if c != cli_to_move]
        # rebuild both routes (optimize orders)
        rebuilt_orig = build_route_for_cluster_tw(df, new_orig, depot, speed) if new_orig else []
        rebuilt_new = build_route_for_cluster_tw(df, [cli_to_move], depot, speed)

        routes[best_route_idx] = rebuilt_orig
        routes[empty_idx] = rebuilt_new

        # update used count
        used = sum(1 for r in routes if r)

        # defensive: if we didn't create an additional non-empty route, break to avoid infinite loop
        if sum(1 for r in routes if r) <= used - 1:
            break

    # ensure we return exactly n_vehicles slots
    if len(routes) < n_vehicles:
        routes += [[] for _ in range(n_vehicles - len(routes))]
    return routes

# ---------------------------
# -----------------------------------------------------
# Helper: Redistribute workload across routes (balance)
# -----------------------------------------------------
def redistribute_workload(routes, df, depot, speed, capacity):
    """
    Balances routes by moving low-demand stops from overloaded routes
    to underutilized ones. Recomputes distances and loads.
    """
    import math

    # Calculate per-route load
    per_route_loads = [df.loc[r, "demand"].sum() if r else 0.0 for r in routes]
    avg_load = sum(per_route_loads) / max(1, len(per_route_loads))

    # Identify heavy and light routes
    overloaded = [i for i, l in enumerate(per_route_loads) if l > capacity * 0.9]
    underused = [i for i, l in enumerate(per_route_loads) if l < capacity * 0.5]

    # Try to move one or two smallest-demand customers from heavy → light
    for hi in overloaded:
        for li in underused:
            if not routes[hi]:
                continue

            # Sort heavy route by smallest demand
            sorted_by_demand = sorted(routes[hi], key=lambda idx: df.loc[idx, "demand"])

            for cust in sorted_by_demand[:2]:
                demand = df.loc[cust, "demand"]
                if per_route_loads[li] + demand <= capacity:
                    # Move stop from hi → li
                    routes[hi].remove(cust)
                    routes[li].append(cust)
                    per_route_loads[hi] -= demand
                    per_route_loads[li] += demand
                    break  # one transfer per underused route

    # Recompute distances for all routes
    per_route_dist = []
    for route in routes:
        if not route:
            per_route_dist.append(0.0)
            continue
        pts = [depot] + [(df.loc[i, "x"], df.loc[i, "y"]) for i in route] + [depot]
        dist = total_distance(pts)
        per_route_dist.append(dist)

    return routes, per_route_dist, per_route_loads

# ---------------------------
# Main solver
# ---------------------------

def solve_vrp_tw(df, depot=(0.0, 0.0), n_vehicles=4,
                 capacity=10, speed=1.0, force_all_vehicles=False) -> Dict:
    if len(df) == 0:
        return {
            "routes": [[] for _ in range(n_vehicles)],
            "total_distance": 0.0,
            "per_route_distance": [0.0] * n_vehicles,
            "assignments_table": pd.DataFrame(),
            "metrics": {}
        }

    # --- Step 1: Create initial clusters (time-window aware) ---
    clusters = tw_aware_clusters(df, depot, n_vehicles, capacity)

    # --- Step 2: Optionally force all vehicles to be used evenly ---
    if force_all_vehicles:
        all_clients = [i for cl in clusters for i in cl]
        clusters = [[] for _ in range(n_vehicles)]
        for i, idx in enumerate(all_clients):
            clusters[i % n_vehicles].append(idx)

    # --- Step 3: Build routes for each cluster ---
    routes, per_route_dist, per_route_loads = [], [], []
    total_late_count = total_late_time = max_late = 0.0

    for cl in clusters:
        if not cl:
            routes.append([])
            per_route_dist.append(0.0)
            per_route_loads.append(0.0)
            continue

        cluster_load = sum(df.loc[i, "demand"] for i in cl)
        if cluster_load <= capacity:
            chunks = [cl]
        else:
            # Split overloaded clusters into smaller chunks by time-window
            cl_sorted = sorted(cl, key=lambda i: df.loc[i, "tw_end"])
            chunks, current, load = [], [], 0
            for i in cl_sorted:
                d = df.loc[i, "demand"]
                if load + d > capacity and current:
                    chunks.append(current)
                    current, load = [i], d
                else:
                    current.append(i)
                    load += d
            if current:
                chunks.append(current)

        for chunk in chunks:
            route = build_route_for_cluster_tw(df, chunk, depot, speed)
            routes.append(route)

            pts = [depot] + [(df.loc[i, "x"], df.loc[i, "y"]) for i in route] + [depot]
            dist = total_distance(pts)
            per_route_dist.append(dist)
            per_route_loads.append(df.loc[route, "demand"].sum() if route else 0.0)

            sched = compute_schedule_for_route(route, depot, df, speed)
            total_late_count += sched["lateness_count"]
            total_late_time += sched["total_lateness"]
            max_late = max(max_late, sched["max_lateness"])

    # --- NEW SECTION: Redistribute workload before computing totals ---
    routes, per_route_dist, per_route_loads = redistribute_workload(routes, df, depot, speed, capacity)

    # --- Step 4: Compute total distance ---
    total_dist = sum(per_route_dist)

    # --- Step 5: Build assignment table for visualization ---
    rows = []
    for v, route in enumerate(routes):
        for seq, idx in enumerate(route, 1):
            rows.append({
                "vehicle": v + 1,
                "sequence": seq,
                "id": df.loc[idx, "id"],
                "x": float(df.loc[idx, "x"]),
                "y": float(df.loc[idx, "y"]),
                "demand": float(df.loc[idx, "demand"]),
            })
    assign_df = pd.DataFrame(rows).sort_values(["vehicle", "sequence"]).reset_index(drop=True)

    # --- Step 6: Time-window performance summary ---
    if total_late_count == 0:
        status = "OK"
    elif total_late_time < 300:
        status = "Minor Violations"
    else:
        status = "Violations"

    time_window_report = {
        "total_lateness_count": int(total_late_count),
        "total_lateness": round(total_late_time, 2),
        "max_lateness": round(max_late, 2),
        "status": status
    }

    # --- Step 7: Compile metrics ---
    metrics = {
        "vehicles_used": int(sum(1 for r in routes if r)),
        "total_distance": round(total_dist, 2),
        "per_route_distance": [round(d, 2) for d in per_route_dist],
        "per_route_load": [round(l, 2) for l in per_route_loads],
        "capacity": capacity,
        "time_window_report": time_window_report,
        "note": "Enhanced heuristic (TW-aware clustering → insertion → 2-opt → Or-opt). Auto lateness scaling + load redistribution."
    }

    # ✅ Convert NumPy values to native Python types
    metrics = convert_numpy(metrics)

    # --- Step 8: Return final structured result ---
    return {
        "routes": routes,
        "total_distance": total_dist,
        "per_route_distance": per_route_dist,
        "assignments_table": assign_df,
        "metrics": metrics,
    }


# ---------------------------
# Visualization
# ---------------------------

def plot_solution(df, sol, depot=(0.0, 0.0)):
    routes = sol["routes"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([depot[0]], [depot[1]], s=120, marker="s", label="Depot", zorder=6)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
    for v, route in enumerate(routes):
        if not route:
            continue
        c = colors[v % len(colors)]
        xs = [depot[0]] + [df.loc[i, "x"] for i in route] + [depot[0]]
        ys = [depot[1]] + [df.loc[i, "y"] for i in route] + [depot[1]]
        ax.plot(xs, ys, "-", lw=2, color=c, alpha=0.9, label=f"Vehicle {v+1}")
        ax.scatter(xs[1:-1], ys[1:-1], s=40, color=c, zorder=5)
        for k, idx in enumerate(route, 1):
            tw_s, tw_e = int(df.loc[idx, "tw_start"]), int(df.loc[idx, "tw_end"])
            ax.text(df.loc[idx, "x"], df.loc[idx, "y"], str(k),
                    fontsize=8, ha="center", va="center",
                    color="white", bbox=dict(boxstyle="circle,pad=0.2", fc=c, ec="none", alpha=0.8))
            ax.annotate(f"{tw_s}-{tw_e}", (df.loc[idx, "x"], df.loc[idx, "y"]),
                        textcoords="offset points", xytext=(6, -6), fontsize=7, color="black", alpha=0.7)

    ax.set_title("VRPTW Routes (Improved Heuristic)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
