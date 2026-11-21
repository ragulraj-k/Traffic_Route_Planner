"""
Intelligent Traffic Route Planner  (single file version)
AI with Python mini project: BFS, Uniform Cost Search and A* on a simulated
Chennai road network, with a Streamlit dashboard.

IMPORTANT: the road distances, speeds and traffic levels are simulated for
academic demonstration. They are NOT real-time Chennai traffic data.

How to run (Windows):
    pip install streamlit networkx matplotlib pandas
    python traffic_planner.py            (opens the dashboard in your browser)
    -- or --
    streamlit run traffic_planner.py

Sections of this file, in order:
    1. Road network and traffic model
    2. Shared search helpers
    3. BFS            4. UCS            5. A*
    6. Alternative routes   7. Trip helpers   8. Algorithm comparison
    9. Visualization  10. Streamlit dashboard
"""

import copy
import heapq
import math
import random
import time as _time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

import matplotlib

matplotlib.use("Agg")  # no GUI window needed, works on any laptop
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D


# ==========================================================================
# ROAD NETWORK AND TRAFFIC MODEL
# ==========================================================================

# ---------------------------------------------------------------------------
# Traffic model
# ---------------------------------------------------------------------------
# travel time (hours) = distance / average speed * congestion multiplier
TRAFFIC_MULTIPLIER = {"Low": 1.0, "Moderate": 1.5, "Heavy": 2.5, "Severe": 4.0}
TRAFFIC_COLOUR = {
    "Low": "#2e9e44",
    "Moderate": "#f2c200",
    "Heavy": "#f28a00",
    "Severe": "#d62828",
}
# Stop-and-go traffic burns more fuel per km (used only for the fuel estimate)
FUEL_FACTOR = {"Low": 1.0, "Moderate": 1.15, "Heavy": 1.35, "Severe": 1.6}

# road type -> (free-flow average speed in km/h, detour factor)
# The detour factor says how much longer the real road is than the straight
# line between the two places. It is always >= 1, which is what keeps the
# straight-line heuristic in A* safe (admissible).
ROAD_TYPES = {
    "highway": (60, 1.15),
    "arterial": (45, 1.25),
    "local": (30, 1.35),
}

# ---------------------------------------------------------------------------
# Locations: name -> (latitude, longitude)
# ---------------------------------------------------------------------------
LOCATIONS = {
    "Guindy": (13.0067, 80.2206),
    "Alandur": (13.0033, 80.2010),
    "Saidapet": (13.0213, 80.2231),
    "T. Nagar": (13.0418, 80.2341),
    "Nungambakkam": (13.0569, 80.2425),
    "Egmore": (13.0732, 80.2609),
    "Mylapore": (13.0368, 80.2676),
    "Adyar": (13.0012, 80.2565),
    "Besant Nagar": (13.0002, 80.2668),
    "Thiruvanmiyur": (12.9830, 80.2594),
    "Velachery": (12.9815, 80.2180),
    "Perungudi": (12.9654, 80.2461),
    "Sholinganallur": (12.9010, 80.2279),
    "Tambaram": (12.9249, 80.1000),
    "Chromepet": (12.9516, 80.1462),
    "Pallavaram": (12.9675, 80.1491),
    "Porur": (13.0382, 80.1565),
    "Koyambedu": (13.0694, 80.1948),
    "Anna Nagar": (13.0850, 80.2101),
    "Vadapalani": (13.0500, 80.2121),
}

# (place A, place B, road type, starting traffic level)
ROADS = [
    ("Guindy", "Alandur", "arterial", "Moderate"),
    ("Guindy", "Saidapet", "arterial", "Severe"),
    ("Guindy", "Velachery", "arterial", "Heavy"),
    ("Guindy", "Porur", "highway", "Moderate"),
    ("Alandur", "Pallavaram", "highway", "Heavy"),
    ("Pallavaram", "Chromepet", "highway", "Moderate"),
    ("Chromepet", "Tambaram", "highway", "Heavy"),
    ("Pallavaram", "Velachery", "local", "Low"),
    ("Saidapet", "T. Nagar", "arterial", "Heavy"),
    ("Saidapet", "Adyar", "arterial", "Moderate"),
    ("T. Nagar", "Nungambakkam", "arterial", "Severe"),
    ("T. Nagar", "Vadapalani", "arterial", "Moderate"),
    ("T. Nagar", "Mylapore", "arterial", "Heavy"),
    ("Nungambakkam", "Egmore", "arterial", "Moderate"),
    ("Nungambakkam", "Anna Nagar", "arterial", "Heavy"),
    ("Egmore", "Mylapore", "arterial", "Heavy"),
    ("Egmore", "Anna Nagar", "arterial", "Moderate"),
    ("Anna Nagar", "Koyambedu", "arterial", "Low"),
    ("Koyambedu", "Vadapalani", "arterial", "Moderate"),
    ("Koyambedu", "Porur", "highway", "Low"),
    ("Vadapalani", "Porur", "arterial", "Low"),
    ("Mylapore", "Adyar", "arterial", "Moderate"),
    ("Adyar", "Besant Nagar", "local", "Low"),
    ("Adyar", "Thiruvanmiyur", "arterial", "Heavy"),
    ("Besant Nagar", "Thiruvanmiyur", "local", "Low"),
    ("Thiruvanmiyur", "Perungudi", "highway", "Moderate"),
    ("Thiruvanmiyur", "Velachery", "arterial", "Moderate"),
    ("Perungudi", "Sholinganallur", "highway", "Severe"),
    ("Perungudi", "Velachery", "arterial", "Heavy"),
    ("Velachery", "Tambaram", "arterial", "Moderate"),
    ("Velachery", "Sholinganallur", "local", "Low"),
    ("Sholinganallur", "Tambaram", "arterial", "Moderate"),
    ("Mylapore", "Besant Nagar", "local", "Low"),
    ("Alandur", "Saidapet", "arterial", "Low"),
    ("Porur", "Alandur", "arterial", "Moderate"),
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line (great circle) distance between two coordinates in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def road_key(a, b):
    """Order-independent key for a road, since roads work both ways."""
    return tuple(sorted((a, b)))


class RoadNetwork:
    """
    Thin wrapper around a networkx Graph.

    networkx is only used as the container (and for drawing). The search
    algorithms in the algorithms/ folder never call any networkx path
    function; they just ask this class for neighbours and edge costs.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self._initial_state = {}

    # ----------------------------------------------------------- building
    def add_location(self, name, lat, lon):
        self.graph.add_node(name, lat=lat, lon=lon)

    def add_road(self, a, b, road_type, traffic="Low", distance_km=None):
        if traffic not in TRAFFIC_MULTIPLIER:
            raise ValueError(f"Unknown traffic level: {traffic}")
        speed, detour = ROAD_TYPES[road_type]
        straight = self.straight_line_km(a, b)
        if distance_km is None:
            # round UP to 0.1 km so the road is never shorter than the straight line
            distance_km = math.ceil(straight * detour * 10) / 10
        if distance_km < straight:
            raise ValueError(f"Road {a}-{b} is shorter than the straight line distance")
        self.graph.add_edge(
            a, b,
            distance_km=distance_km,
            speed_kmph=speed,
            road_type=road_type,
            traffic=traffic,
            closed=False,
        )

    def save_initial_state(self):
        """Remember the starting traffic so reset() can bring it back."""
        self._initial_state = {
            road_key(u, v): (d["traffic"], d["closed"])
            for u, v, d in self.graph.edges(data=True)
        }

    # ------------------------------------------------------------ queries
    @property
    def locations(self):
        return sorted(self.graph.nodes)

    @property
    def max_speed_kmph(self):
        """Fastest free-flow speed on any road (needed for the A* heuristic)."""
        return max(d["speed_kmph"] for _, _, d in self.graph.edges(data=True))

    def has_location(self, name):
        return name in self.graph

    def straight_line_km(self, a, b):
        na, nb = self.graph.nodes[a], self.graph.nodes[b]
        return haversine_km(na["lat"], na["lon"], nb["lat"], nb["lon"])

    def neighbors(self, node):
        """Yield (neighbour, edge data) for every road that is currently open."""
        for nbr, data in self.graph[node].items():
            if not data["closed"]:
                yield nbr, data

    @staticmethod
    def time_hours(data):
        """Travel time of one road segment in hours."""
        return data["distance_km"] / data["speed_kmph"] * TRAFFIC_MULTIPLIER[data["traffic"]]

    def edge_cost(self, u, v, criterion="time"):
        """Cost of one segment: hours for 'time', kilometres for 'distance'."""
        data = self.graph[u][v]
        if criterion == "distance":
            return data["distance_km"]
        return self.time_hours(data)

    # ---------------------------------------------------- traffic updates
    def set_traffic(self, a, b, level):
        if level not in TRAFFIC_MULTIPLIER:
            raise ValueError(f"Unknown traffic level: {level}")
        self.graph[a][b]["traffic"] = level

    def close_road(self, a, b):
        self.graph[a][b]["closed"] = True

    def open_road(self, a, b):
        self.graph[a][b]["closed"] = False

    def randomize_traffic(self, seed=None):
        """Give every road a new random traffic level (light traffic is more likely)."""
        rng = random.Random(seed)
        levels = list(TRAFFIC_MULTIPLIER)
        weights = [4, 3, 2, 1]
        for u, v in sorted(road_key(u, v) for u, v in self.graph.edges):
            self.graph[u][v]["traffic"] = rng.choices(levels, weights)[0]

    def reset(self):
        for (a, b), (traffic, closed) in self._initial_state.items():
            self.graph[a][b]["traffic"] = traffic
            self.graph[a][b]["closed"] = closed

    def copy(self):
        clone = RoadNetwork()
        clone.graph = self.graph.copy()
        clone._initial_state = copy.deepcopy(self._initial_state)
        return clone

    # -------------------------------------------------------------- tables
    def road_table(self):
        """One dict per road, ready to turn into a table."""
        rows = []
        for u, v in sorted(road_key(u, v) for u, v in self.graph.edges):
            d = self.graph[u][v]
            rows.append({
                "Road": f"{u} - {v}",
                "Type": d["road_type"],
                "Distance (km)": d["distance_km"],
                "Avg speed (km/h)": d["speed_kmph"],
                "Traffic": d["traffic"],
                "Multiplier": TRAFFIC_MULTIPLIER[d["traffic"]],
                "Travel time (min)": round(self.time_hours(d) * 60, 1),
                "Status": "CLOSED" if d["closed"] else "Open",
            })
        return rows


def build_chennai_network():
    """Create the default simulated network with its starting traffic."""
    net = RoadNetwork()
    for name, (lat, lon) in LOCATIONS.items():
        net.add_location(name, lat, lon)
    for a, b, road_type, traffic in ROADS:
        net.add_road(a, b, road_type, traffic)
    net.save_initial_state()
    return net


# ==========================================================================
# SHARED SEARCH HELPERS
# ==========================================================================

@dataclass
class SearchResult:
    algorithm: str
    start: str
    goal: str
    criterion: str                 # "time" or "distance"
    found: bool
    path: list = field(default_factory=list)
    explored_nodes: int = 0
    explored_order: list = field(default_factory=list)
    time_ms: float = 0.0           # how long the search itself took
    total_distance_km: float = None
    total_time_h: float = None
    message: str = ""

    @property
    def hops(self):
        """Number of road segments on the route."""
        return max(len(self.path) - 1, 0)

    @property
    def total_cost(self):
        """Route cost in the unit of the criterion (hours or km)."""
        if not self.found:
            return None
        return self.total_distance_km if self.criterion == "distance" else self.total_time_h


def check_inputs(network, start, goal, criterion):
    if not network.has_location(start):
        raise ValueError(f"Unknown starting location: {start!r}")
    if not network.has_location(goal):
        raise ValueError(f"Unknown destination: {goal!r}")
    if criterion not in ("time", "distance"):
        raise ValueError("criterion must be 'time' or 'distance'")


def rebuild_path(parent, goal):
    """Walk the parent pointers back from the goal to the start."""
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def path_totals(network, path):
    """Total distance (km) and travel time (hours) of a path, with current traffic."""
    km = 0.0
    hours = 0.0
    for u, v in zip(path, path[1:]):
        data = network.graph[u][v]
        km += data["distance_km"]
        hours += network.time_hours(data)
    return km, hours


def make_result(network, name, start, goal, criterion, path, order, elapsed_ms):
    """Package up what a search found. path is None when no route exists."""
    if path is None:
        return SearchResult(
            algorithm=name, start=start, goal=goal, criterion=criterion,
            found=False, explored_nodes=len(order), explored_order=order,
            time_ms=elapsed_ms,
            message=f"No route exists from {start} to {goal} with the current road closures.",
        )
    km, hours = path_totals(network, path)
    return SearchResult(
        algorithm=name, start=start, goal=goal, criterion=criterion,
        found=True, path=path, explored_nodes=len(order), explored_order=order,
        time_ms=elapsed_ms, total_distance_km=km, total_time_h=hours,
    )


# ==========================================================================
# ALGORITHM 1: BREADTH FIRST SEARCH
# ==========================================================================

def bfs(network, start, goal, criterion="time"):
    """
    criterion is only used to report the distance/time of the path BFS found.
    It does not change the search.
    """
    check_inputs(network, start, goal, criterion)

    t0 = _time.perf_counter()

    frontier = deque([start])       # FIFO queue
    parent = {start: None}          # doubles as the "seen" set
    order = []                      # nodes in the order they were expanded
    path = None

    while frontier:
        node = frontier.popleft()
        order.append(node)

        if node == goal:
            path = rebuild_path(parent, goal)
            break

        # sorted so ties are always broken the same way
        for nbr, _ in sorted(network.neighbors(node), key=lambda item: item[0]):
            if nbr not in parent:
                parent[nbr] = node
                frontier.append(nbr)

    elapsed_ms = (_time.perf_counter() - t0) * 1000
    return make_result(network, "BFS", start, goal, criterion, path, order, elapsed_ms)


# ==========================================================================
# ALGORITHM 2: UNIFORM COST SEARCH
# ==========================================================================

def ucs(network, start, goal, criterion="time"):
    check_inputs(network, start, goal, criterion)

    t0 = _time.perf_counter()

    counter = 0                         # tie-breaker so the heap never compares node names
    heap = [(0.0, counter, start)]      # entries are (g, counter, node)
    best_g = {start: 0.0}
    parent = {start: None}
    order = []
    path = None

    while heap:
        g, _, node = heapq.heappop(heap)

        # We may have pushed this node more than once. If a cheaper way to it
        # was found after this entry was pushed, this entry is out of date.
        if g > best_g[node]:
            continue

        order.append(node)
        if node == goal:
            path = rebuild_path(parent, goal)
            break

        for nbr, _data in network.neighbors(node):
            new_g = g + network.edge_cost(node, nbr, criterion)
            if new_g < best_g.get(nbr, float("inf")):
                best_g[nbr] = new_g
                parent[nbr] = node
                counter += 1
                heapq.heappush(heap, (new_g, counter, nbr))

    elapsed_ms = (_time.perf_counter() - t0) * 1000
    return make_result(network, "UCS", start, goal, criterion, path, order, elapsed_ms)


# ==========================================================================
# ALGORITHM 3: A* SEARCH
# ==========================================================================

def heuristic(network, node, goal, criterion="time"):
    km = network.straight_line_km(node, goal)
    if criterion == "distance":
        return km
    return km / network.max_speed_kmph


def astar(network, start, goal, criterion="time"):
    check_inputs(network, start, goal, criterion)

    t0 = _time.perf_counter()

    counter = 0
    h_start = heuristic(network, start, goal, criterion)
    heap = [(h_start, counter, 0.0, start)]     # (f, counter, g, node)
    best_g = {start: 0.0}
    parent = {start: None}
    order = []
    path = None

    while heap:
        f, _, g, node = heapq.heappop(heap)

        if g > best_g[node]:        # outdated queue entry
            continue

        order.append(node)
        if node == goal:
            path = rebuild_path(parent, goal)
            break

        for nbr, _data in network.neighbors(node):
            new_g = g + network.edge_cost(node, nbr, criterion)
            if new_g < best_g.get(nbr, float("inf")):
                best_g[nbr] = new_g
                parent[nbr] = node
                new_f = new_g + heuristic(network, nbr, goal, criterion)
                counter += 1
                heapq.heappush(heap, (new_f, counter, new_g, nbr))

    elapsed_ms = (_time.perf_counter() - t0) * 1000
    return make_result(network, "A*", start, goal, criterion, path, order, elapsed_ms)


# ==========================================================================
# ALTERNATIVE ROUTES
# ==========================================================================

def find_alternatives(network, start, goal, criterion="time", k=3):
    """Return up to k routes (best first) as SearchResult objects."""
    best = astar(network, start, goal, criterion)
    if not best.found or best.hops == 0:
        return [best]

    routes = {tuple(best.path): best}
    for a, b in zip(best.path, best.path[1:]):
        trial = network.copy()
        trial.close_road(a, b)
        result = astar(trial, start, goal, criterion)
        if result.found:
            routes.setdefault(tuple(result.path), result)

    ranked = sorted(routes.values(), key=lambda r: r.total_cost)
    return ranked[:k]


# ==========================================================================
# TRIP HELPERS (fuel, arrival time, route steps)
# ==========================================================================

def path_text(path):
    return " → ".join(path) if path else "-"


def format_duration(hours):
    """0.7 -> '42 min', 1.25 -> '1 h 15 min'."""
    total_min = int(round(hours * 60))
    if total_min < 60:
        return f"{total_min} min"
    return f"{total_min // 60} h {total_min % 60:02d} min"


def route_segments(network, path):
    """Details of each road on the path, one dict per segment."""
    rows = []
    for u, v in zip(path, path[1:]):
        d = network.graph[u][v]
        rows.append({
            "From": u,
            "To": v,
            "Distance (km)": d["distance_km"],
            "Traffic": d["traffic"],
            "Multiplier": TRAFFIC_MULTIPLIER[d["traffic"]],
            "Time (min)": round(network.time_hours(d) * 60, 1),
        })
    return rows


def fuel_litres(network, path, km_per_litre):
    """
    Rough fuel estimate: distance / mileage, scaled up per segment for the
    stop-and-go traffic on it (see FUEL_FACTOR in city_graph.py).
    """
    if km_per_litre <= 0:
        raise ValueError("Mileage must be greater than zero")
    litres = 0.0
    for u, v in zip(path, path[1:]):
        d = network.graph[u][v]
        litres += d["distance_km"] / km_per_litre * FUEL_FACTOR[d["traffic"]]
    return litres


def arrival_time(departure, hours):
    """departure is a datetime; returns the estimated arrival datetime."""
    return departure + timedelta(hours=hours)


# ==========================================================================
# ALGORITHM COMPARISON
# ==========================================================================

ALGORITHMS = {"BFS": bfs, "UCS": ucs, "A*": astar}


def run_all(network, start, goal, criterion="time", repeats=200):
    """
    Run every algorithm and return {name: SearchResult}.

    One search on a graph this small takes a few microseconds, which a timer
    cannot measure reliably, so each algorithm is run `repeats` times and the
    mean time is stored in the result.
    """
    results = {}
    for name, algo in ALGORITHMS.items():
        result = algo(network, start, goal, criterion)
        total = result.time_ms
        for _ in range(repeats - 1):
            total += algo(network, start, goal, criterion).time_ms
        result.time_ms = total / repeats
        results[name] = result
    return results


def optimality_labels(results):
    """
    Compare each route's cost with the cheapest cost found by any algorithm.
    (UCS is the reference in theory; using the minimum means a bug in UCS
    would show up here instead of being hidden.)
    """
    costs = [r.total_cost for r in results.values() if r.found]
    labels = {}
    if not costs:
        return {name: "-" for name in results}
    best = min(costs)
    for name, r in results.items():
        if not r.found:
            labels[name] = "-"
        elif r.total_cost <= best * (1 + 1e-9):
            labels[name] = "Optimal"
        else:
            extra = (r.total_cost / best - 1) * 100
            labels[name] = f"Not optimal (+{extra:.1f}%)"
    return labels


def comparison_table(results):
    """DataFrame with one row per metric and one column per algorithm."""
    labels = optimality_labels(results)
    metrics = [
        "Final Route",
        "Road Segments",
        "Number of Explored Nodes",
        "Execution Time (ms)",
        "Total Distance (km)",
        "Estimated Travel Time (min)",
        "Optimality",
    ]
    table = {m: {} for m in metrics}
    for name, r in results.items():
        table["Final Route"][name] = path_text(r.path) if r.found else "No route"
        table["Road Segments"][name] = r.hops if r.found else "-"
        table["Number of Explored Nodes"][name] = r.explored_nodes
        table["Execution Time (ms)"][name] = f"{r.time_ms:.4f}"
        table["Total Distance (km)"][name] = f"{r.total_distance_km:.1f}" if r.found else "-"
        table["Estimated Travel Time (min)"][name] = f"{r.total_time_h * 60:.1f}" if r.found else "-"
        table["Optimality"][name] = labels[name]
    df = pd.DataFrame(table).T.astype(str)     # all text, so every table widget can show it
    df.index.name = "Metric"
    return df


# ==========================================================================
# VISUALIZATION
# ==========================================================================

ROUTE_COLOUR = "#6a1b9a"      # purple, not used by any traffic level
CLOSED_COLOUR = "#8a8a8a"
ALGO_COLOURS = {"BFS": "#0072B2", "UCS": "#D55E00", "A*": "#009E73"}

# Some places are close together on the map, so their labels need a nudge
# (x offset, y offset in points, horizontal alignment)
LABEL_OFFSETS = {
    "Besant Nagar": (8, -14, "left"),
    "Adyar": (-8, -14, "right"),
    "Thiruvanmiyur": (10, -4, "left"),
    "Mylapore": (10, 4, "left"),
    "Alandur": (-9, 6, "right"),
    "Pallavaram": (-9, 6, "right"),
    "Chromepet": (-9, -12, "right"),
    "Velachery": (9, -12, "left"),
    "Guindy": (9, -12, "left"),
}


def draw_network(network, route=None, start=None, goal=None,
                 alternatives=None, show_distances=False, title=None):
    """
    Road map: roads coloured by traffic, closed roads dashed grey, the chosen
    route drawn on top with a thick purple outline, start and destination marked.
    """
    g = network.graph
    pos = {n: (d["lon"], d["lat"]) for n, d in g.nodes(data=True)}
    route = route or []
    route_edges = {frozenset(e) for e in zip(route, route[1:])}
    alt_edges = set()
    for alt in alternatives or []:
        alt_edges |= {frozenset(e) for e in zip(alt, alt[1:])}

    fig, ax = plt.subplots(figsize=(11, 9))

    for u, v, d in g.edges(data=True):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        key = frozenset((u, v))
        if d["closed"]:
            ax.plot(x, y, color=CLOSED_COLOUR, linewidth=2, linestyle=(0, (4, 4)), zorder=1)
            ax.plot(sum(x) / 2, sum(y) / 2, marker="X", color="black", markersize=9, zorder=5)
            continue
        if key in route_edges:
            ax.plot(x, y, color=ROUTE_COLOUR, linewidth=10, solid_capstyle="round", zorder=2)
        elif key in alt_edges:
            ax.plot(x, y, color=ROUTE_COLOUR, linewidth=7, alpha=0.3,
                    solid_capstyle="round", zorder=2)
        ax.plot(x, y, color=TRAFFIC_COLOUR[d["traffic"]], linewidth=3,
                solid_capstyle="round", zorder=3)
        if show_distances:
            ax.text(sum(x) / 2, sum(y) / 2, f"{d['distance_km']:.1f}", fontsize=7,
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    for name, (x, y) in pos.items():
        if name == start:
            ax.scatter(x, y, s=330, marker="s", color="#0b2545", edgecolor="white",
                       linewidth=1.5, zorder=7)
        elif name == goal:
            ax.scatter(x, y, s=520, marker="*", color="#111111", edgecolor="white",
                       linewidth=1.2, zorder=7)
        else:
            ax.scatter(x, y, s=170, color="#e8eef5", edgecolor="#33475b",
                       linewidth=1.6, zorder=4)

        dx, dy, ha = LABEL_OFFSETS.get(name, (0, 11, "center"))
        label = name
        weight = "normal"
        if name == start:
            label, weight = f"{name}\n(START)", "bold"
            dx, dy, ha = -16, 6, "right"
        elif name == goal:
            label, weight = f"{name}\n(DESTINATION)", "bold"
            dx, dy, ha = -16, 6, "right"
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=9, fontweight=weight, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    legend = [Line2D([0], [0], color=TRAFFIC_COLOUR[lvl], lw=3,
                     label=f"{lvl} traffic ({TRAFFIC_MULTIPLIER[lvl]}x)")
              for lvl in TRAFFIC_COLOUR]
    legend.append(Line2D([0], [0], color=CLOSED_COLOUR, lw=2, linestyle=(0, (4, 4)),
                         marker="X", markerfacecolor="black", markeredgecolor="black",
                         label="Closed road"))
    if route:
        legend.append(Line2D([0], [0], color=ROUTE_COLOUR, lw=9, label="Recommended route"))
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.95)

    ax.set_title(title or "Simulated Chennai road network", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.0 / math.cos(math.radians(13.0)))
    ax.margins(0.08)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def comparison_charts(results):
    """Three bar charts: execution time, explored nodes, travel time."""
    names = list(results)
    colours = [ALGO_COLOURS[n] for n in names]
    panels = [
        ("Execution time (ms)", [results[n].time_ms for n in names], "{:.4f}"),
        ("Explored nodes", [results[n].explored_nodes for n in names], "{:d}"),
        ("Estimated travel time (min)",
         [(results[n].total_time_h or 0) * 60 for n in names], "{:.1f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, values, fmt) in zip(axes, panels):
        bars = ax.bar(names, values, color=colours, width=0.55)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            text = fmt.format(int(value)) if fmt == "{:d}" else fmt.format(value)
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.15)
    fig.tight_layout()
    return fig


# ==========================================================================
# STREAMLIT DASHBOARD
# ==========================================================================

CRITERIA = {
    "Fastest route (minimum travel time)": "time",
    "Shortest route (minimum distance)": "distance",
}
ALGO_LONG_NAMES = {
    "A*": "A* Search",
    "UCS": "Uniform Cost Search",
    "BFS": "Breadth First Search",
}
DEFAULT_START = "Saidapet"
DEFAULT_GOAL = "Anna Nagar"


# --------------------------------------------------------------- state
def init_state():
    if "network" not in st.session_state:
        st.session_state["network"] = build_chennai_network()
        st.session_state["version"] = 0          # bumped whenever many roads change at once
        st.session_state["route_requested"] = False
        st.session_state["last_path"] = None


def request_route():
    st.session_state["route_requested"] = True


def reset_conditions():
    st.session_state["network"].reset()
    st.session_state["version"] += 1


def random_traffic_update():
    st.session_state["network"].randomize_traffic()
    st.session_state["version"] += 1


def apply_traffic(a, b, key):
    st.session_state["network"].set_traffic(a, b, st.session_state[key])


def apply_closure(a, b, key):
    net = st.session_state["network"]
    if st.session_state[key]:
        net.close_road(a, b)
    else:
        net.open_road(a, b)


# -------------------------------------------------------------- helpers
def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


def solve(net, algo_name, start, goal, criterion):
    """Run the chosen algorithm once. Returns None if the input is not usable."""
    try:
        return ALGORITHMS[algo_name](net, start, goal, criterion)
    except ValueError as err:       # e.g. an unknown location name
        st.error(str(err))
        return None


# --------------------------------------------------------------- tabs
def tab_planner(net, result, start, goal, algo_name, criterion, departure, mileage):
    if not st.session_state["route_requested"]:
        st.info("Choose a start, a destination and an algorithm in the sidebar, "
                "then press **Find optimal route**.")
        show_figure(draw_network(net, start=start, goal=goal))
        return

    if start == goal:
        st.warning("The starting location and the destination are the same, "
                   "so there is nothing to travel. Pick a different destination.")
        show_figure(draw_network(net, start=start, goal=goal))
        return

    if result is None:
        return

    if not result.found:
        st.error(result.message + " Re-open a road in the Traffic Management tab.")
        show_figure(draw_network(net, start=start, goal=goal))
        return

    # tell the user if new traffic conditions changed the recommendation
    last = st.session_state["last_path"]
    if last is not None and last != result.path:
        st.info("Conditions changed, so the recommended route was recalculated "
                "and is now different from the previous one.")
    st.session_state["last_path"] = result.path

    if algo_name == "BFS":
        st.warning("BFS counts road segments only. This is the route with the fewest "
                   "roads, which is not always the fastest one. Compare it with UCS or "
                   "A* in the Algorithm Comparison tab.")

    litres = fuel_litres(net, result.path, mileage)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Distance", f"{result.total_distance_km:.1f} km")
    c2.metric("Travel time", format_duration(result.total_time_h))
    c3.metric("Estimated arrival",
              arrival_time(departure, result.total_time_h).strftime("%I:%M %p"))
    c4.metric("Fuel (estimate)", f"{litres:.2f} L")
    c5.metric("Nodes explored", result.explored_nodes)

    st.markdown(f"**{ALGO_LONG_NAMES[algo_name]} route:** {path_text(result.path)}")

    alternatives = find_alternatives(net, start, goal, criterion, k=3)
    other_paths = [r.path for r in alternatives if r.path != result.path]
    show_alt = st.checkbox("Show alternative routes on the map (faint purple)",
                           key="show_alt")
    show_km = st.checkbox("Show road distances on the map", key="show_km")
    fig = draw_network(
        net, route=result.path, start=start, goal=goal,
        alternatives=other_paths if show_alt else None,
        show_distances=show_km,
        title=f"{start} to {goal} using {ALGO_LONG_NAMES[algo_name]}",
    )
    show_figure(fig)

    st.subheader("Route step by step")
    st.dataframe(pd.DataFrame(route_segments(net, result.path)), hide_index=True)

    st.subheader("Alternative routes")
    if len(alternatives) < 2:
        st.write("No other route exists with the current closures.")
    else:
        best_hours = alternatives[0].total_time_h
        rows = []
        for i, alt in enumerate(alternatives, start=1):
            rows.append({
                "Option": "Best" if i == 1 else f"Alternative {i - 1}",
                "Route": path_text(alt.path),
                "Distance (km)": round(alt.total_distance_km, 1),
                "Travel time": format_duration(alt.total_time_h),
                "Extra time vs best (min)": round((alt.total_time_h - best_hours) * 60, 1),
                "Fuel (L)": round(fuel_litres(net, alt.path, mileage), 2),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Alternatives are found by closing each road of the best route "
                   "in turn and searching again with A*.")


def tab_traffic(net, result, start, goal):
    left, right = st.columns([1, 2])
    version = st.session_state["version"]

    with left:
        st.subheader("Change a road")
        roads = sorted(road_key(u, v) for u, v in net.graph.edges)
        labels = {f"{a} - {b}": (a, b) for a, b in roads}
        choice = st.selectbox("Road", list(labels), key="edit_road")
        a, b = labels[choice]
        data = net.graph[a][b]

        level_key = f"level_{version}_{a}_{b}"
        st.radio("Traffic level", list(TRAFFIC_MULTIPLIER),
                 index=list(TRAFFIC_MULTIPLIER).index(data["traffic"]),
                 key=level_key, on_change=apply_traffic, args=(a, b, level_key),
                 help="Low 1.0x, Moderate 1.5x, Heavy 2.5x, Severe 4.0x")

        closed_key = f"closed_{version}_{a}_{b}"
        st.checkbox("Road closed (blocked)", value=data["closed"],
                    key=closed_key, on_change=apply_closure, args=(a, b, closed_key))

        st.caption(f"{data['distance_km']} km at {data['speed_kmph']} km/h free flow. "
                   f"Current travel time: {net.time_hours(data) * 60:.1f} min.")
        n_closed = sum(1 for _, _, d in net.graph.edges(data=True) if d["closed"])
        st.write(f"Closed roads right now: **{n_closed}**")
        st.caption("The route is recalculated automatically after every change. "
                   "Use the sidebar to reset everything or apply random traffic.")

    with right:
        path = result.path if (result is not None and result.found
                               and st.session_state["route_requested"]) else None
        show_figure(draw_network(net, route=path, start=start, goal=goal,
                                 title="Current traffic conditions"))

    st.subheader("All roads")
    st.dataframe(pd.DataFrame(net.road_table()), hide_index=True)


def tab_comparison(net, start, goal, criterion):
    st.subheader("BFS vs UCS vs A*")
    if start == goal:
        st.info("Choose two different locations to compare the algorithms.")
        return

    results = run_all(net, start, goal, criterion, repeats=200)
    if not any(r.found for r in results.values()):
        st.error(next(iter(results.values())).message)
        return

    unit = "travel time" if criterion == "time" else "distance"
    st.caption(f"Same problem for all three: {start} to {goal}, optimising {unit}. "
               "Execution time is the average of 200 runs.")
    st.table(comparison_table(results))
    show_figure(comparison_charts(results))

    st.markdown(
        "**How to read this.** BFS minimises the *number of road segments* and ignores "
        "distance and traffic, so its route can be slower than the others. UCS and A* "
        "minimise the weighted cost (travel time or distance), so both return an optimal "
        "route. A* reaches the same answer by expanding fewer nodes, because its heuristic "
        "steers the search towards the destination. On a graph this small, A* can still "
        "take slightly longer in milliseconds, since it computes a heuristic for every "
        "node it generates."
    )

    with st.expander("Order in which each algorithm explored the nodes"):
        for name, r in results.items():
            st.write(f"**{name}** ({r.explored_nodes} nodes): " + ", ".join(r.explored_order))


def tab_about():
    st.subheader("Simulated data notice")
    st.warning("The road distances, speeds and traffic levels in this application are "
               "simulated for academic demonstration. They are not real-time Chennai "
               "traffic data. Locations are only roughly placed on a real map.")

    st.subheader("Cost model")
    st.markdown("""
`Travel time = (Distance / Average speed) x Congestion multiplier`

| Traffic | Multiplier | Colour |
|---|---|---|
| Low | 1.0 | green |
| Moderate | 1.5 | yellow |
| Heavy | 2.5 | orange |
| Severe | 4.0 | red |

The recommended route is drawn with a thick purple outline. Closed roads are grey and dashed.
""")
    st.subheader("Algorithms")
    st.markdown("""
* **BFS** - fewest road segments. Ignores distance and traffic.
* **UCS** - priority queue ordered by g(n), the cost so far. Optimal for the chosen cost.
* **A\\*** - priority queue ordered by f(n) = g(n) + h(n), where
  h(n) = straight-line distance to the destination / 60 km/h (the fastest road speed).
  This never overestimates, so A* is also optimal.
""")


# ---------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="Intelligent Traffic Route Planner", layout="wide")
    init_state()
    net = st.session_state["network"]

    st.title("Intelligent Traffic Route Planner")
    st.caption("BFS, Uniform Cost Search and A* on a simulated Chennai road network. "
               "Distances and traffic are simulated, not real-time data.")

    # ---- sidebar: route selection
    sb = st.sidebar
    sb.header("Route selection")
    places = net.locations
    start = sb.selectbox("Starting location", places, index=places.index(DEFAULT_START),
                         key="start")
    goal = sb.selectbox("Destination", places, index=places.index(DEFAULT_GOAL),
                        key="goal")
    algo_name = sb.radio("Algorithm", list(ALGORITHMS), format_func=ALGO_LONG_NAMES.get,
                         key="algo")
    criterion_label = sb.radio("Optimise for", list(CRITERIA), key="criterion")
    criterion = CRITERIA[criterion_label]
    departure_time = sb.time_input("Departure time", value=time(8, 30), key="departure")
    mileage = sb.number_input("Vehicle mileage (km per litre)", min_value=1.0,
                              max_value=60.0, value=15.0, step=1.0, key="mileage")
    sb.button("Find optimal route", type="primary", on_click=request_route)

    sb.divider()
    sb.header("Traffic conditions")
    sb.button("Random traffic update", on_click=random_traffic_update)
    sb.button("Reset traffic and roads", on_click=reset_conditions)

    departure = datetime.combine(datetime.today(), departure_time)

    # ---- solve once, share between tabs
    result = None
    if st.session_state["route_requested"] and start != goal:
        result = solve(net, algo_name, start, goal, criterion)

    t1, t2, t3, t4 = st.tabs(["Route Planner", "Traffic Management",
                              "Algorithm Comparison", "About the Model"])
    with t1:
        tab_planner(net, result, start, goal, algo_name, criterion, departure, mileage)
    with t2:
        tab_traffic(net, result, start, goal)
    with t3:
        tab_comparison(net, start, goal, criterion)
    with t4:
        tab_about()


def _running_inside_streamlit():
    """True when Streamlit is executing this file (streamlit run ...)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:           # older Streamlit versions
            return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_inside_streamlit():
        main()
    else:
        # Started with plain "python traffic_planner.py": open the dashboard
        # in the browser by launching Streamlit on this same file.
        import subprocess
        import sys
        print("Starting the dashboard... it will open in your browser.")
        print("(Press Ctrl+C in this window to stop it.)")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
