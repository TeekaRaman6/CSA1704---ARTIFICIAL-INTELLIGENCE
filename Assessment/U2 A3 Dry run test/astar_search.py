"""
A* Search Algorithm - Dry Run Implementation
=============================================
Graph        : A, B, C, D, E, G   (undirected, weighted)
Start node   : A
Goal node    : G
Heuristic    : Straight-line-style estimate h(n), given in the problem

This script performs A* Search on the graph described in the problem
statement, prints an iteration-by-iteration trace (current node, g, h, f,
Open List, Closed List) identical to the manual dry run, reconstructs the
optimal path, and finally draws the graph with the optimal path highlighted
and saves it as output.png.

Author : Dry-run solution generator
"""

import heapq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# --------------------------------------------------------------------------
# 1. Graph and heuristic definition
# --------------------------------------------------------------------------
GRAPH = {
    "A": {"B": 2, "C": 4},
    "B": {"A": 2, "C": 3, "D": 7, "E": 2},
    "C": {"A": 4, "B": 3, "E": 3},
    "D": {"B": 7, "E": 2},
    "E": {"B": 2, "C": 3, "D": 2, "G": 2},
    "G": {"E": 2},
}

HEURISTIC = {"A": 7, "B": 6, "C": 4, "D": 3, "E": 2, "G": 0}

START, GOAL = "A", "G"


# --------------------------------------------------------------------------
# 2. A* Search with a full iteration trace
# --------------------------------------------------------------------------
def a_star(graph, heuristic, start, goal):
    """
    Returns (path, total_cost, trace) where `trace` is a list of dicts
    describing every iteration of the algorithm for the dry run table.
    """
    # open list holds tuples: (f, node)  -- a Python list is used (not just
    # heapq) so that we can print its *entire* contents at every step, the
    # way a dry run table requires.
    open_list = [start]
    closed_list = []

    g_score = {start: 0}
    f_score = {start: heuristic[start]}
    parent = {start: None}

    trace = []
    iteration = 0

    while open_list:
        iteration += 1

        # Pick the node in open_list with the smallest f(n).
        # Tie-break rule: smaller f first; if still tied, alphabetical order
        # (this matches the manual dry run convention used in the solution).
        open_list.sort(key=lambda n: (f_score[n], n))
        current = open_list[0]

        record = {
            "iteration": iteration,
            "current": current,
            "g": g_score[current],
            "h": heuristic[current],
            "f": f_score[current],
            "open_before": [(n, g_score[n], heuristic[n], f_score[n]) for n in open_list],
            "closed_before": list(closed_list),
        }

        # Goal test happens when the goal is SELECTED as current
        # (standard A* termination condition).
        if current == goal:
            record["note"] = "GOAL REACHED - stop expansion"
            trace.append(record)
            break

        open_list.remove(current)
        closed_list.append(current)

        expansions = []
        for neighbor, cost in graph[current].items():
            if neighbor in closed_list:
                continue
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                old_g = g_score.get(neighbor)
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic[neighbor]
                parent[neighbor] = current
                if neighbor not in open_list:
                    open_list.append(neighbor)
                    expansions.append(f"{neighbor}: new -> g={tentative_g}, h={heuristic[neighbor]}, f={f_score[neighbor]}")
                else:
                    expansions.append(f"{neighbor}: updated {old_g}->{tentative_g} (better path found)")
            else:
                expansions.append(f"{neighbor}: g={tentative_g} not better than existing {g_score[neighbor]} -> discarded")

        record["expansions"] = expansions
        record["open_after"] = [(n, g_score[n], heuristic[n], f_score[n]) for n in open_list]
        record["closed_after"] = list(closed_list)
        trace.append(record)

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()

    total_cost = g_score[goal]
    return path, total_cost, trace


# --------------------------------------------------------------------------
# 3. Pretty-print the dry run trace to the console
# --------------------------------------------------------------------------
def print_trace(trace):
    print("=" * 78)
    print("A* SEARCH - DRY RUN TRACE (Start = A, Goal = G)")
    print("=" * 78)
    for rec in trace:
        print(f"\nIteration {rec['iteration']}")
        print("-" * 78)
        print(f"Current Node : {rec['current']}   "
              f"g={rec['g']}  h={rec['h']}  f={rec['f']}")
        if "note" in rec:
            print(f"  >> {rec['note']}")
            open_str = ", ".join(f"{n}(f={f})" for n, g, h, f in rec["open_before"])
            print(f"Open List    : [{open_str}]")
            print(f"Closed List  : {rec['closed_before']}")
            continue

        for line in rec["expansions"]:
            print(f"  - {line}")

        open_str = ", ".join(f"{n}(g={g},h={h},f={f})" for n, g, h, f in rec["open_after"])
        print(f"Open List (after)   : [{open_str}]")
        print(f"Closed List (after) : {rec['closed_after']}")


# --------------------------------------------------------------------------
# 4. Draw the graph with the optimal path highlighted -> output.png
# --------------------------------------------------------------------------
def draw_result(path, total_cost):
    G = nx.Graph()
    for u, nbrs in GRAPH.items():
        for v, w in nbrs.items():
            G.add_edge(u, v, weight=w)

    pos = {
        "A": (2.0, 4.0), "B": (0.5, 2.2), "C": (3.5, 2.2),
        "D": (0.5, 0.2), "E": (3.5, 0.2), "G": (3.5, -1.6),
    }

    path_edges = list(zip(path, path[1:]))

    fig, ax = plt.subplots(figsize=(7, 8.5))

    nx.draw_networkx_edges(G, pos, ax=ax, width=2.0, edge_color="#c9c9c9")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges, width=4.0,
                            edge_color="#e74c3c")

    node_colors = ["#2ecc71" if n == path[0] else
                   "#e74c3c" if n == path[-1] else
                   "#dbe9f6" if n not in path else "#f9d879"
                   for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1600, node_color=node_colors,
                            edgecolors="#1f4e79", linewidths=2.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=15, font_weight="bold",
                             font_color="#1f2d3d")

    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=12, font_color="#7f2d16",
                                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    for n, (x, y) in pos.items():
        ax.text(x + 0.38, y - 0.05, f"h={HEURISTIC[n]}", fontsize=10,
                 color="#0e6b3a", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#eafaf1", ec="#0e6b3a", lw=0.7))

    ax.set_title("A* Search Result - Optimal Path Highlighted", fontsize=13,
                  fontweight="bold", color="#1f2d3d", pad=12)

    info = f"Path: {' -> '.join(path)}\nTotal Cost: {total_cost}"
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=11,
            va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#c9a227", lw=1.2))

    ax.axis("off")
    ax.set_xlim(-1.0, 5.2)
    ax.set_ylim(-2.4, 5.0)

    plt.tight_layout()
    plt.savefig("output.png", dpi=200, bbox_inches="tight")
    print("\nSaved output.png")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    path, total_cost, trace = a_star(GRAPH, HEURISTIC, START, GOAL)

    print_trace(trace)

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"Optimal Path : {' -> '.join(path)}")
    print(f"Total Path Cost : {total_cost}")

    draw_result(path, total_cost)
