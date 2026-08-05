"""
Minimax with Alpha-Beta Pruning - Dry Run Implementation
==========================================================
Game tree:

                        MAX
                      /      \\
                   MIN         MIN
                 /  |   \\    /  |   \\
                3   5    6  9   1    2

This script performs Minimax search with Alpha-Beta Pruning on the tree
above, evaluated strictly left to right. It prints an iteration-by-iteration
trace (node visited, alpha, beta, value, pruning decisions) identical to the
manual dry run, then draws the tree with alpha/beta values and the pruned
branch highlighted, saving it as output_minimax.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. Tree definition
#    Internal nodes are dicts: {"type": "MAX"/"MIN", "children": [...]}
#    Leaf nodes are plain integers.
# --------------------------------------------------------------------------
TREE = {
    "name": "MAX",
    "type": "MAX",
    "children": [
        {"name": "MIN_L", "type": "MIN", "children": [
            {"name": "L1", "value": 3},
            {"name": "L2", "value": 5},
            {"name": "L3", "value": 6},
        ]},
        {"name": "MIN_R", "type": "MIN", "children": [
            {"name": "L4", "value": 9},
            {"name": "L5", "value": 1},
            {"name": "L6", "value": 2},
        ]},
    ],
}

trace = []          # dry-run log
pruned_nodes = []    # leaves that were never evaluated


# --------------------------------------------------------------------------
# 2. Alpha-Beta Pruning (left to right), fully traced
# --------------------------------------------------------------------------
def alphabeta(node, alpha, beta, depth=0):
    indent = "  " * depth

    # ---- Leaf node ----
    if "value" in node:
        trace.append({
            "node": node["name"], "kind": "LEAF", "depth": depth,
            "value": node["value"], "alpha": alpha, "beta": beta,
        })
        return node["value"]

    # ---- Internal node ----
    is_max = node["type"] == "MAX"
    best = float("-inf") if is_max else float("inf")

    trace.append({
        "node": node["name"], "kind": node["type"] + "-ENTER", "depth": depth,
        "alpha": alpha, "beta": beta,
    })

    for i, child in enumerate(node["children"]):
        # If a previous sibling already triggered a cutoff, every remaining
        # child is skipped (pruned) without being visited at all.
        if (is_max and alpha >= beta) or (not is_max and beta <= alpha):
            _mark_pruned(child)
            trace.append({
                "node": child.get("name", "?"), "kind": "PRUNED", "depth": depth + 1,
                "alpha": alpha, "beta": beta,
            })
            continue

        child_value = alphabeta(child, alpha, beta, depth + 1)

        if is_max:
            if child_value > best:
                best = child_value
            if best > alpha:
                alpha = best
        else:
            if child_value < best:
                best = child_value
            if best < beta:
                beta = best

        trace.append({
            "node": node["name"], "kind": node["type"] + "-UPDATE", "depth": depth,
            "child": child.get("name", "?"), "child_value": child_value,
            "value": best, "alpha": alpha, "beta": beta,
        })

        # Cutoff check right after processing this child.
        if alpha >= beta:
            remaining = node["children"][i + 1:]
            for r in remaining:
                _mark_pruned(r)
                trace.append({
                    "node": r.get("name", "?"), "kind": "PRUNED", "depth": depth + 1,
                    "alpha": alpha, "beta": beta,
                })
            break

    trace.append({
        "node": node["name"], "kind": node["type"] + "-RETURN", "depth": depth,
        "value": best, "alpha": alpha, "beta": beta,
    })
    return best


def _mark_pruned(node):
    if "value" in node:
        pruned_nodes.append((node["name"], node["value"]))
    else:
        for c in node["children"]:
            _mark_pruned(c)


# --------------------------------------------------------------------------
# 3. Pretty-print the dry run trace
# --------------------------------------------------------------------------
def print_trace(trace):
    print("=" * 78)
    print("MINIMAX WITH ALPHA-BETA PRUNING - DRY RUN TRACE (left to right)")
    print("=" * 78)
    for rec in trace:
        pad = "  " * rec["depth"]
        if rec["kind"] == "LEAF":
            print(f"{pad}Leaf {rec['node']}: value={rec['value']}   "
                  f"(alpha={rec['alpha']}, beta={rec['beta']} inherited)")
        elif rec["kind"].endswith("-ENTER"):
            print(f"{pad}Enter {rec['node']} [{rec['kind'].split('-')[0]}]  "
                  f"alpha={rec['alpha']}, beta={rec['beta']}")
        elif rec["kind"].endswith("-UPDATE"):
            print(f"{pad}  {rec['node']} sees child {rec['child']}={rec['child_value']}  "
                  f"-> value={rec['value']}, alpha={rec['alpha']}, beta={rec['beta']}")
        elif rec["kind"] == "PRUNED":
            print(f"{pad}*** {rec['node']} PRUNED (alpha={rec['alpha']} >= beta={rec['beta']}) ***")
        elif rec["kind"].endswith("-RETURN"):
            print(f"{pad}{rec['node']} returns value={rec['value']}  "
                  f"(final alpha={rec['alpha']}, beta={rec['beta']})")


# --------------------------------------------------------------------------
# 4. Draw the tree with alpha/beta annotations and pruned branch marked
# --------------------------------------------------------------------------
def draw_result(root_value, best_move, pruned_names):
    pos = {
        "MAX": (5.0, 4.0), "MIN_L": (2.3, 2.5), "MIN_R": (7.7, 2.5),
        "L1": (1.0, 1.0), "L2": (2.3, 1.0), "L3": (3.6, 1.0),
        "L4": (6.4, 1.0), "L5": (7.7, 1.0), "L6": (9.0, 1.0),
    }
    leaf_values = {"L1": 3, "L2": 5, "L3": 6, "L4": 9, "L5": 1, "L6": 2}
    edges = [
        ("MAX", "MIN_L"), ("MAX", "MIN_R"),
        ("MIN_L", "L1"), ("MIN_L", "L2"), ("MIN_L", "L3"),
        ("MIN_R", "L4"), ("MIN_R", "L5"), ("MIN_R", "L6"),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if v in pruned_names:
            ax.plot([x1, x2], [y1, y2], color="#c0392b", linewidth=2.0,
                     linestyle="--", zorder=1)
        elif u == "MAX" and v == "MIN_L":
            ax.plot([x1, x2], [y1, y2], color="#0e6b3a", linewidth=3.2, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color="#4a4a4a", linewidth=1.8, zorder=1)

    # Internal nodes with alpha/beta labels
    node_info = {
        "MAX": ("MAX\n(alpha=3)", "#f9d879"),
        "MIN_L": ("MIN\nvalue=3", "#c8f7d4"),
        "MIN_R": ("MIN\nvalue=1", "#dbe9f6"),
    }
    for n, (label, color) in node_info.items():
        x, y = pos[n]
        ax.scatter([x], [y], s=2600, color=color, edgecolors="#1f4e79", linewidths=2.2, zorder=2)
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5, fontweight="bold",
                color="#1f2d3d", zorder=3)

    # Leaf nodes
    for n, v in leaf_values.items():
        x, y = pos[n]
        if n in pruned_names:
            ax.scatter([x], [y], s=1500, color="#f5d5d0", edgecolors="#c0392b",
                        linewidths=2, zorder=2)
            ax.text(x, y, str(v), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="#c0392b", zorder=3)
            ax.text(x, y - 0.42, "PRUNED", ha="center", va="center", fontsize=8.3,
                    fontweight="bold", color="#c0392b")
        else:
            selected = (n == "L1")  # leftmost leaf under the winning branch
            face = "#a8e6bf" if selected else "#eafaf1"
            ax.scatter([x], [y], s=1500, color=face, edgecolors="#0e6b3a",
                        linewidths=2, zorder=2)
            ax.text(x, y, str(v), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="#0e6b3a", zorder=3)

    ax.set_title("Minimax with Alpha-Beta Pruning - Result", fontsize=13.5,
                  fontweight="bold", color="#1f2d3d", pad=14)

    info = (f"Best move for MAX: LEFT subtree (MIN_L)\n"
            f"Final Minimax Value: {root_value}\n"
            f"Pruned node(s): {', '.join(str(v) for _, v in pruned_leaves) if pruned_leaves else 'None'}")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=10.5,
            va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#c9a227", lw=1.2))

    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 4.9)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("output_minimax.png", dpi=200, bbox_inches="tight")
    print("\nSaved output_minimax.png")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    root_value = alphabeta(TREE, float("-inf"), float("inf"))
    print_trace(trace)

    pruned_leaves = pruned_nodes
    pruned_names = {n for n, _ in pruned_leaves}

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"Final Minimax Value at root : {root_value}")
    print(f"Best move for MAX            : LEFT subtree (MIN_L), leaf value 3")
    print(f"Pruned node(s)                : "
          f"{', '.join(f'{n}={v}' for n, v in pruned_leaves) if pruned_leaves else 'None'}")

    draw_result(root_value, "MIN_L", pruned_names)
