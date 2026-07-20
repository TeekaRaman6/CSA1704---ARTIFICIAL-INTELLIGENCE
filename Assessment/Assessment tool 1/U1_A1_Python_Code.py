"""
Assessment Tool 1 - Analytical Problem Solving
Course: Artificial Intelligence (CSA17)

Solves the computational problems from the assessment:
    1. Water Jug Problem          -> Breadth-First Search
    3. 8-Queens Problem           -> Backtracking Search
    4. OLA Cab Booking            -> Utility-based agent scoring demo
    5. Logistics Network          -> Uniform Cost Search
"""

from collections import deque
import heapq


# ---------------------------------------------------------------------------
# 1. WATER JUG PROBLEM (BFS over the (4-gallon, 3-gallon) state space)
# ---------------------------------------------------------------------------
JUG_A, JUG_B, TARGET = 4, 3, 2


def next_states(a, b):
    """All states reachable from (a, b) in a single fill/empty/pour move."""
    results = []
    results.append(((JUG_A, b), "Fill 4-gallon jug"))
    results.append(((a, JUG_B), "Fill 3-gallon jug"))
    results.append(((0, b), "Empty 4-gallon jug"))
    results.append(((a, 0), "Empty 3-gallon jug"))

    pour_to_b = min(a, JUG_B - b)
    results.append(((a - pour_to_b, b + pour_to_b), "Pour 4-gallon jug into 3-gallon jug"))

    pour_to_a = min(b, JUG_A - a)
    results.append(((a + pour_to_a, b - pour_to_a), "Pour 3-gallon jug into 4-gallon jug"))
    return results


def solve_water_jug():
    """Breadth-first search that starts by filling the 3-gallon jug first."""
    start = (0, 0)
    frontier = deque()
    # Seed the frontier with "fill the 3-gallon jug" as the first move so the
    # search commits to that branch before trying anything else.
    frontier.append(((0, JUG_B), [((0, 0)), (0, JUG_B, "Fill 3-gallon jug")]))
    seen = {start, (0, JUG_B)}

    while frontier:
        (a, b), trace = frontier.popleft()
        if a == TARGET:
            return trace

        for (na, nb), action in next_states(a, b):
            if (na, nb) not in seen:
                seen.add((na, nb))
                frontier.append(((na, nb), trace + [(na, nb, action)]))
    return None


def run_water_jug():
    print("=" * 62)
    print("1. WATER JUG PROBLEM (Breadth-First Search)")
    print("=" * 62)
    trace = solve_water_jug()
    state0 = trace[0]
    print(f"  Start: 4-gallon jug = {state0[0]}, 3-gallon jug = {state0[1]}")
    for a, b, action in trace[1:]:
        print(f"    -> {action:<38s} => (4G={a}, 3G={b})")
    last = trace[-1]
    print(f"\n  Result: {last[0]} gallons isolated in the 4-gallon jug "
          f"in {len(trace) - 1} moves.\n")


# ---------------------------------------------------------------------------
# 3. 8-QUEENS PROBLEM (Backtracking Search)
# ---------------------------------------------------------------------------
BOARD_SIZE = 8


def is_safe(placement, row, col):
    for c, r in enumerate(placement[:col]):
        if r == row or abs(r - row) == abs(c - col):
            return False
    return True


def solve_eight_queens():
    placement = [-1] * BOARD_SIZE

    def backtrack(col):
        if col == BOARD_SIZE:
            return True
        for row in range(BOARD_SIZE):
            if is_safe(placement, row, col):
                placement[col] = row
                if backtrack(col + 1):
                    return True
                placement[col] = -1
        return False

    backtrack(0)
    return placement  # index = column (0-based), value = row (0-based)


def run_eight_queens():
    print("=" * 62)
    print("3. 8-QUEENS PROBLEM (Backtracking Search)")
    print("=" * 62)
    placement = solve_eight_queens()
    readable = [(c + 1, r + 1) for c, r in enumerate(placement)]
    print("  Column -> Row (1-indexed):", readable)
    print()
    for row in range(BOARD_SIZE):
        line = " ".join("Q" if placement[col] == row else "." for col in range(BOARD_SIZE))
        print("  " + line)
    print()


# ---------------------------------------------------------------------------
# 4. OLA CAB BOOKING (Utility-based agent - scoring demo)
# ---------------------------------------------------------------------------
def run_ola_cab_booking():
    print("=" * 62)
    print("4. OLA CAB BOOKING (Utility-Based Agent)")
    print("=" * 62)

    # comfort (1-10), fare (INR), wait_time (minutes)
    cab_options = {
        "Micro":  {"comfort": 4, "fare": 105, "wait": 5},
        "Mini":   {"comfort": 6, "fare": 135, "wait": 4},
        "Sedan":  {"comfort": 8, "fare": 210, "wait": 7},
        "Prime":  {"comfort": 9, "fare": 275, "wait": 3},
        "Shared": {"comfort": 3, "fare": 80,  "wait": 10},
    }
    weight_comfort, weight_fare, weight_wait = 6, 0.04, 2.0

    print(f"  Weights -> comfort: {weight_comfort}, fare: {weight_fare}, wait: {weight_wait}\n")

    best_name, best_score = None, float("-inf")
    for name, attrs in cab_options.items():
        utility = (weight_comfort * attrs["comfort"]
                   - weight_fare * attrs["fare"]
                   - weight_wait * attrs["wait"])
        print(f"  {name:8s} utility = {utility:6.2f}  "
              f"(comfort={attrs['comfort']}, fare={attrs['fare']}, wait={attrs['wait']})")
        if utility > best_score:
            best_name, best_score = name, utility

    print(f"\n  Agent selects: {best_name} (highest utility = {best_score:.2f})\n")


# ---------------------------------------------------------------------------
# 5. LOGISTICS NETWORK (Uniform Cost Search)
# ---------------------------------------------------------------------------
def uniform_cost_search(graph, start, goal):
    frontier = [(0, start, [start])]
    expanded_order = []
    closed = set()
    best_known = {start: 0}

    while frontier:
        cost, node, path = heapq.heappop(frontier)
        if node in closed:
            continue          # a cheaper expansion of this node already happened
        closed.add(node)
        expanded_order.append((node, cost))

        if node == goal:
            return cost, path, expanded_order

        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in best_known or new_cost < best_known[neighbor]:
                best_known[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))

    return None, None, expanded_order


def run_logistics_network():
    print("=" * 62)
    print("5. LOGISTICS NETWORK - UNIFORM COST SEARCH")
    print("=" * 62)

    # Weighted delivery network: warehouse S to destination warehouse G
    graph = {
        "S": [("A", 2), ("B", 5), ("C", 4)],
        "A": [("B", 2), ("D", 7)],
        "B": [("D", 4), ("G", 6)],
        "C": [("D", 3)],
        "D": [("G", 2)],
        "G": [],
    }

    cost, path, expanded = uniform_cost_search(graph, "S", "G")
    print("  Nodes expanded in order (node, path cost so far):")
    for node, c in expanded:
        print(f"    {node}  (cost so far = {c})")
    print(f"\n  Least-cost route: {' -> '.join(path)}")
    print(f"  Minimum total cost: {cost}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_water_jug()
    run_eight_queens()
    run_ola_cab_booking()
    run_logistics_network()
