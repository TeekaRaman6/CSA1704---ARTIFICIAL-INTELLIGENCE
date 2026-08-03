"""
=================================================================================
 CSA17 - ARTIFICIAL INTELLIGENCE | ASSESSMENT TOOL 2
 Scenario Based Assignment - Python Implementation
 Covers CO2: Greedy Search, A*, Hill Climbing, Simulated Annealing, CSP
          (Backtracking + Forward Checking), Online Search, Minimax & Alpha-Beta
=================================================================================
 This single script implements a working demonstration for every scenario in
 the assignment (Q1 - Q5). Each scenario is isolated in its own function so it
 can be read, run and understood independently. Running this file end-to-end:
     1. Prints a clearly labelled console report for every scenario.
     2. Builds a 5-panel summary figure and saves it as 'output.png'.
=================================================================================
"""

import heapq
import math
import random
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

random.seed(42)

# =================================================================================
# SCENARIO 1: EMERGENCY DRONE DELIVERY  -> Greedy Best-First Search vs A*
# =================================================================================
# The flood region is modelled as a weighted graph. Edge weights represent the
# real (variable) travel cost -- affected by terrain/weather. Each node has a
# straight-line-distance heuristic to the goal (a relaxation of the true cost).

DRONE_GRAPH = {
    "Base":    {"A": 4, "B": 7},
    "A":       {"Base": 4, "C": 3, "D": 6},
    "B":       {"Base": 7, "D": 2, "E": 5},
    "C":       {"A": 3, "Hospital": 7},
    "D":       {"A": 6, "B": 2, "Hospital": 4, "E": 3},
    "E":       {"B": 5, "D": 3, "Hospital": 2},
    "Hospital": {"C": 7, "D": 4, "E": 2},
}

# Straight-line-distance heuristic to "Hospital" (assumed known/estimated by drone)
DRONE_HEURISTIC = {
    "Base": 9, "A": 6, "B": 5, "C": 5, "D": 3, "E": 2, "Hospital": 0
}

# Roads that become blocked mid-flight (simulates a dynamic environment).
BLOCKED_EDGES_EVENT = {("D", "Hospital"), ("Hospital", "D")}


def greedy_best_first_search(graph, heuristic, start, goal, blocked=frozenset()):
    """Expands the node that LOOKS closest to the goal (heuristic only)."""
    frontier = [(heuristic[start], start, [start], 0)]
    visited = set()
    expansions = []
    while frontier:
        h, node, path, cost_so_far = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        expansions.append(node)
        if node == goal:
            return path, cost_so_far, expansions
        for neighbor, weight in graph[node].items():
            if (node, neighbor) in blocked or neighbor in visited:
                continue
            heapq.heappush(frontier, (heuristic[neighbor], neighbor,
                                       path + [neighbor], cost_so_far + weight))
    return None, math.inf, expansions


def a_star_search(graph, heuristic, start, goal, blocked=frozenset()):
    """Expands the node with lowest f(n) = g(n) + h(n) -> cost-aware and complete."""
    frontier = [(heuristic[start], 0, start, [start])]
    best_g = {start: 0}
    expansions = []
    while frontier:
        f, g, node, path = heapq.heappop(frontier)
        if node in expansions and g > best_g.get(node, math.inf):
            continue
        expansions.append(node)
        if node == goal:
            return path, g, expansions
        for neighbor, weight in graph[node].items():
            if (node, neighbor) in blocked:
                continue
            new_g = g + weight
            if new_g < best_g.get(neighbor, math.inf):
                best_g[neighbor] = new_g
                heapq.heappush(frontier, (new_g + heuristic[neighbor], new_g,
                                           neighbor, path + [neighbor]))
    return None, math.inf, expansions


def run_scenario_1():
    print("\n" + "=" * 90)
    print("SCENARIO 1: EMERGENCY DRONE DELIVERY  |  Greedy Best-First Search vs A*")
    print("=" * 90)

    g_path, g_cost, g_exp = greedy_best_first_search(DRONE_GRAPH, DRONE_HEURISTIC, "Base", "Hospital")
    a_path, a_cost, a_exp = a_star_search(DRONE_GRAPH, DRONE_HEURISTIC, "Base", "Hospital")
    print(f"[Normal conditions] Greedy Best-First -> path={g_path}, cost={g_cost}, nodes expanded={len(g_exp)}")
    print(f"[Normal conditions] A* Search          -> path={a_path}, cost={a_cost}, nodes expanded={len(a_exp)}")

    # Now simulate a dynamic event: the shortest final leg (D-Hospital) is blocked mid-route.
    g_path2, g_cost2, g_exp2 = greedy_best_first_search(DRONE_GRAPH, DRONE_HEURISTIC, "Base", "Hospital", BLOCKED_EDGES_EVENT)
    a_path2, a_cost2, a_exp2 = a_star_search(DRONE_GRAPH, DRONE_HEURISTIC, "Base", "Hospital", BLOCKED_EDGES_EVENT)
    print(f"[Path D-Hospital BLOCKED] Greedy Best-First -> path={g_path2}, cost={g_cost2}, nodes expanded={len(g_exp2)}")
    print(f"[Path D-Hospital BLOCKED] A* Search          -> path={a_path2}, cost={a_cost2}, nodes expanded={len(a_exp2)}")

    return {
        "graph": DRONE_GRAPH,
        "normal": {"greedy": (g_path, g_cost), "astar": (a_path, a_cost)},
        "blocked": {"greedy": (g_path2, g_cost2), "astar": (a_path2, a_cost2)},
    }


# =================================================================================
# SCENARIO 2: SMART CITY TRAFFIC SIGNAL OPTIMIZATION -> Hill Climbing vs Sim. Annealing
# =================================================================================
# The "cost" function models total waiting time + fuel + congestion as a function
# of a signal-timing vector. It is intentionally bumpy (multiple local minima) to
# demonstrate why plain Hill Climbing gets trapped.

def traffic_cost(signal_timings):
    """Synthetic multi-modal cost surface over a single aggregated timing variable."""
    x = signal_timings
    return (math.sin(x * 0.9) * 12 + math.sin(x * 2.1) * 4 +
            0.015 * (x - 50) ** 2 + 20)


def hill_climbing(cost_fn, start, step=1.0, max_iter=200):
    current = start
    trace = [current]
    for _ in range(max_iter):
        neighbors = [current + step, current - step]
        best_neighbor = min(neighbors, key=cost_fn)
        if cost_fn(best_neighbor) >= cost_fn(current):
            break  # stuck: local optimum, plateau, or ridge
        current = best_neighbor
        trace.append(current)
    return current, cost_fn(current), trace


def simulated_annealing(cost_fn, start, temp=50.0, cooling=0.95, max_iter=300):
    current = start
    best, best_cost = current, cost_fn(current)
    trace = [current]
    t = temp
    for _ in range(max_iter):
        candidate = current + random.uniform(-3, 3)
        delta = cost_fn(candidate) - cost_fn(current)
        if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-6)):
            current = candidate
            if cost_fn(current) < best_cost:
                best, best_cost = current, cost_fn(current)
        trace.append(current)
        t *= cooling
    return best, best_cost, trace


def run_scenario_2():
    print("\n" + "=" * 90)
    print("SCENARIO 2: SMART CITY TRAFFIC SIGNALS  |  Hill Climbing vs Simulated Annealing")
    print("=" * 90)
    start_point = 22.0
    hc_sol, hc_cost, hc_trace = hill_climbing(traffic_cost, start_point, step=2.0)
    sa_sol, sa_cost, sa_trace = simulated_annealing(traffic_cost, start_point)
    print(f"Hill Climbing        -> timing={hc_sol:.2f}, cost={hc_cost:.2f}, steps={len(hc_trace)} (trapped in local optimum)")
    print(f"Simulated Annealing  -> timing={sa_sol:.2f}, cost={sa_cost:.2f}, steps={len(sa_trace)} (escapes local optima)")
    return {"hc_trace": hc_trace, "sa_trace": sa_trace, "cost_fn": traffic_cost}


# =================================================================================
# SCENARIO 3: MARS ROVER -> Online Search Agent under Partial Observability
# =================================================================================
# The rover only "discovers" hazards (craters/rocks) once adjacent to them, so it
# cannot plan a complete offline route in advance. It repeatedly re-plans using
# an online variant of A* (LRTA*-style: act, sense, update local knowledge, repeat).

def run_scenario_3(grid_size=6):
    print("\n" + "=" * 90)
    print("SCENARIO 3: MARS ROVER EXPLORATION  |  Online Search under Partial Observability")
    print("=" * 90)

    random.seed(7)
    true_hazards = set(random.sample(
        [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) not in [(0, 0), (grid_size - 1, grid_size - 1)]],
        6))
    start, goal = (0, 0), (grid_size - 1, grid_size - 1)

    known_hazards = set()          # what the rover has actually SENSED so far
    position = start
    path_taken = [position]
    steps, replans = 0, 0

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(cell):
        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                yield (nr, nc)

    def local_a_star(src, dst, avoid):
        frontier = [(manhattan(src, dst), 0, src, [src])]
        seen = {src: 0}
        while frontier:
            f, g, node, path = heapq.heappop(frontier)
            if node == dst:
                return path
            for nb in neighbors(node):
                if nb in avoid:
                    continue
                ng = g + 1
                if ng < seen.get(nb, math.inf):
                    seen[nb] = ng
                    heapq.heappush(frontier, (ng + manhattan(nb, dst), ng, nb, path + [nb]))
        return None

    while position != goal and steps < 100:
        route = local_a_star(position, goal, known_hazards)
        if route is None:
            print("No route found with current knowledge -- rover halts and requests help.")
            break
        replans += 1
        for next_cell in route[1:]:
            steps += 1
            # SENSE: rover discovers a hazard only when it tries to move onto it
            if next_cell in true_hazards:
                known_hazards.add(next_cell)
                break  # replan from current position
            position = next_cell
            path_taken.append(position)
            if position == goal:
                break

    print(f"Hazards actually present in terrain : {len(true_hazards)}")
    print(f"Hazards sensed/learned by rover      : {len(known_hazards)}")
    print(f"Total moves executed                 : {steps}")
    print(f"Number of re-plans triggered          : {replans}")
    print(f"Rover reached goal?                  : {position == goal}")

    return {"grid_size": grid_size, "true_hazards": true_hazards, "known_hazards": known_hazards,
            "path_taken": path_taken, "start": start, "goal": goal}


# =================================================================================
# SCENARIO 4: UNIVERSITY EXAM TIMETABLING -> CSP with Backtracking + Forward Checking
# =================================================================================
# Variables  : Courses to be scheduled
# Domains    : Available exam slots (Day-Slot pairs)
# Constraints: (a) courses sharing students cannot share a slot,
#              (b) limited halls -> at most HALL_CAPACITY exams per slot,
#              (c) precedence: some subjects must be scheduled before others.

COURSES = ["AI", "DBMS", "OS", "CN", "ML", "SE"]
SLOTS = ["Day1-9AM", "Day1-1PM", "Day2-9AM", "Day2-1PM"]
HALL_CAPACITY = 2  # max exams that can run in parallel in the same slot

# Students enrolled in overlapping courses -> those courses cannot share a slot
CONFLICTS = {
    ("AI", "ML"), ("AI", "DBMS"), ("DBMS", "OS"), ("OS", "CN"), ("CN", "SE"), ("ML", "SE"),
}
# Precedence constraint: DBMS must be scheduled before OS (slot index of DBMS < OS)
PRECEDENCE = [("DBMS", "OS")]


def conflicts_with(course, slot, assignment):
    for (c1, c2) in CONFLICTS:
        other = c2 if c1 == course else (c1 if c2 == course else None)
        if other and assignment.get(other) == slot:
            return True
    if sum(1 for c, s in assignment.items() if s == slot) >= HALL_CAPACITY and course not in assignment:
        return True
    for (before, after) in PRECEDENCE:
        if course == after and before in assignment and SLOTS.index(assignment[before]) >= SLOTS.index(slot):
            return True
        if course == before and after in assignment and SLOTS.index(slot) >= SLOTS.index(assignment[after]):
            return True
    return False


def forward_check(course, slot, domains, assignment):
    """Prunes the domains of unassigned neighboring courses; returns pruned dict or None if a wipeout occurs."""
    pruned = {}
    trial = dict(assignment)
    trial[course] = slot
    for other in COURSES:
        if other in trial:
            continue
        remaining = [s for s in domains[other] if not conflicts_with(other, s, trial)]
        if not remaining:
            return None
        pruned[other] = domains[other]
        domains[other] = remaining
    return pruned


def restore(domains, pruned):
    for course, old_domain in pruned.items():
        domains[course] = old_domain


def backtracking_csp(domains, assignment=None, nodes_explored=None):
    if assignment is None:
        assignment = {}
    if nodes_explored is None:
        nodes_explored = [0]
    if len(assignment) == len(COURSES):
        return dict(assignment)
    # Most-constrained-variable heuristic
    unassigned = [c for c in COURSES if c not in assignment]
    course = min(unassigned, key=lambda c: len(domains[c]))
    for slot in list(domains[course]):
        nodes_explored[0] += 1
        if conflicts_with(course, slot, assignment):
            continue
        assignment[course] = slot
        pruned = forward_check(course, slot, domains, assignment)
        if pruned is not None:
            result = backtracking_csp(domains, assignment, nodes_explored)
            if result:
                return result
        if pruned is not None:
            restore(domains, pruned)
        del assignment[course]
    return None


def run_scenario_4():
    print("\n" + "=" * 90)
    print("SCENARIO 4: EXAM TIMETABLING  |  CSP - Backtracking Search + Forward Checking")
    print("=" * 90)
    domains = {c: list(SLOTS) for c in COURSES}
    nodes = [0]
    solution = backtracking_csp(domains, {}, nodes)
    print(f"Variables (courses): {COURSES}")
    print(f"Domain (slots): {SLOTS}")
    print(f"Nodes explored during search: {nodes[0]}")
    if solution:
        for course in COURSES:
            print(f"   {course:6s} -> {solution[course]}")
    else:
        print("No valid timetable found under current constraints.")
    return {"solution": solution, "nodes": nodes[0]}


# =================================================================================
# SCENARIO 5: REAL-TIME STRATEGY GAME AI -> Minimax with Alpha-Beta Pruning
# =================================================================================
# Demonstrated on Tic-Tac-Toe (a minimal but complete two-player, zero-sum game)
# so the full search tree, pruning behaviour and evaluation function are all
# fully verifiable, while the same principles scale to larger RTS decision trees.

PLAYER, OPPONENT = "X", "O"


def winner(board):
    lines = [board[0:3], board[3:6], board[6:9],
             [board[0], board[3], board[6]], [board[1], board[4], board[7]], [board[2], board[5], board[8]],
             [board[0], board[4], board[8]], [board[2], board[4], board[6]]]
    for line in lines:
        if line[0] != "." and line[0] == line[1] == line[2]:
            return line[0]
    return None


def evaluate(board):
    """Evaluation function: +10 for AI win, -10 for opponent win, 0 otherwise (terminal-only here)."""
    w = winner(board)
    if w == PLAYER:
        return 10
    if w == OPPONENT:
        return -10
    return 0


def minimax(board, depth, maximizing, nodes):
    nodes[0] += 1
    score = evaluate(board)
    if score != 0 or "." not in board:
        return score
    moves = [i for i, v in enumerate(board) if v == "."]
    if maximizing:
        best = -math.inf
        for m in moves:
            board[m] = PLAYER
            best = max(best, minimax(board, depth + 1, False, nodes))
            board[m] = "."
        return best
    else:
        best = math.inf
        for m in moves:
            board[m] = OPPONENT
            best = min(best, minimax(board, depth + 1, True, nodes))
            board[m] = "."
        return best


def minimax_alpha_beta(board, depth, alpha, beta, maximizing, nodes):
    nodes[0] += 1
    score = evaluate(board)
    if score != 0 or "." not in board:
        return score
    moves = [i for i, v in enumerate(board) if v == "."]
    if maximizing:
        best = -math.inf
        for m in moves:
            board[m] = PLAYER
            best = max(best, minimax_alpha_beta(board, depth + 1, alpha, beta, False, nodes))
            board[m] = "."
            alpha = max(alpha, best)
            if beta <= alpha:
                break  # PRUNE
        return best
    else:
        best = math.inf
        for m in moves:
            board[m] = OPPONENT
            best = min(best, minimax_alpha_beta(board, depth + 1, alpha, beta, True, nodes))
            board[m] = "."
            beta = min(beta, best)
            if beta <= alpha:
                break  # PRUNE
        return best


def best_move(board, use_alpha_beta=True):
    nodes = [0]
    best_val, chosen = -math.inf, None
    for m in [i for i, v in enumerate(board) if v == "."]:
        board[m] = PLAYER
        if use_alpha_beta:
            val = minimax_alpha_beta(board, 0, -math.inf, math.inf, False, nodes)
        else:
            val = minimax(board, 0, False, nodes)
        board[m] = "."
        if val > best_val:
            best_val, chosen = val, m
    return chosen, best_val, nodes[0]


def run_scenario_5():
    print("\n" + "=" * 90)
    print("SCENARIO 5: REAL-TIME GAME AI  |  Minimax vs Alpha-Beta Pruning")
    print("=" * 90)
    # A mid-game board -- large enough branching factor left to show pruning benefit
    board = ["X", ".", "O",
             ".", "X", ".",
             ".", ".", "O"]
    move_mm, val_mm, nodes_mm = best_move(list(board), use_alpha_beta=False)
    move_ab, val_ab, nodes_ab = best_move(list(board), use_alpha_beta=True)
    print(f"Board state: {board[0:3]} / {board[3:6]} / {board[6:9]}")
    print(f"Plain Minimax     -> best cell={move_mm}, value={val_mm}, nodes explored={nodes_mm}")
    print(f"Alpha-Beta Pruned -> best cell={move_ab}, value={val_ab}, nodes explored={nodes_ab}")
    reduction = 100 * (1 - nodes_ab / nodes_mm)
    print(f"Search-space reduction from Alpha-Beta pruning: {reduction:.1f}%")
    return {"nodes_mm": nodes_mm, "nodes_ab": nodes_ab, "reduction": reduction}


# =================================================================================
# FIGURE: 5-panel summary of all scenario outputs -> saved as output.png
# =================================================================================

def build_summary_figure(r1, r2, r3, r4, r5):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CSA17 - AI Assessment Tool 2 : Scenario-wise Algorithm Outputs", fontsize=15, fontweight="bold")

    # Panel 1: Drone delivery graph with Greedy vs A* routes
    ax1 = fig.add_subplot(2, 3, 1)
    G = nx.Graph()
    for u, edges in r1["graph"].items():
        for v, w in edges.items():
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=3)
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color="#cfe3ff", node_size=650, font_size=8, edge_color="#bbbbbb")
    g_path = r1["blocked"]["greedy"][0]
    a_path = r1["blocked"]["astar"][0]
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=list(zip(a_path, a_path[1:])), edge_color="green", width=3)
    ax1.set_title(f"Scenario 1: Drone Routing (path blocked)\nA* path cost={r1['blocked']['astar'][1]}", fontsize=9)

    # Panel 2: Hill Climbing vs Simulated Annealing convergence
    ax2 = fig.add_subplot(2, 3, 2)
    xs = [i for i in range(-10, 90)]
    ys = [r2["cost_fn"](x) for x in xs]
    ax2.plot(xs, ys, color="#999999", linewidth=1)
    hc_trace = r2["hc_trace"]
    sa_trace = r2["sa_trace"]
    ax2.plot(hc_trace, [r2["cost_fn"](x) for x in hc_trace], "o-", color="red", label="Hill Climbing", markersize=3)
    ax2.plot(sa_trace, [r2["cost_fn"](x) for x in sa_trace], ".-", color="blue", alpha=0.4, label="Simulated Annealing")
    ax2.set_title("Scenario 2: Traffic Signal Cost Landscape", fontsize=9)
    ax2.legend(fontsize=7)

    # Panel 3: Mars rover grid with true hazards, sensed hazards, and path taken
    ax3 = fig.add_subplot(2, 3, 3)
    n = r3["grid_size"]
    for (r, c) in r3["true_hazards"]:
        ax3.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1, color="#ffcccc"))
    for (r, c) in r3["known_hazards"]:
        ax3.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1, color="#ff5555"))
    xs = [c + 0.5 for (r, c) in r3["path_taken"]]
    ys = [n - 1 - r + 0.5 for (r, c) in r3["path_taken"]]
    ax3.plot(xs, ys, "o-", color="green", markersize=3)
    ax3.set_xlim(0, n)
    ax3.set_ylim(0, n)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Scenario 3: Mars Rover Online Search\n(light red=undiscovered, dark red=sensed hazard)", fontsize=9)

    # Panel 4: Exam timetable as a table
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis("off")
    sol = r4["solution"]
    if sol:
        cell_text = [[c, sol[c]] for c in COURSES]
        table = ax4.table(cellText=cell_text, colLabels=["Course", "Assigned Slot"],
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
    ax4.set_title(f"Scenario 4: Exam Timetable (CSP)\nnodes explored={r4['nodes']}", fontsize=9)

    # Panel 5: Minimax vs Alpha-Beta node comparison
    ax5 = fig.add_subplot(2, 3, 5)
    bars = ax5.bar(["Minimax", "Alpha-Beta"], [r5["nodes_mm"], r5["nodes_ab"]], color=["#e67e22", "#27ae60"])
    for b in bars:
        ax5.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, str(int(b.get_height())), ha="center", fontsize=8)
    ax5.set_title(f"Scenario 5: Nodes Explored\n({r5['reduction']:.1f}% reduction via pruning)", fontsize=9)

    # Panel 6: text summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary = (
        "SUMMARY\n\n"
        f"S1 Drone: A* guarantees optimal cost\n({r1['blocked']['astar'][1]}) vs Greedy "
        f"({r1['blocked']['greedy'][1]})\n\n"
        "S2 Traffic: Simulated Annealing escapes\nlocal minima that trap Hill Climbing\n\n"
        f"S3 Rover: reached goal="
        f"{r3['path_taken'][-1] == r3['goal']}, re-planned\nusing online search under partial info\n\n"
        f"S4 Timetable: CSP solved with\n{r4['nodes']} nodes explored (forward checking)\n\n"
        f"S5 Game AI: Alpha-Beta cut search by\n{r5['reduction']:.1f}% vs plain Minimax"
    )
    ax6.text(0, 1, summary, fontsize=8.5, va="top", family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("output.png", dpi=150, bbox_inches="tight")
    print("\nSaved summary visualization to output.png")


if __name__ == "__main__":
    r1 = run_scenario_1()
    r2 = run_scenario_2()
    r3 = run_scenario_3()
    r4 = run_scenario_4()
    r5 = run_scenario_5()
    build_summary_figure(r1, r2, r3, r4, r5)
    print("\n" + "=" * 90)
    print("ALL SCENARIOS EXECUTED SUCCESSFULLY")
    print("=" * 90)
