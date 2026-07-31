import math

# Grid Configuration
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

# Actual Environment (Unknown to robot initially)
REAL_OBSTACLES = {(1, 1), (1, 2), (2, 1), (3, 3)}
REAL_RISKY_ZONES = {(2, 2), (3, 2)}  # Cost = 3

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right


def manhattan_distance(pos):
    """Heuristic h(n): Manhattan distance to Goal."""
    return abs(pos[0] - GOAL[0]) + abs(pos[1] - GOAL[1])


class OnlineRescueRobot:
    def __init__(self, start, goal):
        self.current_pos = start
        self.goal = goal
        self.known_obstacles = set()
        self.known_risky = set()
        # Heuristic lookup table h(s) initialized with Manhattan distance
        self.h_table = {}

    def get_h(self, pos):
        if pos not in self.h_table:
            self.h_table[pos] = manhattan_distance(pos)
        return self.h_table[pos]

    def sense_adjacent(self):
        """Senses adjacent cells and updates internal map knowledge."""
        r, c = self.current_pos
        for dr, dc in DIRECTIONS:
            adj = (r + dr, c + dc)
            if 0 <= adj[0] < GRID_SIZE and 0 <= adj[1] < GRID_SIZE:
                if adj in REAL_OBSTACLES:
                    self.known_obstacles.add(adj)
                elif adj in REAL_RISKY_ZONES:
                    self.known_risky.add(adj)

    def get_step_cost(self, neighbor):
        """Returns step cost: 3 for risky cells, 1 for normal clear cells."""
        return 3 if neighbor in self.known_risky else 1

    def step(self):
        """Executes one online search step using LRTA* decision rule."""
        self.sense_adjacent()

        best_move = None
        min_f = math.inf

        r, c = self.current_pos
        # Evaluate valid unblocked adjacent cells
        for dr, dc in DIRECTIONS:
            neighbor = (r + dr, c + dc)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if neighbor not in self.known_obstacles:
                    cost = self.get_step_cost(neighbor)
                    f_val = cost + self.get_h(neighbor)

                    if f_val < min_f:
                        min_f = f_val
                        best_move = neighbor

        # Update local heuristic (Learning step to prevent dead-end loops)
        self.h_table[self.current_pos] = min_f
        # Move robot
        self.current_pos = best_move
        return best_move, self.get_step_cost(best_move)


def run_simulation():
    robot = OnlineRescueRobot(START, GOAL)
    total_cost = 0
    path = [START]

    print("=" * 55)
    print("      ONLINE RESCUE ROBOT NAVIGATION SIMULATION      ")
    print("=" * 55)

    step = 0
    while robot.current_pos != GOAL:
        step += 1
        pos, move_cost = robot.step()
        total_cost += move_cost
        path.append(pos)
        cell_type = "Risky (Cost=3)" if pos in REAL_RISKY_ZONES else "Clear (Cost=1)"
        print(f"Step {step:2d}: Moved to {pos} | Cell: {cell_type} | Path Cost: {total_cost}")

    print("\n" + "=" * 55)
    print(">>> SURVIVOR LOCATED SUCCESSFULLY! <<<")
    print(f"Optimal Path Taken: {path}")
    print(f"Total Path Cost: {total_cost}")
    print("=" * 55)


if __name__ == "__main__":
    run_simulation()