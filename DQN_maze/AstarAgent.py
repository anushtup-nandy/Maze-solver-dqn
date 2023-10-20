import numpy as np
import cv2
import heapq
from environ import Environment

class AStarAgent:

    def __init__(self, environment):
        self.environment = environment

    def heuristic(self, state, goal):
        # Euclidean distance is used as the heuristic
        state = self.environment.init_state
        return np.linalg.norm(state - goal)

    def is_valid_state(self, state):
        # Check if a state is within the valid boundaries of the environment
        return 0 <= state[0] <= 1 and 0 <= state[1] <= 1

    def undiscretize_state(self, state):
        # Convert a discrete state to its continuous representation
        x = state[0] * self.environment.width
        y = state[1] * self.environment.height
        return x, y

    def discretize_state(self, state):
        # Convert a continuous state to its discrete representation
        x = state[0] / self.environment.width
        y = state[1] / self.environment.height
        return x, y

    def search(self, start_state, goal_state):
        # Perform A* search to find a path from the start state to the goal state
        start_node = self.discretize_state(start_state)
        goal_node = self.discretize_state(goal_state)

        open_list = [(0, start_node)]
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self.heuristic(start_node, goal_node)}

        while open_list:
            current = heapq.heappop(open_list)[1]

            if current == goal_node:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                path.reverse()
                return [self.undiscretize_state(node) for node in path]

            neighbors = [(current[0] - 1, current[1]), (current[0] + 1, current[1]),
                         (current[0], current[1] - 1), (current[0], current[1] + 1)]

            for neighbor in neighbors:
                if not self.is_valid_state(neighbor):
                    continue

                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal_node)

                    if neighbor not in [node[1] for node in open_list]:
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None

    def plan_path(self, start_state, goal_state):
        # Plan a path from the start state to the goal state using A*
        path = self.search(start_state, goal_state)

        if path is not None:
            # Smooth the path
            smoothed_path = self.smooth_path(path)
            return smoothed_path

        return None

    def smooth_path(self, path):
        # Smooth the given path using a simple line-of-sight algorithm
        smoothed_path = [path[0]]
        current = 0
        last_visible = 0

        for i in range(len(path)):
            if self.line_of_sight(smoothed_path[current], path[i]):
                last_visible = i

        smoothed_path.append(path[last_visible])

        return smoothed_path

    def line_of_sight(self, start, end):
        # Check if there is a line of sight between two states (no obstacles in between)
        x0, y0 = start
        x1, y1 = end

        # Calculate the step size for interpolation
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        step = max(dx, dy)

        # Perform interpolation and check for obstacles
        for i in range(step + 1):
            x = int(x0 + i * (x1 - x0) / step)
            y = int(y0 + i * (y1 - y0) / step)

            if not self.is_valid_state((x, y)):
                return False

        return True

    def execute_path(self, path):
        # Execute the planned path in the environment and return the final distance to the goal
        current_state = self.environment.reset()
        total_distance = 0.0

        for target_state in path:
            action = target_state - current_state
            next_state, distance_to_goal = self.environment.step(current_state, action)
            current_state = next_state
            total_distance += distance_to_goal
            self.environment.show(current_state)

        return total_distance

    def run(self, start_state, goal_state):
        # Run the A* agent in the environment
        path = self.plan_path(start_state, goal_state)

        if path is not None:
            distance_to_goal = self.execute_path(path)
            print("Path found! Distance to goal:", distance_to_goal)
        else:
            print("No path found!")

# Create an instance of the environment
env = Environment()

# Create an instance of the A* agent
agent = AStarAgent(env)

# Set the start and goal states
start_state = env.reset()
goal_state = env.goal_state

# Run the agent in the environment
agent.run(start_state, goal_state)
