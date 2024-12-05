import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from typing import Dict, Tuple


class Maze:
    def __init__(self, height: int, width: int, seed=None):
        self.height = height
        self.width = width
        self.maze = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.generate_maze()

    def generate_maze_old(self):
        # Initialize the maze with walls
        self.maze = np.ones((self.height * 2 + 1, self.width * 2 + 1), dtype=int)
        
        # Create a list of all possible walls
        walls = [(x, y) for x in range(1, self.height * 2, 2) for y in range(1, self.width * 2, 2)]
        random.shuffle(walls)

        # Start with the cell at (1, 1)
        self.maze[1, 1] = 0
        walls.append((1, 1))

        while walls:
            x, y = walls.pop()
            neighbors = [(x + dx, y + dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]
                         if 0 < x + dx < self.height * 2 and 0 < y + dy < self.width * 2]
            
            unvisited = [n for n in neighbors if self.maze[n] == 1]
            
            if unvisited:
                nx, ny = random.choice(unvisited)
                self.maze[nx, ny] = 0
                self.maze[(x + nx) // 2, (y + ny) // 2] = 0
                walls.append((nx, ny))

    def generate_maze(self):
        # Initialize the maze with walls
        self.maze = np.ones((self.height * 2 + 1, self.width * 2 + 1), dtype=int)
        
        def dfs(x, y):
            self.maze[y, x] = 0
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if 0 <= nx < self.width * 2 + 1 and 0 <= ny < self.height * 2 + 1 and self.maze[ny, nx] == 1:
                    self.maze[y + dy, x + dx] = 0
                    dfs(nx, ny)

        # Start DFS from (1, 1)
        dfs(1, 1)

        # Ensure path to goal
        goal_y, goal_x = self.height * 2 - 1, self.width * 2 - 1
        self.maze[goal_y, goal_x] = 0
        if self.maze[goal_y - 1, goal_x] == 1 and self.maze[goal_y, goal_x - 1] == 1:
            if random.choice([True, False]):
                self.maze[goal_y - 1, goal_x] = 0
            else:
                self.maze[goal_y, goal_x - 1] = 0

    def extract_graph(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Extract the graph representation of the maze. The graph is represented as a dictionary where each key is a point.

        Returns:
            Dict[Tuple[int, int], Tuple[int, int]]: A dictionary representing the graph of the maze.
        """
        start = (1, 1)
        graph = {}
        visited = set()
        queue = deque()
        queue.append(start)

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
            graph[current] = neighbors
            visited.add(current)

        return graph

    def compute_distances_from_start(self) -> Dict[Tuple[int, int], int]:
        """Compute the distances from the start point to all other points in the maze.

        Returns:
            Dict[Tuple[int, int], int]: A dictionary mapping each point to its distance from the start point.
        """
        graph = self.extract_graph()
        start = (1, 1)
        distances = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.maze.shape[0] and 0 <= ny < self.maze.shape[1] and self.maze[nx, ny] == 0:
                neighbors.append((nx, ny))

        return neighbors

    def draw_maze(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap='binary')
        plt.title(f"Maze ({self.height}x{self.width})")
        plt.axis('off')
        plt.show()

    def draw_maze_with_graph(self, draw_graph: bool = True):
        plt.figure(figsize=(12, 12))
        plt.imshow(self.maze, cmap='binary')
        plt.title(f"Maze ({self.height}x{self.width})")

        if draw_graph:
            graph = self.extract_graph()
            for node, neighbors in graph.items():
                y, x = node
                for neighbor in neighbors:
                    ny, nx = neighbor
                    plt.plot([x, nx], [y, ny], 'r-', linewidth=2, alpha=0.7)

            # Highlight start and end points
            plt.plot(1, 1, 'go', markersize=10)  # Start point
            plt.plot(self.width*2-1, self.height*2-1, 'bo', markersize=10)  # End point

        plt.axis('off')
        plt.show()

    def get_start_point(self):
        return 1, 1

    def get_goal_point(self):
        return self.width * 2 - 1, self.height * 2 - 1


if __name__ == "__main__":
    maze = Maze(5, 10)
    distances = maze.compute_distances_from_start()
    print(distances)
    maze.draw_maze_with_graph()

