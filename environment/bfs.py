# environment/bfs.py
from collections import deque


def bfs(start, goal, grid, rows, cols):
    """
    Perform BFS on grid from start to goal.
    grid: 2D list where 0 is free path and 1 is wall.
    Returns first step direction (dr, dc).
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()
    queue = deque([(start, [])])
    visited.add(start)

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            if path:
                return path[0]
            else:
                return (0, 0)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(dr, dc)]))
    return (0, 0)
