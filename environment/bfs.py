# environment/bfs.py

from collections import deque

def bfs(start, goal, grid, rows, cols):
    """
    A simple BFS that finds the first step from start to goal.
    grid: 2D list with 0 = passable, 1 = wall.
    start, goal: (row, col) tuples.
    Returns: (dr, dc) for the first move along the found path (or (0,0) if already at goal or no path).
    """
    queue = deque([(start[0], start[1], [])])
    visited = set([start])
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == goal:
            return path[0] if path else (0, 0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [(dr, dc)]))
    return (0, 0)
