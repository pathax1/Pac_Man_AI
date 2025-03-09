# environment/bfs.py

from collections import deque

def bfs(start, goal, grid, rows, cols):
    """
    grid[r][c] == 1 means WALL
    0 means passable
    Returns a (dr, dc) step from start to move towards goal.
    If no path, returns (0, 0).
    """
    (sr, sc) = start
    (gr, gc) = goal

    if (sr, sc) == (gr, gc):
        return (0, 0)

    visited = set()
    queue = deque()
    queue.append((sr, sc))
    visited.add((sr, sc))

    parents = {(sr, sc): None}

    while queue:
        r, c = queue.popleft()
        if (r, c) == (gr, gc):
            break

        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parents[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    # Reconstruct path if we reached the goal
    if (gr, gc) not in parents:
        # No path found
        return (0, 0)

    path = []
    current = (gr, gc)
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()

    # The first element is the start itself, the second is the next step
    if len(path) > 1:
        next_cell = path[1]
        return (next_cell[0] - sr, next_cell[1] - sc)
    else:
        return (0, 0)
