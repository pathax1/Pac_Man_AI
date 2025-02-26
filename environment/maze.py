import numpy as np

def generate_maze(grid_size):
    """
    Generates a semi-handcrafted maze:
      - A border of walls.
      - A symmetrical arrangement of corridors in the center.
      - Some random walls/pellets for variation.
    Returns a 2D numpy array with:
      0 = floor
      1 = wall
      2 = pellet
      3 = power pellet
    """
    maze = np.zeros((grid_size, grid_size), dtype=int)

    # 1) Create solid outer walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # 2) Create a symmetrical corridor pattern
    # Example: place walls in a grid-like structure
    for y in range(2, grid_size-2, 4):
        for x in range(2, grid_size-2, 4):
            # A small 2x2 block of walls
            maze[y:y+2, x:x+2] = 1

    # 3) Random walls
    # Add a few random walls to keep it interesting
    for _ in range(grid_size // 2):
        rx = np.random.randint(1, grid_size-1)
        ry = np.random.randint(1, grid_size-1)
        if maze[ry, rx] == 0:
            maze[ry, rx] = 1

    # 4) Place pellets in empty floor cells
    # We'll skip placing them on walls or the border
    for y in range(1, grid_size-1):
        for x in range(1, grid_size-1):
            if maze[y, x] == 0:
                # 80% chance normal pellet, 20% chance floor
                if np.random.rand() < 0.8:
                    maze[y, x] = 2

    # 5) Place a few power pellets
    # e.g., 4 power pellets in corners inside the walls
    corners = [
        (1,1), (1, grid_size-2), (grid_size-2, 1), (grid_size-2, grid_size-2)
    ]
    for (cx, cy) in corners:
        if maze[cy, cx] == 2 or maze[cy, cx] == 0:
            maze[cy, cx] = 3

    return maze
