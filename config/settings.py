"""
Arcade-like Pac-Man settings with an arcade-like maze layout.
"""

import os

# Paths to tile images (ensure these paths point to valid image files)
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
TILE_ASSETS = {
    "wall": os.path.join(ASSETS_DIR, "tiles", "wall.png"),
    "pellet": os.path.join(ASSETS_DIR, "tiles", "pellet.png"),
    "power": os.path.join(ASSETS_DIR, "tiles", "power_pellet.png"),
    "floor": os.path.join(ASSETS_DIR, "tiles", "floor.png"),
}
PACMAN_IMAGE = os.path.join(ASSETS_DIR, "pacman", "pacman.png")
GHOST_IMAGES = [
    os.path.join(ASSETS_DIR, "ghosts", "ghost_red.png"),
    os.path.join(ASSETS_DIR, "ghosts", "ghost_pink.png"),
    os.path.join(ASSETS_DIR, "ghosts", "ghost_cyan.png"),
    os.path.join(ASSETS_DIR, "ghosts", "ghost_orange.png"),
]

# Tile codes
EMPTY = 0    # Floor
WALL = 1
PELLET = 2
POWER = 3    # Power pellet

# Maze and Screen Settings
# Increase TILE_SIZE to make the grid larger (e.g., from 16 to 32)
TILE_SIZE = 32
MAZE_WIDTH = 28
MAZE_HEIGHT = 31
SCREEN_WIDTH = MAZE_WIDTH * TILE_SIZE  # 28*32 = 896 px
SCREEN_HEIGHT = MAZE_HEIGHT * TILE_SIZE  # 31*32 = 992 px
FPS = 60

# Rewards
PELLET_REWARD = 10
POWER_REWARD = 50
STEP_PENALTY = -1
GHOST_COLLISION_PENALTY = -100

# Speeds
PACMAN_SPEED = 1
GHOST_SPEED = 1

# Expanded MAZE_LAYOUT: 31 rows × 28 columns
# (This layout is an approximation—feel free to tweak it.)
MAZE_LAYOUT = [
    [1]*28,  # Row 0: Top border
    [1,3] + [2]*24 + [3,1],  # Row 1
    [1,2] + [1]*24 + [2,1],  # Row 2
    [1,2,1] + [0]*22 + [1,2,1],  # Row 3
    [1,2,1,0] + [1]*20 + [0,1,2,1],  # Row 4
    [1,2,1,0] + [2]*20 + [0,1,2,1],  # Row 5
    [1,2,1,0] + ([2,1]*10) + [0,1,2,1],  # Row 6
    [1,2,1,0] + [0]*20 + [0,1,2,1],  # Row 7
    [1,2,1,0] + [2]*20 + [0,1,2,1],  # Row 8
    [1,2,1,0] + [1]*20 + [0,1,2,1],  # Row 9
    [1,2,1,0] + [1]*20 + [0,1,2,1],  # Row 10
    [1,2,1,0] + [0]*20 + [0,1,2,1],  # Row 11
    [1,2,1,0] + [2]*20 + [0,1,2,1],  # Row 12
    [1,2,1,0] + [2]*20 + [0,1,2,1],  # Row 13
    [1,2,1,0] + [2]*20 + [0,1,2,1],  # Row 14
    [1,2,1,0] + [0]*20 + [0,1,2,1],  # Row 15
    [1,2,1,0] + [1]*20 + [0,1,2,1],  # Row 16
    [1,2,1,0] + [1]*20 + [0,1,2,1],  # Row 17
    [1,2,1,0] + [2]*20 + [0,1,2,1],  # Row 18
    [1,2,1,0] + [0]*20 + [0,1,2,1],  # Row 19
    [1,2,1,0] + [1]*20 + [0,1,2,1],  # Row 20
    [1,2,1,0] + [2]*20 + [0,1,2,1],  # Row 21
    [1,2,1,0] + [1]*20 + [0,1,2,1],  # Row 22
    [1,2,1,0] + [0]*20 + [0,1,2,1],  # Row 23
    [1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1],  # Row 24
    [1,3] + [2]*24 + [3,1],  # Row 25
    [1]*28,  # Row 26
    [1,3] + [2]*24 + [3,1],  # Row 27
    [1,2] + [1]*24 + [2,1],  # Row 28
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],  # Row 29
    [1]*28,  # Row 30: Bottom border
]
