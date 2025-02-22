"""
Generates the maze based on the layout in settings.
"""
import numpy as np
from config.settings import MAZE_LAYOUT, WALL, PELLET, EMPTY

def generate_maze():
    """
    Convert the 2D list in MAZE_LAYOUT to a NumPy array for easier handling.
    """
    return np.array(MAZE_LAYOUT, dtype=int)
