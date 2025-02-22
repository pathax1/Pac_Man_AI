import numpy as np
from scipy.ndimage import label

def evaluate_fitness(map_grid):
    # Fitness considers connectivity and pellet accessibility
    structure = np.ones((3, 3), dtype=int)  # Allow diagonal connections
    labeled, num_features = label(map_grid == 0, structure=structure)

    connectivity_score = 1 if num_features == 1 else 0
    pellet_score = np.sum(map_grid == 2) / (map_grid.size * 0.1)  # Aim for 10% pellets

    return connectivity_score + pellet_score