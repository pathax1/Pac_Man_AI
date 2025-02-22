import numpy as np

def mutate_map(map_grid, mutation_rate):
    mutation_mask = np.random.rand(*map_grid.shape) < mutation_rate
    map_grid[mutation_mask] = 1 - map_grid[mutation_mask]  # Flip 0 to 1 and vice versa
    return map_grid