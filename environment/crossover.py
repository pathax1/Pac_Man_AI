import numpy as np

def crossover_maps(parent1, parent2):
    crossover_point = np.random.randint(1, parent1.shape[0] - 1)
    child1 = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    child2 = np.vstack((parent2[:crossover_point, :], parent1[crossover_point:, :]))
    return child1, child2