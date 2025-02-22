"""
Selection function for genetic algorithm.
"""

def select_parents(population, fitnesses, num_parents=2):
    """
    Select parents based on fitness-proportional selection.
    """
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    parents = []
    for _ in range(num_parents):
        r = np.random.rand()
        cumulative = 0
        for individual, prob in zip(population, probabilities):
            cumulative += prob
            if r < cumulative:
                parents.append(individual)
                break
    return parents

if __name__ == "__main__":
    import numpy as np
    pop = [[1,2,3], [4,5,6], [7,8,9]]
    fits = [10, 20, 30]
    print("Selected parents:", select_parents(pop, fits))
