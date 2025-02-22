"""
Crossover operations for genetic algorithm components.
"""

def crossover(parent1, parent2):
    """
    Perform crossover between two parents.
    For example, average their weights.
    """
    return [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]

if __name__ == "__main__":
    print("Crossover result:", crossover([1, 2, 3], [4, 5, 6]))
