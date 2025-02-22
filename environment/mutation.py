"""
Mutation functions for genetic algorithms.
"""

import random

def mutate(weights, mutation_rate=0.1):
    """
    Mutate a list of weights by adding small random changes.
    """
    return [w + random.uniform(-mutation_rate, mutation_rate) if random.random() < mutation_rate else w for w in weights]

if __name__ == "__main__":
    print("Mutated weights:", mutate([0.5, 0.8, 0.3]))
