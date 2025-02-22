"""
Fitness function for evaluating game states.
"""

def compute_fitness(score, steps):
    """
    Example: higher score and fewer steps yield higher fitness.
    """
    return score / (steps + 1)

if __name__ == "__main__":
    print("Fitness:", compute_fitness(100, 50))
