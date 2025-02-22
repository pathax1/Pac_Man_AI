import numpy as np
from environment.fitness import evaluate_fitness
from environment.selection import tournament_selection
from environment.crossover import crossover_maps
from environment.mutation import mutate_map

POPULATION_SIZE = 20
GENERATIONS = 50
MUTATION_RATE = 0.1

def generate_map(grid_size):
    population = [np.random.randint(0, 2, size=grid_size) for _ in range(POPULATION_SIZE)]

    for _ in range(GENERATIONS):
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = tournament_selection(population, fitness_scores), tournament_selection(population, fitness_scores)
            child1, child2 = crossover_maps(parent1, parent2)
            new_population.extend([mutate_map(child1, MUTATION_RATE), mutate_map(child2, MUTATION_RATE)])
        population = new_population

    best_map = population[np.argmax([evaluate_fitness(ind) for ind in population])]
    return best_map