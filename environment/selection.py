import random

def tournament_selection(population, fitness_scores, tournament_size=3):
    participants = random.sample(list(zip(population, fitness_scores)), tournament_size)
    return max(participants, key=lambda x: x[1])[0]