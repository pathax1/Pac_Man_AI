"""
Pac-Man Agent
This agent controls Pac-Man's movements in the game.
"""

import numpy as np

class PacmanAgent:
    def __init__(self):
        self.speed = 1.0
        self.direction = [0, 0]

    def decide_move(self, observation):
        # Example logic: move randomly
        moves = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        return moves[np.random.randint(0, len(moves))]

if __name__ == "__main__":
    agent = PacmanAgent()
    move = agent.decide_move(None)
    print("Pac-Man move:", move)
