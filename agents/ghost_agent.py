"""
Ghost Agent for Pac-Man.
This module defines the behavior of ghosts controlled by AI.
"""

import numpy as np

class GhostAgent:
    def __init__(self):
        # Initialize ghost parameters
        self.speed = 1.0

    def choose_direction(self, ghost_pos, pacman_pos, maze):
        # Simple chase algorithm: move towards Pac-Man
        direction = np.sign(np.array(pacman_pos) - np.array(ghost_pos))
        return direction.tolist()

if __name__ == "__main__":
    # Test ghost agent
    agent = GhostAgent()
    direction = agent.choose_direction([5, 5], [10, 10], None)
    print("Chosen direction:", direction)
