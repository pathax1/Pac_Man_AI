import random
from config.settings import GHOST_SPEED

class GhostAgent:
    def __init__(self, strategy="chase"):
        self.strategy = strategy

    def choose_action(self, pacman_pos, ghost_pos):
        if self.strategy == "chase":
            return self.move_towards(pacman_pos, ghost_pos)
        elif self.strategy == "random":
            return random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        elif self.strategy == "ambush":
            future_pos = [pacman_pos[0] + 2, pacman_pos[1] + 2]
            return self.move_towards(future_pos, ghost_pos)
        elif self.strategy == "avoid":
            return self.move_away_from(pacman_pos, ghost_pos)

    def move_towards(self, target, current):
        dx = int((target[0] - current[0]) > 0) - int((target[0] - current[0]) < 0)
        dy = int((target[1] - current[1]) > 0) - int((target[1] - current[1]) < 0)
        return dx, dy

    def move_away_from(self, target, current):
        dx = int((target[0] - current[0]) < 0) - int((target[0] - current[0]) > 0)
        dy = int((target[1] - current[1]) < 0) - int((target[1] - current[1]) > 0)
        return dx, dy