import random
import pygame
from config import settings

class Ghost:
    def __init__(self, start_pos):
        self.row, self.col = start_pos
        self.speed = settings.GHOST_SPEED
        self.color = settings.RED

    def update(self, maze):
        # Move randomly
        action = random.choice([0,1,2,3])  # up, down, left, right
        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_r = self.row + delta[action][0]
        new_c = self.col + delta[action][1]
        # Check if it's a wall
        if maze[new_r, new_c] != settings.WALL:
            self.row = new_r
            self.col = new_c
