import pygame
from config import settings

class Player:
    def __init__(self, start_pos):
        self.row, self.col = start_pos
        self.speed = settings.PACMAN_SPEED
        self.color = settings.YELLOW

    def update(self, maze):
        keys = pygame.key.get_pressed()
        delta_r, delta_c = 0, 0
        if keys[pygame.K_UP]:
            delta_r = -1
        elif keys[pygame.K_DOWN]:
            delta_r = 1
        elif keys[pygame.K_LEFT]:
            delta_c = -1
        elif keys[pygame.K_RIGHT]:
            delta_c = 1

        new_r = self.row + delta_r
        new_c = self.col + delta_c

        # Check if it's not a wall
        if maze[new_r, new_c] != settings.WALL:
            self.row = new_r
            self.col = new_c
