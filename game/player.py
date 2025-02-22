import pygame

class Pacman:
    def __init__(self, position):
        self.position = position

    def get_action(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 0
        elif keys[pygame.K_DOWN]:
            return 1
        elif keys[pygame.K_LEFT]:
            return 2
        elif keys[pygame.K_RIGHT]:
            return 3
        return -1  # No action