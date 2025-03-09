# environment/constants.py

import pygame

# Screen / Grid
TILE_SIZE = 24
FPS = 10

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE  = (0, 0, 255)

# Screen size can be set based on the largest maze dimension
SCREEN_WIDTH = 21 * TILE_SIZE
SCREEN_HEIGHT = 23 * TILE_SIZE

# Reward shaping
REWARD_STEP            = -0.04  # Slight negative reward per step to encourage shorter paths
REWARD_PELLET          = 1.0
REWARD_POWER_PELLET    = 3.0
REWARD_GHOST_EATEN     = 5.0
REWARD_GHOST_COLLISION = -10.0
REWARD_WIN             = 50.0

# Power pellet duration
POWER_DURATION = 100  # Frames or steps (tweak as needed)

pygame.init()
pygame.mixer.init()
