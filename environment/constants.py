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

# Reward shaping (Improved)
REWARD_STEP = -0.1                # Small penalty per step (forces faster completion)
REWARD_PELLET = 5.0               # Increased reward for eating a pellet
REWARD_POWER_PELLET = 10.0        # More valuable power pellets
REWARD_GHOST_EATEN = 15.0         # Balanced reward for eating ghosts
REWARD_GHOST_COLLISION = -100.0   # Huge penalty for getting caught by a ghost
REWARD_WIN = 500.0                # MASSIVE reward for finishing the maze


# Power pellet duration
POWER_DURATION = 100  # Frames or steps (tweak as needed)

pygame.init()
pygame.mixer.init()
