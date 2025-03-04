# environment/constants.py
import pygame

# Screen settings
TILE_SIZE = 32
SCREEN_WIDTH = 21 * TILE_SIZE  # 21 columns
SCREEN_HEIGHT = 23 * TILE_SIZE  # 23 rows
FPS = 10

# Colors (RGB)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# Rewards
REWARD_STEP = -1
REWARD_PELLET = 5
REWARD_POWER_PELLET = 10
REWARD_WIN = 10
REWARD_GHOST_EATEN = 20
REWARD_GHOST_COLLISION = -50

# Power pellet duration (frames)
POWER_DURATION = FPS * 5  # e.g., 10 seconds of power mode
