# environment/constants.py

SCREEN_WIDTH = 672    # 21 columns * 32 px
SCREEN_HEIGHT = 736   # 23 rows * 32 px
FPS = 60             # Lowered from 30 to slow down animation

TILE_SIZE = 32

# Colors (R, G, B)
BLACK  = (0, 0, 0)
BLUE   = (0, 0, 255)
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 0)

# Rewards
REWARD_STEP            = -1
REWARD_PELLET          = 10
REWARD_POWER_PELLET    = 20
REWARD_GHOST_EATEN     = 50
REWARD_GHOST_COLLISION = -100
REWARD_WIN             = 200

# Duration (in frames) of Pac-Man's power state
POWER_DURATION = 300

# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTIONS = [0, 1, 2, 3]
NUM_ACTIONS = len(ACTIONS)
