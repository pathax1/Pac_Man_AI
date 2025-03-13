# config.py

import torch

# Hyperparameters (DQN)
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 50000


EPS_START = 1.0
EPS_END = 0.01
# EPS_DECAY can be interpreted in different ways (episodes vs. steps).
# For demonstration, we’ll do an episode-based decay.
EPS_DECAY = 500

TARGET_UPDATE = 10       # Update target network every N episodes
NUM_EPISODES = 50000     # Total training episodes

# Q-Learning
QL_ALPHA = 0.1           # Learning rate for Q-learning
QL_EPSILON = 1.0
QL_EPSILON_MIN = 0.01
QL_EPSILON_DECAY = 0.995

# Monte Carlo
MC_EPISODES = 50000
MC_GAMMA = 0.99

# Device configuration: use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
