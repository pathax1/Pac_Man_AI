import torch

# Hyperparameters
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10    # Update target network every 10 episodes
NUM_EPISODES = 500    # Total training episodes

# Device configuration: use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
