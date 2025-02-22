import gym
from gym import spaces
import numpy as np
from environment.map_generator import generate_map

class PacmanEnv(gym.Env):
    """
    Gym environment for the Pac-Man game with support for different agent types.
    """
    def __init__(self, agent_type="dqn"):
        super(PacmanEnv, self).__init__()
        self.agent_type = agent_type.lower()
        self.grid_size = (15, 15)  # Example grid size
        self.map = generate_map(self.grid_size)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Actions: 0-Up, 1-Down, 2-Left, 3-Right
        self.observation_space = spaces.Box(low=0, high=4, shape=self.grid_size, dtype=np.int8)

        # Initialize game state
        self.reset()

    def reset(self):
        """Resets the game environment to the initial state."""
        self.map = generate_map(self.grid_size)
        self.pacman_position = (1, 1)
        self.ghost_positions = [(13, 13)]
        self.score = 0
        return self.map

    def step(self, action):
        """Executes an action and returns the next state, reward, done flag, and info."""
        x, y = self.pacman_position

        # Update Pac-Man's position based on the action
        if action == 0 and x > 0 and self.map[x - 1][y] != 1:  # Up
            x -= 1
        elif action == 1 and x < self.grid_size[0] - 1 and self.map[x + 1][y] != 1:  # Down
            x += 1
        elif action == 2 and y > 0 and self.map[x][y - 1] != 1:  # Left
            y -= 1
        elif action == 3 and y < self.grid_size[1] - 1 and self.map[x][y + 1] != 1:  # Right
            y += 1

        self.pacman_position = (x, y)
        reward = 1 if self.map[x][y] == 2 else 0  # Reward for collecting a pellet
        self.score += reward
        self.map[x][y] = 0  # Remove pellet after being eaten

        # End the game after collecting 10 pellets
        done = self.score >= 10
        return self.map, reward, done, {}

    def render(self, mode='human'):
        """Renders the game map in the console."""
        for row in self.map:
            print(' '.join(map(str, row)))