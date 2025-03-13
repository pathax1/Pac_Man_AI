# environment/pacman_env.py
import gym
import numpy as np
from gym import spaces
from .game import PacmanGame

class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, level="simple"):
        super().__init__()
        self.game = PacmanGame(level=level)

        # Action space: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # State shape: [pacman_row, pacman_col, ghost1_row, ghost1_col, ...,
        #               powered_flag, pellets_left]
        # For 4 ghosts, that’s 1 Pac-Man position (2D) + 4 ghost positions (2D each = 8) + 1 powered + 1 pellets_left = 2 + 8 + 1 + 1 = 12
        high = np.array([self.game.rows, self.game.cols] * 5 + [1, self.game.total_pellets], dtype=np.float32)
        low  = np.zeros_like(high, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        state = self.game.reset()
        return np.array(state, dtype=np.float32)

    def step(self, action):
        state, reward, done, info = self.game.step(action)
        return np.array(state, dtype=np.float32), float(reward), done, info

    def render(self, mode='human'):
        self.game.render()

    def close(self):
        self.game.close()

    def get_state(self):
        state = [
            self.pacman_row, self.pacman_col,
            *[coord for ghost in self.ghosts for coord in (ghost["row"], ghost["col"])],
            int(self.powered),
            self.pellets_left
        ]
        return tuple(state)
