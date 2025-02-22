import gym
import numpy as np
import random
from gym import spaces
from config.settings import (
    MAZE_LAYOUT, MAZE_WIDTH, MAZE_HEIGHT,
    WALL, PELLET, POWER,
    PELLET_REWARD, POWER_REWARD, STEP_PENALTY, GHOST_COLLISION_PENALTY
)

class PacmanEnv(gym.Env):
    """
    Gym environment for training RL on a tile-based Pac-Man maze.
    Pac-Man moves in discrete steps, ghosts move randomly.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(MAZE_HEIGHT, MAZE_WIDTH),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)
        self.maze = None
        self.pacman_pos = None
        self.ghosts = None
        self.done = False
        self.reset()

    def reset(self):
        self.maze = np.array(MAZE_LAYOUT, dtype=int)
        # Place Pac-Man near bottom center
        self.pacman_pos = [23, 13]
        # 4 ghosts in the ghost house region (center)
        self.ghosts = [[15,12],[15,13],[15,14],[15,15]]
        self.done = False
        return self._get_obs()

    def step(self, action):
        reward = STEP_PENALTY
        self._move_pacman(action)

        # Eat pellet / power
        r, c = self.pacman_pos
        if self.maze[r,c] == PELLET:
            reward += PELLET_REWARD
            self.maze[r,c] = 0
        elif self.maze[r,c] == POWER:
            reward += POWER_REWARD
            self.maze[r,c] = 0

        # Move ghosts
        for i in range(len(self.ghosts)):
            self.ghosts[i] = self._move_ghost_random(self.ghosts[i])

        # Check collision
        if any(tuple(self.pacman_pos) == tuple(g) for g in self.ghosts):
            reward += GHOST_COLLISION_PENALTY
            self.done = True

        # Check if pellets are done
        if not (2 in self.maze or 3 in self.maze):
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        # Typically integrated with Pygame for visual, but left empty for training
        pass

    def _move_pacman(self, action):
        # 0=up,1=down,2=left,3=right
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        nr = self.pacman_pos[0] + moves[action][0]
        nc = self.pacman_pos[1] + moves[action][1]
        if self.maze[nr,nc] != WALL:
            self.pacman_pos = [nr,nc]

    def _move_ghost_random(self, pos):
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        dr,dc = random.choice(moves)
        nr = pos[0]+dr
        nc = pos[1]+dc
        if self.maze[nr,nc] != WALL:
            return [nr,nc]
        return pos

    def _get_obs(self):
        return self.maze
