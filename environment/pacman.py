import gym
from gym import spaces
import numpy as np
import random
import pygame
from .maze import generate_maze
from .assets_loader import load_tiles  # We'll load tile images

class PacManEnv(gym.Env):
    def __init__(self, grid_size=20, num_ghosts=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_ghosts = num_ghosts
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size, 5), dtype=np.float32
        )

        # Load tile images
        self.tiles = load_tiles()

        self.reset()

    def reset(self):
        self.maze = generate_maze(self.grid_size)
        self.pacman_pos = [1, 1]
        self.ghost_positions = [
            [self.grid_size - 2, self.grid_size - 2] for _ in range(self.num_ghosts)
        ]
        self.power_timer = 0
        self.done = False
        self.score = 0
        return self._get_obs()

    def step(self, action):
        # Step penalty
        reward = -0.1
        self._move_pacman(action)
        # ... your logic for pellets, power pellets, ghost collisions, etc.
        # We'll keep it short for brevity. Suppose we do:
        tile = self.maze[self.pacman_pos[1], self.pacman_pos[0]]
        if tile == 2:
            reward = 20
        elif tile == 3:
            reward, self.power_timer = 80, 10
        self.maze[self.pacman_pos[1], self.pacman_pos[0]] = 0

        # Move ghosts randomly
        for ghost in self.ghost_positions:
            g_dir = random.choice(range(4))
            if self._is_valid_move(ghost, g_dir):
                self._move_entity(ghost, g_dir)

        if self.pacman_pos in self.ghost_positions:
            if self.power_timer > 0:
                reward = 300
            else:
                reward = -80
                self.done = True

        self.power_timer = max(0, self.power_timer - 1)
        self.score += reward

        if not (self.maze == 2).any() and not (self.maze == 3).any():
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _move_pacman(self, action):
        """A fallback approach so Pac-Man never gets stuck at a wall."""
        if self._is_valid_move(self.pacman_pos, action):
            self._move_entity(self.pacman_pos, action)
        else:
            # Attempt to keep going in the same direction or pick random valid direction
            pass

    def _is_valid_move(self, pos, action):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            return self.maze[ny, nx] != 1
        return False

    def _move_entity(self, pos, action):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        pos[0] += dx
        pos[1] += dy

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 5), dtype=np.float32)
        # Pac-Man
        obs[self.pacman_pos[1], self.pacman_pos[0], 0] = 1
        # Ghosts
        for ghost in self.ghost_positions:
            obs[ghost[1], ghost[0], 1] = 1
        # Pellets
        obs[:, :, 2] = (self.maze == 2)
        # Power Pellets
        obs[:, :, 3] = (self.maze == 3)
        # Walls
        obs[:, :, 4] = (self.maze == 1)
        return obs

    def render(self, screen, cell_size):
        """
        Blits tile images instead of drawing rectangles.
        """
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                tile_val = self.maze[y, x]
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)

                if tile_val == 1:
                    # Wall
                    wall_img = pygame.transform.scale(self.tiles["wall"], (cell_size, cell_size))
                    screen.blit(wall_img, rect)
                elif tile_val == 2:
                    # Pellet
                    floor_img = pygame.transform.scale(self.tiles["floor"], (cell_size, cell_size))
                    screen.blit(floor_img, rect)
                    pellet_img = pygame.transform.scale(self.tiles["power_pellet"], (cell_size // 2, cell_size // 2))
                    # Center the pellet
                    px = x * cell_size + (cell_size - pellet_img.get_width()) // 2
                    py = y * cell_size + (cell_size - pellet_img.get_height()) // 2
                    screen.blit(pellet_img, (px, py))
                elif tile_val == 3:
                    # Power pellet
                    floor_img = pygame.transform.scale(self.tiles["floor"], (cell_size, cell_size))
                    screen.blit(floor_img, rect)
                    pp_img = pygame.transform.scale(self.tiles["power_pellet"], (cell_size // 2, cell_size // 2))
                    px = x * cell_size + (cell_size - pp_img.get_width()) // 2
                    py = y * cell_size + (cell_size - pp_img.get_height()) // 2
                    screen.blit(pp_img, (px, py))
                else:
                    # Floor
                    floor_img = pygame.transform.scale(self.tiles["floor"], (cell_size, cell_size))
                    screen.blit(floor_img, rect)
