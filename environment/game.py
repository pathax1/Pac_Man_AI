# environment/game.py

import pygame
import sys
import random
import os

from .constants import *
from .bfs import bfs
from .mazes.simple_maze import SIMPLE_LAYOUT
from .mazes.medium_maze import MEDIUM_LAYOUT
from .mazes.complex_maze import COMPLEX_LAYOUT

class PacmanGame:
    def __init__(self, level="simple"):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pac-Man Reinforcement")

        self.clock = pygame.time.Clock()

        # Choose maze layout
        if level == "simple":
            self.layout = SIMPLE_LAYOUT
        elif level == "medium":
            self.layout = MEDIUM_LAYOUT
        else:
            self.layout = COMPLEX_LAYOUT

        self.rows = len(self.layout)
        self.cols = len(self.layout[0])

        # Load assets
        asset_path = os.path.join(os.path.dirname(__file__), '..', 'assets')
        self.pacman_img = pygame.transform.scale(
            pygame.image.load(os.path.join(asset_path, "pacman.png")), (TILE_SIZE, TILE_SIZE)
        )
        self.pacman_power_img = pygame.transform.scale(
            pygame.image.load(os.path.join(asset_path, "pacman_power.png")), (TILE_SIZE, TILE_SIZE)
        )
        self.ghost_sprites = {
            "blinky": pygame.transform.scale(
                pygame.image.load(os.path.join(asset_path, "blinky.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "pinky": pygame.transform.scale(
                pygame.image.load(os.path.join(asset_path, "pinky.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "inky": pygame.transform.scale(
                pygame.image.load(os.path.join(asset_path, "inky.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "clyde": pygame.transform.scale(
                pygame.image.load(os.path.join(asset_path, "clyde.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "vulnerable": pygame.transform.scale(
                pygame.image.load(os.path.join(asset_path, "blue_ghost.png")), (TILE_SIZE, TILE_SIZE)
            ),
        }

        # Load sounds
        try:
            self.sound_pellet = pygame.mixer.Sound(os.path.join(asset_path, "pellet.wav"))
            #self.sound_power_pellet = pygame.mixer.Sound(os.path.join(asset_path, "power_pellet.wav"))
            #self.sound_eat_ghost = pygame.mixer.Sound(os.path.join(asset_path, "eat_ghost.wav"))
            #self.sound_death = pygame.mixer.Sound(os.path.join(asset_path, "death.wav"))
        except:
            self.sound_pellet = None
            self.sound_power_pellet = None
            self.sound_eat_ghost = None
            self.sound_death = None

        # Build grid (1=wall, 2=pellet, 3=power pellet, 0=empty)
        self.grid = []
        self.total_pellets = 0
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                ch = self.layout[r][c]
                if ch == '1':
                    row_data.append(1)
                elif ch == '.':
                    row_data.append(2)
                    self.total_pellets += 1
                elif ch == '3':
                    row_data.append(3)
                    self.total_pellets += 1
                else:
                    row_data.append(0)
            self.grid.append(row_data)

        self.powered = False
        self.power_timer = 0
        self.reset()

    def reset(self):
        self.done = False
        self.score = 0
        self.current_grid = [row[:] for row in self.grid]
        self.pellets_left = self.total_pellets

        self.pacman_row, self.pacman_col = 1, 1  # Starting position
        self.powered = False
        self.power_timer = 0

        # Initialize counters
        self.survival_time = 0
        self.pellets_consumed = 0

        # Initialize ghosts
        self.ghosts = [
            {"name": "blinky", "row": 11, "col": 10, "behavior": "chase", "cooldown": 5},
            {"name": "pinky", "row": 11, "col": 9, "behavior": "ambush", "cooldown": 5},
            {"name": "inky", "row": 12, "col": 10, "behavior": "random", "cooldown": 10},
            {"name": "clyde", "row": 12, "col": 9, "behavior": "scatter", "cooldown": 10},
        ]

        self.done = False
        self.score = 0
        self.current_grid = [row[:] for row in self.grid]
        self.pellets_left = self.total_pellets

        return self.get_state()

    def step(self, action):
        """
        action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        # Slight negative reward each step
        reward = REWARD_STEP

        # Increment survival time each step
        self.survival_time += 1

        dr, dc = 0, 0
        if action == 0:  # UP
            dr = -1
        elif action == 1:  # DOWN
            dr = 1
        elif action == 2:  # LEFT
            dc = -1
        elif action == 3:  # RIGHT
            dc = 1

        nr, nc = self.pacman_row + dr, self.pacman_col + dc
        if not self.is_wall(nr, nc):
            self.pacman_row, self.pacman_col = nr, nc

        cell_val = self.current_grid[self.pacman_row][self.pacman_col]

        if cell_val == 2:  # Normal pellet
            self.current_grid[self.pacman_row][self.pacman_col] = 0
            self.pellets_left -= 1
            self.pellets_consumed += 1  # Track pellets consumed
            reward += REWARD_PELLET
            self.score += REWARD_PELLET
            self.play_sound(self.sound_pellet)

        elif cell_val == 3:  # Power pellet
            self.current_grid[self.pacman_row][self.pacman_col] = 0
            self.pellets_left -= 1
            self.pellets_consumed += 1  # Track power pellets consumed
            reward += REWARD_POWER_PELLET
            self.score += REWARD_POWER_PELLET
            self.powered = True
            self.power_timer = POWER_DURATION
            self.play_sound(self.sound_power_pellet)

        # Check if level is won
        if self.pellets_left <= 0:
            reward += REWARD_WIN
            self.score += REWARD_WIN
            self.done = True

        # Move ghosts
        for ghost in self.ghosts:
            self.move_ghost(ghost)

        # Check collisions
        for ghost in self.ghosts:
            if ghost["row"] == self.pacman_row and ghost["col"] == self.pacman_col:
                if self.powered:
                    reward += REWARD_GHOST_EATEN
                    self.score += REWARD_GHOST_EATEN
                    self.play_sound(self.sound_eat_ghost)
                    ghost["row"], ghost["col"] = 11, 10
                else:
                    reward += REWARD_GHOST_COLLISION
                    self.score += REWARD_GHOST_COLLISION
                    self.done = True
                    self.play_sound(self.sound_death)

        # Move ghosts
        for ghost in self.ghosts:
            self.move_ghost(ghost)

        # Manage power timer
        if self.powered:
            self.power_timer -= 1
            if self.power_timer <= 0:
                self.powered = False

        return self.get_state(), reward, self.done, {}

    def is_wall(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True
        return self.current_grid[r][c] == 1

    def move_ghost(self, ghost):
        if ghost["cooldown"] > 0:
            ghost["cooldown"] -= 1
            return

        behavior = ghost["behavior"]
        if behavior == "chase":
            # BFS direct chase
            dr, dc = bfs(
                (ghost["row"], ghost["col"]),
                (self.pacman_row, self.pacman_col),
                self.get_bfs_grid(),
                self.rows,
                self.cols
            )
            ghost["cooldown"] = 2

        elif behavior == "ambush":
            # Ambush tries a random offset near Pac-Man
            offset_r = random.randint(-2, 2)
            offset_c = random.randint(-2, 2)
            target = (self.pacman_row + offset_r, self.pacman_col + offset_c)
            dr, dc = bfs(
                (ghost["row"], ghost["col"]),
                target,
                self.get_bfs_grid(),
                self.rows,
                self.cols
            )
            ghost["cooldown"] = 3

        elif behavior == "random":
            # Random movement ignoring BFS
            possible_moves = []
            for (dr_, dc_) in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = ghost["row"] + dr_, ghost["col"] + dc_
                if not self.is_wall(nr, nc):
                    possible_moves.append((dr_, dc_))
            if possible_moves:
                dr, dc = random.choice(possible_moves)
            else:
                dr, dc = 0, 0
            ghost["cooldown"] = 4

        else:  # "scatter" or fallback
            # Scatter tries to move away from Pac-Man
            # We'll pick BFS but to a corner or something.
            # For simplicity, just pick the top-left corner as a "scatter" point
            corner = (1, 1)
            dr, dc = bfs(
                (ghost["row"], ghost["col"]),
                corner,
                self.get_bfs_grid(),
                self.rows,
                self.cols
            )
            ghost["cooldown"] = 3

        ghost["row"] += dr
        ghost["col"] += dc

    def get_bfs_grid(self):
        """
        Return a grid that BFS can interpret:
        1 => wall
        0 => free
        """
        grid = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                row_data.append(1 if self.current_grid[r][c] == 1 else 0)
            grid.append(row_data)
        return grid

    def get_state(self):
        """
        Return a tuple describing the environment state:
          (pacman_row, pacman_col,
           ghost1_row, ghost1_col, ghost2_row, ghost2_col, ...,
           powered_flag, pellets_left)
        """
        state = [self.pacman_row, self.pacman_col]
        for ghost in self.ghosts:
            state.extend([ghost["row"], ghost["col"]])
        state.append(1 if self.powered else 0)
        state.append(self.pellets_left)
        return tuple(state)

    def render(self):
        self.screen.fill(BLACK)

        # Draw maze
        for r in range(self.rows):
            for c in range(self.cols):
                cell_val = self.current_grid[r][c]
                x = c * TILE_SIZE
                y = r * TILE_SIZE

                if cell_val == 1:
                    pygame.draw.rect(self.screen, BLUE, (x, y, TILE_SIZE, TILE_SIZE))
                elif cell_val == 2:
                    pygame.draw.circle(self.screen, WHITE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 4)
                elif cell_val == 3:
                    pygame.draw.circle(self.screen, WHITE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 8)

        # Draw Pac-Man
        px = self.pacman_col * TILE_SIZE
        py = self.pacman_row * TILE_SIZE
        if self.powered:
            self.screen.blit(self.pacman_power_img, (px, py))
        else:
            self.screen.blit(self.pacman_img, (px, py))

        # Draw ghosts
        for ghost in self.ghosts:
            gx = ghost["col"] * TILE_SIZE
            gy = ghost["row"] * TILE_SIZE
            if self.powered:
                self.screen.blit(self.ghost_sprites["vulnerable"], (gx, gy))
            else:
                self.screen.blit(self.ghost_sprites[ghost["name"]], (gx, gy))

        font = pygame.font.SysFont(None, 32)

        # Display Score
        score_text = font.render(f"Score: {int(self.score)}", True, WHITE)
        self.screen.blit(score_text, (10, SCREEN_HEIGHT - 40))

        # Display Pellets Consumed
        pellets_text = font.render(f"Pellets: {self.pellets_consumed}", True, WHITE)
        self.screen.blit(pellets_text, (200, SCREEN_HEIGHT - 40))

        # Display Survival Time
        survival_text = font.render(f"Survival: {self.survival_time}", True, WHITE)
        self.screen.blit(survival_text, (400, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        self.clock.tick(FPS)

    def play_sound(self, sound):
        if sound:
            sound.play()

    def close(self):
        pygame.quit()
        #sys.exit()
