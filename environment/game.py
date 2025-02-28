# environment/game.py

import pygame
import os
import sys
import random

# Import your constants and BFS function:
from .constants import *
from .bfs import bfs

class PacmanGame:
    def __init__(self):
        pygame.init()

        # Set up the display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pac-Man RL - Diversified Ghosts")
        self.clock = pygame.time.Clock()

        # Load images (ensure these files exist in assets/)
        self.pacman_img = pygame.image.load(os.path.join("assets", "pacman.png"))
        self.pacman_img = pygame.transform.scale(self.pacman_img, (TILE_SIZE, TILE_SIZE))

        self.pacman_power_img = pygame.image.load(os.path.join("assets", "pacman_power.png"))
        self.pacman_power_img = pygame.transform.scale(self.pacman_power_img, (TILE_SIZE, TILE_SIZE))

        self.ghost_sprites = {
            "blinky": pygame.transform.scale(
                pygame.image.load(os.path.join("assets", "blinky.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "pinky": pygame.transform.scale(
                pygame.image.load(os.path.join("assets", "pinky.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "inky": pygame.transform.scale(
                pygame.image.load(os.path.join("assets", "inky.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "clyde": pygame.transform.scale(
                pygame.image.load(os.path.join("assets", "clyde.png")), (TILE_SIZE, TILE_SIZE)
            ),
            "vulnerable": pygame.transform.scale(
                pygame.image.load(os.path.join("assets", "blue_ghost.png")), (TILE_SIZE, TILE_SIZE)
            ),
        }

        # Load sounds (optional)
        pygame.mixer.init()
        try:
            self.sound_pellet = pygame.mixer.Sound(os.path.join("assets", "pellet.wav"))
            self.sound_power_pellet = pygame.mixer.Sound(os.path.join("assets", "power_pellet.wav"))
            self.sound_eat_ghost = pygame.mixer.Sound(os.path.join("assets", "eat_ghost.wav"))
            self.sound_death = pygame.mixer.Sound(os.path.join("assets", "death.wav"))
        except Exception as e:
            print("Warning: Could not load some sound files.", e)

        # Maze layout: 23 rows, each exactly 21 characters
        self.layout = [
            "111111111111111111111",  # Row 0
            "1...................1",  # Row 1
            "1.11.111111.111111.11",  # Row 2
            "1.3.....1......1....1",  # Row 3
            "1.11.1.1.111111.1.111",  # Row 4
            "1....1.......1...1...",  # Row 5
            "1111.1111.1111.1111.1",  # Row 6
            "1..........1........1",  # Row 7
            "1.1111.111111.1111.11",  # Row 8
            "1.1...............1.1",  # Row 9
            "1.1.1111.111111.1.1.1",  # Row 10
            "1.1.1...........1.1.1",  # Row 11
            "1.1.1.111111111.1.1.1",  # Row 12
            "1.3...............3.1",  # Row 13
            "1.1111.111111.1111.11",  # Row 14
            "1........1..........1",  # Row 15
            "1111111.1.11111.11111",  # Row 16
            "1..........1........1",  # Row 17
            "1.11.111111.111111.11",  # Row 18
            "1....3...1......1...1",  # Row 19
            "1.11.1.1.11111.1.11.1",  # Row 20
            "1...................1",  # Row 21
            "111111111111111111111",  # Row 22
        ]

        # Debug: Print row lengths for sanity check
        print("\n[DEBUG] Checking row lengths in self.layout:")
        for i, row in enumerate(self.layout):
            print(f"  Row {i} length={len(row)} => '{row}'")

        self.rows = len(self.layout)
        self.cols = len(self.layout[0])

        # Build numeric grid: 1 = wall, 2 = pellet, 3 = power pellet, 0 = empty
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

        # Initialize power-state attributes
        self.powered = False
        self.power_timer = 0

        # Finally, reset for first episode
        self.reset()

    def reset(self):
        self.done = False
        self.score = 0
        self.current_grid = [row[:] for row in self.grid]
        self.pellets_left = self.total_pellets

        # Pac-Man starting position
        self.pacman_row = 1
        self.pacman_col = 1

        # Reset power state
        self.powered = False
        self.power_timer = 0

        # Initialize ghosts with diversified behavior
        self.ghosts = [
            {"name": "blinky", "row": 11, "col": 10, "behavior": "chase",   "cooldown": 0},
            {"name": "pinky",  "row": 11, "col": 9,  "behavior": "ambush",  "cooldown": 1},
            {"name": "inky",   "row": 12, "col": 10, "behavior": "random",  "cooldown": 2},
            {"name": "clyde",  "row": 12, "col": 9,  "behavior": "scatter", "cooldown": 2},
        ]

        return self.get_state()

    def step(self, action):
        """
        action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        dr, dc = 0, 0
        if action == 0:   # UP
            dr = -1
        elif action == 1: # DOWN
            dr = 1
        elif action == 2: # LEFT
            dc = -1
        elif action == 3: # RIGHT
            dc = 1

        reward = REWARD_STEP

        # Attempt to move Pac-Man if not a wall
        nr = self.pacman_row + dr
        nc = self.pacman_col + dc
        if not self.is_wall(nr, nc):
            self.pacman_row, self.pacman_col = nr, nc

        # Check what is in the cell
        cell_val = self.current_grid[self.pacman_row][self.pacman_col]
        if cell_val == 2:  # Normal pellet
            self.current_grid[self.pacman_row][self.pacman_col] = 0
            self.pellets_left -= 1
            reward += REWARD_PELLET
            self.score += REWARD_PELLET
            self.play_sound(self.sound_pellet)

        elif cell_val == 3:  # Power pellet
            self.current_grid[self.pacman_row][self.pacman_col] = 0
            self.pellets_left -= 1
            reward += REWARD_POWER_PELLET
            self.score += REWARD_POWER_PELLET
            self.powered = True
            self.power_timer = POWER_DURATION
            self.play_sound(self.sound_power_pellet)

        # Check if level is won (no pellets left)
        if self.pellets_left <= 0:
            reward += REWARD_WIN
            self.score += REWARD_WIN
            self.done = True

        # Move ghosts according to their behavior
        for ghost in self.ghosts:
            self.move_ghost(ghost)

        # Collision check with ghosts
        for ghost in self.ghosts:
            if ghost["row"] == self.pacman_row and ghost["col"] == self.pacman_col:
                if self.powered:
                    # Pac-Man eats ghost
                    reward += REWARD_GHOST_EATEN
                    self.score += REWARD_GHOST_EATEN
                    self.play_sound(self.sound_eat_ghost)
                    # Respawn ghost
                    ghost["row"], ghost["col"] = 11, 10
                else:
                    # Pac-Man caught by ghost -> Episode ends
                    reward += REWARD_GHOST_COLLISION
                    self.score += REWARD_GHOST_COLLISION
                    self.done = True
                    self.play_sound(self.sound_death)
                    print("[DEBUG] Pac-Man caught by ghost => done = True")

        # Decrement power timer if powered
        if self.powered:
            self.power_timer -= 1
            if self.power_timer <= 0:
                self.powered = False

        return self.get_state(), reward, self.done, {}

    def move_ghost(self, ghost):
        """
        Move a single ghost according to its behavior (chase, ambush, random, scatter).
        BFS logic is used for 'chase' or fallback.
        """
        # Ghost moves only if cooldown is 0. Otherwise, decrement cooldown first.
        if ghost["cooldown"] > 0:
            ghost["cooldown"] -= 1
            return

        behavior = ghost.get("behavior", "chase")
        dr, dc = 0, 0

        if behavior == "chase":
            # BFS path to Pac-Man
            dr, dc = bfs(
                (ghost["row"], ghost["col"]),
                (self.pacman_row, self.pacman_col),
                self.get_bfs_grid(), self.rows, self.cols
            )
            ghost["cooldown"] = 0

        elif behavior == "ambush":
            # Attempt to guess Pac-Man's future position
            target_r = self.pacman_row + random.choice([-1, 0, 1])
            target_c = self.pacman_col + random.choice([-1, 0, 1])
            dr, dc = bfs(
                (ghost["row"], ghost["col"]),
                (target_r, target_c),
                self.get_bfs_grid(), self.rows, self.cols
            )
            ghost["cooldown"] = 1

        elif behavior == "random":
            # Move in a random valid direction
            possible_moves = []
            for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_r = ghost["row"] + move[0]
                new_c = ghost["col"] + move[1]
                if not self.is_wall(new_r, new_c):
                    possible_moves.append(move)
            if possible_moves:
                dr, dc = random.choice(possible_moves)
            ghost["cooldown"] = 2

        elif behavior == "scatter":
            # Move away from Pac-Man
            possible_moves = []
            for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_r = ghost["row"] + move[0]
                new_c = ghost["col"] + move[1]
                if not self.is_wall(new_r, new_c):
                    dist = (new_r - self.pacman_row) ** 2 + (new_c - self.pacman_col) ** 2
                    possible_moves.append((move, dist))
            if possible_moves:
                best_move = max(possible_moves, key=lambda x: x[1])[0]
                dr, dc = best_move
            ghost["cooldown"] = 2

        else:
            # Default fallback = chase
            dr, dc = bfs(
                (ghost["row"], ghost["col"]),
                (self.pacman_row, self.pacman_col),
                self.get_bfs_grid(), self.rows, self.cols
            )
            ghost["cooldown"] = 0

        ghost["row"] += dr
        ghost["col"] += dc

    def get_bfs_grid(self):
        """
        Return a grid suitable for BFS, marking walls as 1, free paths as 0.
        This is used by the BFS function to find paths around walls.
        """
        bfs_grid = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                # 1 if it's a wall, else 0
                row_data.append(1 if self.current_grid[r][c] == 1 else 0)
            bfs_grid.append(row_data)
        return bfs_grid

    def is_wall(self, r, c):
        """Check if position (r, c) is a wall or out of bounds."""
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True
        return self.current_grid[r][c] == 1

    def get_state(self):
        """
        Construct the state vector that your DQN sees:
        - Pac-Man (row, col)
        - Each ghost (row, col)
        - Whether Pac-Man is powered
        - Pellets left
        """
        state = [self.pacman_row, self.pacman_col]
        for ghost in self.ghosts:
            state.append(ghost["row"])
            state.append(ghost["col"])
        state.append(1 if self.powered else 0)
        state.append(self.pellets_left)
        return tuple(state)

    def render(self):
        """Draw the walls, pellets, Pac-Man, ghosts, and score on-screen."""
        self.screen.fill(BLACK)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.current_grid[r][c]
                x = c * TILE_SIZE
                y = r * TILE_SIZE

                if cell == 1:
                    pygame.draw.rect(self.screen, BLUE, (x, y, TILE_SIZE, TILE_SIZE))
                elif cell == 2:
                    pygame.draw.circle(self.screen, WHITE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 4)
                elif cell == 3:
                    pygame.draw.circle(self.screen, WHITE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 8)

        px = self.pacman_col * TILE_SIZE
        py = self.pacman_row * TILE_SIZE
        if self.powered:
            self.screen.blit(self.pacman_power_img, (px, py))
        else:
            self.screen.blit(self.pacman_img, (px, py))

        # Draw each ghost
        for ghost in self.ghosts:
            gx = ghost["col"] * TILE_SIZE
            gy = ghost["row"] * TILE_SIZE
            if self.powered:
                self.screen.blit(self.ghost_sprites["vulnerable"], (gx, gy))
            else:
                self.screen.blit(self.ghost_sprites[ghost["name"]], (gx, gy))

        # Score text
        font = pygame.font.SysFont(None, 30)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(text, (20, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        self.clock.tick(FPS)

    def play_sound(self, sound):
        """Play a sound effect if available."""
        if sound:
            sound.play()

    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()
        sys.exit()
