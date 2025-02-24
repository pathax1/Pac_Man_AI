import numpy as np
import pygame
import random
import torch
from config.settings import (
    MAZE_LAYOUT, MAZE_WIDTH, MAZE_HEIGHT, TILE_SIZE,
    WALL, PELLET, POWER,
    PELLET_REWARD, POWER_REWARD
)
from game.display import Display
from models.dqn import DQN

class GameController:
    def __init__(self):
        self.maze = np.array(MAZE_LAYOUT, dtype=int)
        self.width_px = MAZE_WIDTH * TILE_SIZE
        self.height_px = MAZE_HEIGHT * TILE_SIZE

        # Pac-Man near bottom center
        self.pacman_pos = [23, 13]
        # 4 ghosts in the ghost house
        self.ghosts = [[14,13],[14,14],[15,13],[15,14]]
        self.running = True
        self.score = 0

        # Load DQN if available
        self.model = DQN(MAZE_WIDTH*MAZE_HEIGHT, 4)
        try:
            self.model.load_state_dict(torch.load("dqn_model.pth", map_location="cpu"))
            self.model.eval()
            print("Loaded DQN model from dqn_model.pth. Pac-Man is AI-controlled.")
            self.ai_control = True
        except FileNotFoundError:
            print("No dqn_model.pth found. Pac-Man moves randomly.")
            self.ai_control = False

        self.display = Display(self)

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self.display.draw()
        self.display.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _update(self):
        # Decide action
        if self.ai_control:
            action = self._choose_action_dqn()
        else:
            action = random.randint(0,3)

        self._move_pacman(action)
        self._move_ghosts()

        # Check ghost collision
        if any(tuple(self.pacman_pos) == tuple(g) for g in self.ghosts):
            print("Pac-Man was caught by a ghost!")
            self.running = False

        # If no pellets remain, end game
        if not (2 in self.maze or 3 in self.maze):
            print("All pellets eaten! Level cleared.")
            self.running = False

    def _choose_action_dqn(self):
        state = torch.FloatTensor(self.maze.flatten()).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
            action = q_values.argmax().item()
        return action

    def _move_pacman(self, action):
        # 0=up,1=down,2=left,3=right
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        nr = self.pacman_pos[0] + moves[action][0]
        nc = self.pacman_pos[1] + moves[action][1]
        if self.maze[nr,nc] != WALL:
            # If pellet or power pellet
            if self.maze[nr,nc] == PELLET:
                self.score += PELLET_REWARD
            elif self.maze[nr,nc] == POWER:
                self.score += POWER_REWARD

            self.pacman_pos = [nr,nc]
            self.maze[nr,nc] = 0  # remove the pellet/power

    def _move_ghosts(self):
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        for i in range(len(self.ghosts)):
            r,c = self.ghosts[i]
            dr,dc = random.choice(moves)
            nr, nc = r+dr, c+dc
            if self.maze[nr,nc] != WALL:
                self.ghosts[i] = [nr,nc]
