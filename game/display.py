import pygame

class Display:
    TILE_SIZE = 32

    def __init__(self, grid_size):
        self.width, self.height = grid_size[1] * self.TILE_SIZE, grid_size[0] * self.TILE_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pac-Man RL")
        self.colors = {
            0: (0, 0, 0),       # Empty
            1: (0, 0, 255),     # Wall
            2: (255, 255, 0),   # Pellet
            3: (255, 0, 0)      # Power Pellet
        }

    def render(self, game_map, pacman_pos, ghost_positions):
        self.screen.fill((0, 0, 0))
        for x, row in enumerate(game_map):
            for y, cell in enumerate(row):
                color = self.colors.get(cell, (0, 0, 0))
                pygame.draw.rect(self.screen, color, (y * self.TILE_SIZE, x * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw Pac-Man
        pygame.draw.circle(self.screen, (255, 255, 0), ((pacman_pos[1] * self.TILE_SIZE) + 16, (pacman_pos[0] * self.TILE_SIZE) + 16), 12)

        # Draw Ghosts
        for gx, gy in ghost_positions:
            pygame.draw.circle(self.screen, (255, 0, 0), ((gy * self.TILE_SIZE) + 16, (gx * self.TILE_SIZE) + 16), 12)

        pygame.display.update()