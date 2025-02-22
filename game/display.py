import pygame
import os
from config.settings import (
    TILE_SIZE, MAZE_WIDTH, MAZE_HEIGHT,
    WALL, PELLET, POWER,
    TILE_ASSETS, PACMAN_IMAGE, GHOST_IMAGES
)

class Display:
    def __init__(self, game_controller):
        self.gc = game_controller
        # Initialize Pygame display
        self.screen = pygame.display.set_mode((self.gc.width_px, self.gc.height_px))
        pygame.display.set_caption("Pac-Man AI")
        self.clock = pygame.time.Clock()

        # Load tile images
        self.wall_img = pygame.image.load(TILE_ASSETS["wall"]).convert_alpha()
        self.wall_img = pygame.transform.scale(self.wall_img, (TILE_SIZE, TILE_SIZE))

        self.pellet_img = pygame.image.load(TILE_ASSETS["pellet"]).convert_alpha()
        self.pellet_img = pygame.transform.scale(self.pellet_img, (TILE_SIZE, TILE_SIZE))

        self.power_img = pygame.image.load(TILE_ASSETS["power"]).convert_alpha()
        self.power_img = pygame.transform.scale(self.power_img, (TILE_SIZE, TILE_SIZE))

        self.floor_img = pygame.image.load(TILE_ASSETS["floor"]).convert_alpha()
        self.floor_img = pygame.transform.scale(self.floor_img, (TILE_SIZE, TILE_SIZE))

        # Load Pac-Man sprite
        self.pacman_img = pygame.image.load(PACMAN_IMAGE).convert_alpha()
        self.pacman_img = pygame.transform.scale(self.pacman_img, (TILE_SIZE, TILE_SIZE))

        # Load Ghost sprites
        self.ghost_imgs = []
        for path in GHOST_IMAGES:
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
            self.ghost_imgs.append(img)

    def draw(self):
        # Draw the maze
        for r in range(MAZE_HEIGHT):
            for c in range(MAZE_WIDTH):
                tile = self.gc.maze[r, c]
                x = c * TILE_SIZE
                y = r * TILE_SIZE

                # Draw floor first
                self.screen.blit(self.floor_img, (x, y))

                # Overlay walls/pellets/power
                if tile == WALL:
                    self.screen.blit(self.wall_img, (x, y))
                elif tile == PELLET:
                    self.screen.blit(self.pellet_img, (x, y))
                elif tile == POWER:
                    self.screen.blit(self.power_img, (x, y))
                # else it's floor, already drawn

        # Draw Pac-Man
        px = self.gc.pacman_pos[1] * TILE_SIZE
        py = self.gc.pacman_pos[0] * TILE_SIZE
        self.screen.blit(self.pacman_img, (px, py))

        # Draw Ghosts
        for i, gpos in enumerate(self.gc.ghosts):
            gx = gpos[1] * TILE_SIZE
            gy = gpos[0] * TILE_SIZE
            ghost_img = self.ghost_imgs[i % len(self.ghost_imgs)]
            self.screen.blit(ghost_img, (gx, gy))

        pygame.display.flip()
        self.clock.tick(60)

    def quit(self):
        pygame.quit()
