"""
Launch the Pac-Man game directly.
"""
import pygame
from game.game_controller import GameController

def main():
    pygame.init()
    game = GameController()
    game.run()
    pygame.quit()

if __name__ == "__main__":
    main()
