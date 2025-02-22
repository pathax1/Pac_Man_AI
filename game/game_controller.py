import pygame
import torch
import numpy as np
from environment.pacman_env import PacmanEnv
from models.dqn import DQN
from game.display import Display
from game.player import Pacman
from game.ghost import Ghost

class GameController:
    def __init__(self):
        pygame.init()

        # Initialize Pac-Man environment
        self.env = PacmanEnv()

        # Load trained DQN model
        input_dim = np.prod(self.env.observation_space.shape)
        output_dim = self.env.action_space.n
        self.policy_net = DQN(input_dim, output_dim)
        try:
            self.policy_net.load_state_dict(torch.load("models/dqn_model.pth"))
            print("✅ Trained model loaded successfully.")
        except FileNotFoundError:
            print("❌ Trained model not found! Please train the model first using 'train/train_dqn.py'.")
            exit()

        self.policy_net.eval()  # Set to evaluation mode

        # Initialize game display and clock
        self.display = Display(self.env.grid_size)
        self.clock = pygame.time.Clock()

        # Initialize Pac-Man and ghosts
        self.pacman = Pacman(self.env.pacman_position)
        self.ghosts = [Ghost(pos) for pos in self.env.ghost_positions]

    def run(self):
        state = self.env.reset().flatten()  # Reset environment and flatten observation
        running = True

        while running:
            self.clock.tick(10)  # Control FPS

            # Use trained model to select best action
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax().item()

            # Take action in environment
            next_state, reward, done, _ = self.env.step(action)

            # Move ghosts towards Pac-Man
            for ghost in self.ghosts:
                ghost.move_towards(self.env.pacman_position, self.env.map)

            # Render the updated game state
            self.display.render(
                self.env.map,
                self.env.pacman_position,
                [ghost.position for ghost in self.ghosts]
            )

            # Update state for next step
            state = next_state.flatten()

            # Event handling to close the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT or done:
                    running = False

        print(f"🏆 Final Score: {self.env.score}")
        pygame.quit()

if __name__ == "__main__":
    print("🚀 Starting Pac-Man with Trained RL Agent...")
    game = GameController()
    game.run()