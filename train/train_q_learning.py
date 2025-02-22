import numpy as np
import torch
from models.q_learning import QLearning
from environment.pacman_env import PacmanEnv

# Hyperparameters
EPISODES = 500
ALPHA = 0.1  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON = 0.1  # Exploration rate


def train():
    env = PacmanEnv()
    input_dim = np.prod(env.observation_space.shape)
    output_dim = env.action_space.n

    policy = QLearning(input_dim, output_dim, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

    for episode in range(EPISODES):
        state = env.reset().flatten()
        done = False

        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # ✅ Added the 'done' argument
            policy.update(state, action, reward, next_state.flatten(), done)

            state = next_state.flatten()

        if (episode + 1) % 50 == 0:
            print(f"✅ Episode {episode + 1}/{EPISODES} completed.")

    torch.save(policy.state_dict(), "models/q_learning_model.pth")
    print("🏆 Q-Learning model training completed and saved.")


if __name__ == "__main__":
    train()
