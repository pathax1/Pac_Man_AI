import torch
import numpy as np
from models.monte_carlo import MonteCarlo
from environment.pacman_env import PacmanEnv

EPISODES = 500
GAMMA = 0.99


def train():
    env = PacmanEnv()
    input_dim = np.prod(env.observation_space.shape)
    output_dim = env.action_space.n

    policy = MonteCarlo(input_dim, output_dim)
    returns = {}  # Store returns for state-action pairs

    for episode in range(EPISODES):
        state = env.reset().flatten()
        episode_memory = []
        done = False

        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_memory.append((state, action, reward))
            state = next_state.flatten()

        # Calculate returns and update policy
        G = 0
        for state, action, reward in reversed(episode_memory):
            G = GAMMA * G + reward
            if (tuple(state), action) not in returns:
                returns[(tuple(state), action)] = []
            returns[(tuple(state), action)].append(G)
            policy.update(state, action, np.mean(returns[(tuple(state), action)]))

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{EPISODES} completed.")

    torch.save(policy.state_dict(), "models/monte_carlo_model.pth")
    print("✅ Monte Carlo model training completed and saved.")


if __name__ == "__main__":
    train()