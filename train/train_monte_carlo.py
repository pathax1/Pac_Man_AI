import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.monte_carlo import MonteCarlo
from environment.pacman_env import PacmanEnv
from tqdm import tqdm

# Hyperparameters
EPISODES = 5000
LEARNING_RATE = 0.01
GAMMA = 0.99
LOG_INTERVAL = 10
MAX_STEPS = 1000

def train():
    env = PacmanEnv()
    input_dim = np.prod(env.observation_space.shape)
    output_dim = env.action_space.n

    agent = MonteCarlo(input_dim, output_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting Monte Carlo training...")

    for episode in tqdm(range(EPISODES), desc="🤖 Training Monte Carlo", unit="episode"):
        state = env.reset().flatten()
        episode_memory = []
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_memory.append((state, action, reward))
            state = next_state.flatten()
            total_reward += reward
            steps += 1

        loss = agent.update(episode_memory)
        tqdm.write(f"✅ Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | Loss: {loss:.4f}")

        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"📢 Progress: {episode + 1}/{EPISODES} episodes completed.")

    torch.save(agent.state_dict(), r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\models\monte_carlo_model.pth")
    print("🏆 Monte Carlo training completed and model saved.")

if __name__ == "__main__":
    train()
