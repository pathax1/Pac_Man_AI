import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.q_learning import QLearning
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

    agent = QLearning(input_dim, output_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting Q-Learning training...")

    for episode in tqdm(range(EPISODES), desc="🤖 Training Q-Learning", unit="episode"):
        state = env.reset().flatten()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            if np.random.rand() < agent.epsilon:
                action = env.action_space.sample()
            else:
                action = agent(torch.FloatTensor(state)).argmax().item()

            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state.flatten(), done)
            state = next_state.flatten()
            total_reward += reward
            steps += 1

        tqdm.write(f"✅ Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | Steps: {steps}")

        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"📢 Progress: {episode + 1}/{EPISODES} episodes completed.")

    torch.save(agent.state_dict(), r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\models\q_learning_model.pth")
    print("🏆 Q-Learning training completed and model saved.")

if __name__ == "__main__":
    train()
