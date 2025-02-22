import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.dqn import DQN
from models.replay_buffer import ReplayBuffer
from environment.pacman_env import PacmanEnv
from tqdm import tqdm

# Hyperparameters
EPISODES = 5000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
BUFFER_SIZE = 10000
TARGET_UPDATE = 10
LOG_INTERVAL = 10  # Log progress every 10 episodes
MAX_STEPS = 1000   # Safety: maximum steps per episode

def train():
    env = PacmanEnv()
    input_dim = np.prod(env.observation_space.shape)
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    target_net = DQN(input_dim, output_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)

    print("🚀 Starting DQN training...")

    for episode in tqdm(range(EPISODES), desc="🤖 Training DQN", unit="episode"):
        state = env.reset().flatten()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            # Epsilon-greedy action selection
            if np.random.rand() > 0.1:
                action = policy_net(torch.FloatTensor(state)).argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state.flatten(), done)
            state = next_state.flatten()
            total_reward += reward
            steps += 1

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(np.array(dones))

                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if steps >= MAX_STEPS:
            tqdm.write(f"⚠️ Episode {episode+1}: Reached maximum steps ({MAX_STEPS}) without termination.")

        tqdm.write(f"✅ Episode {episode + 1}/{EPISODES} | Reward: {total_reward:.2f} | Steps: {steps}")

        if (episode + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"📢 Progress: {episode + 1}/{EPISODES} episodes completed.")

    torch.save(policy_net.state_dict(), r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\models\dqn_model.pth")
    print("🏆 DQN model training completed and saved.")

if __name__ == "__main__":
    train()
