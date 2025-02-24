import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

from models.dqn import DQN
from train.replay_buffer import ReplayBuffer
from environment.pacman_env import PacmanEnv

EPISODES = 2000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0005
BUFFER_SIZE = 50000
TARGET_UPDATE = 100
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100000
MAX_STEPS_PER_EPISODE = 500

def train():
    env = PacmanEnv()
    input_dim = np.prod(env.observation_space.shape)
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    steps_done = 0
    print("🚀 Starting DQN Training...")

    for episode in tqdm(range(EPISODES), desc="Training"):
        state = env.reset().flatten()
        total_reward = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            # Epsilon schedule
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
            steps_done += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.FloatTensor(np.array(states))
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(np.array(next_states))
                dones_t = torch.FloatTensor(dones)

                q_values = policy_net(states_t).gather(1, actions_t).squeeze()
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1)[0]
                targets = rewards_t + GAMMA * next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    torch.save(policy_net.state_dict(), "dqn_model.pth")
    print("🏆 Training complete! Model saved to dqn_model.pth.")

if __name__ == "__main__":
    train()
