import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.dqn import DQN
from models.replay_buffer import ReplayBuffer
from environment.pacman_env import PacmanEnv

# Hyperparameters
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
BUFFER_SIZE = 10000
TARGET_UPDATE = 10


def train():
    """
    Train a Deep Q-Network (DQN) to play Pac-Man using reinforcement learning.
    """
    env = PacmanEnv()
    input_dim = np.prod(env.observation_space.shape)
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)

    for episode in range(EPISODES):
        state = env.reset().flatten()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() > 0.1:
                action = policy_net(torch.FloatTensor(state)).argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state.flatten(), done)

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # ✅ Optimized conversion: Convert lists to NumPy arrays first
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(np.array(dones))

                # Calculate current Q-values and target Q-values
                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + GAMMA * next_q_values * (1 - dones)

                # Compute loss and update network
                loss = nn.functional.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state.flatten()
            total_reward += reward

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"✅ Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward}")

    # Save the trained model
    torch.save(policy_net.state_dict(), "models/dqn_model.pth")
    print("🏆 DQN model training completed and saved.")


if __name__ == "__main__":
    train()