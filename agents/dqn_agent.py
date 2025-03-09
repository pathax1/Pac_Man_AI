# agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from config import LR, GAMMA, BATCH_SIZE, MEMORY_SIZE, EPS_START, EPS_END, EPS_DECAY, DEVICE

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_net = DQNNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = DQNNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE

        # Epsilon params
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.global_episodes = 0
        self.epsilon = self.eps_start

    def select_action(self, state):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.global_episodes += 1
        self.epsilon = max(self.eps_end, self.eps_start - (self.global_episodes / self.eps_decay))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
        return states, actions, rewards, next_states, dones

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        # Current Q estimates
        q_values = self.q_net(states).gather(1, actions)

        # Double DQN:
        with torch.no_grad():
            # select action with online network
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            # evaluate action with target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
