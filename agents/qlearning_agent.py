import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import DEVICE, QL_ALPHA, GAMMA, QL_EPSILON, QL_EPSILON_MIN, QL_EPSILON_DECAY

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

class QLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=QL_ALPHA)
        self.gamma = GAMMA
        self.epsilon = QL_EPSILON
        self.epsilon_min = QL_EPSILON_MIN
        self.epsilon_decay = QL_EPSILON_DECAY
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).to(DEVICE)
        next_state_t = torch.FloatTensor(next_state).to(DEVICE)
        reward_t = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
        done_t = torch.tensor(done, dtype=torch.float32).to(DEVICE)

        q_value = self.q_net(state_t)[action]
        next_q_value = self.q_net(next_state_t).max()

        expected_q = reward_t + (1 - done_t) * self.gamma * next_q_value
        loss = nn.MSELoss()(q_value, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
