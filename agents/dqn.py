import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# Detect device and enable AMP only for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler(device.type) if use_amp else None
print(f"Training on device: {device} | AMP enabled: {use_amp}")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_dim, action_size, lr=1e-3, gamma=0.99, batch_size=128):
        self.model = DQN(state_dim, action_size).to(device)
        self.target = DQN(state_dim, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def store(self, transition):
        self.memory.append(transition)

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 3)
        state = torch.FloatTensor(np.array(state).flatten()).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample and convert data efficiently
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, self.batch_size))
        states = torch.FloatTensor(np.array(states)).reshape(self.batch_size, -1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).reshape(self.batch_size, -1).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Use AMP if CUDA is available
        if use_amp:
            with torch.amp.autocast(device_type='cuda'):
                q_values = self.model(states).gather(1, actions)
                next_q = self.target(next_states).max(1)[0].unsqueeze(1).detach()
                expected = rewards + self.gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(q_values, expected)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

        else:
            q_values = self.model(states).gather(1, actions)
            next_q = self.target(next_states).max(1)[0].unsqueeze(1).detach()
            expected = rewards + self.gamma * next_q * (1 - dones)
            loss = nn.MSELoss()(q_values, expected)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
