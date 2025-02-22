import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MonteCarlo(nn.Module):
    def __init__(self, input_dim, output_dim, gamma=0.99):
        super(MonteCarlo, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.gamma = gamma
        self.returns = {}  # Store returns for state-action pairs

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            return self.forward(state_tensor).argmax().item()

    def update(self, episode_memory):
        G = 0
        for state, action, reward in reversed(episode_memory):
            G = self.gamma * G + reward
            state_key = tuple(state)

            if (state_key, action) not in self.returns:
                self.returns[(state_key, action)] = []

            self.returns[(state_key, action)].append(G)

            # Update weights based on return
            target = torch.tensor(G, dtype=torch.float32)
            prediction = self.forward(torch.FloatTensor(state))[action]
            loss = F.mse_loss(prediction, target)

            self.zero_grad()
            loss.backward()
            for param in self.parameters():
                param.data -= 0.01 * param.grad  # Simple SGD update