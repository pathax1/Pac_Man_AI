import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MonteCarlo(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.01, gamma=0.99, epsilon=0.1):
        super(MonteCarlo, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = {}

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.fc3.out_features)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            return self.forward(state_tensor).argmax().item()

    def update(self, episode_memory):
        G = 0
        total_loss = 0.0
        for state, action, reward in reversed(episode_memory):
            G = self.gamma * G + reward
            state_key = tuple(state)
            if (state_key, action) not in self.returns:
                self.returns[(state_key, action)] = []
            self.returns[(state_key, action)].append(G)
            target = torch.tensor(G, dtype=torch.float32).to(device)
            prediction = self.forward(torch.FloatTensor(state).to(device))[action]
            loss = F.mse_loss(prediction, target)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in self.parameters():
                    param.data -= self.alpha * param.grad
            total_loss += loss.item()
        return total_loss / len(episode_memory)
