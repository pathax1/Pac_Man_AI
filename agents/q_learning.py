import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QLearning(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.1, gamma=0.99, epsilon=0.1):
        super(QLearning, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

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

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(device)
        next_state_tensor = torch.FloatTensor(next_state).to(device)
        current_q = self.forward(state_tensor)[action]
        max_next_q = self.forward(next_state_tensor).max().detach()
        target = reward + (self.gamma * max_next_q * (1 - done))
        loss = F.mse_loss(current_q, target)
        self.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.parameters():
                param.data -= self.alpha * param.grad
