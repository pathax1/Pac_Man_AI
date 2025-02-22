import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma * best_next - self.Q[state][action])