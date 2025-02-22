import numpy as np
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, actions, epsilon=0.1, gamma=0.9):
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.Q[state])

    def update(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            self.Q[state][action] += 0.01 * (G - self.Q[state][action])