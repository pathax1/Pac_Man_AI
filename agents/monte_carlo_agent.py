# agents/monte_carlo_agent.py
import random
import numpy as np
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, n_actions, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.Q = defaultdict(float)
        self.episode = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        return int(np.argmax(q_values))

    def store_transition(self, state, action, reward):
        self.episode.append((state, action, reward))

    def update(self):
        G = 0
        for state, action, reward in reversed(self.episode):
            G = reward + G
            key = (state, action)
            self.returns_sum[key] += G
            self.returns_count[key] += 1
            self.Q[key] = self.returns_sum[key] / self.returns_count[key]
        self.episode = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
