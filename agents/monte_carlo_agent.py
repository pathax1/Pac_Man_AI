# agents/monte_carlo_agent.py

import numpy as np
import random
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, state_size, action_size,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        self.episode = []

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        qs = self.q_table[state]
        return np.argmax(qs)

    def start_episode(self):
        self.episode = []

    def record_transition(self, state, action, reward):
        self.episode.append((state, action, reward))

    def end_episode(self):
        G = 0
        visited = set()

        for t in reversed(range(len(self.episode))):
            s, a, r = self.episode[t]
            G = self.gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                self.returns_sum[(s, a)] += G
                self.returns_count[(s, a)] += 1
                self.q_table[s][a] = (
                    self.returns_sum[(s, a)] / self.returns_count[(s, a)]
                )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
