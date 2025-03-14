import numpy as np
from collections import defaultdict
from config import MC_GAMMA

class MonteCarloAgent:
    def __init__(self, action_dim, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        self.action_dim = action_dim
        self.Q = defaultdict(float)
        self.returns = defaultdict(list)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = MC_GAMMA

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_vals = [self.Q[(state, a)] for a in range(self.action_dim)]
        return int(np.argmax(q_vals))

    def store_episode(self, episode):
        G = 0
        visited_pairs = set()
        for s, a, r in reversed(episode):
            G = self.gamma * G + r
            if (s, a) not in visited_pairs:
                self.returns[(s, a)].append(G)
                self.Q[(s, a)] = np.mean(self.returns[(s, a)])
                visited_pairs.add((s, a))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
