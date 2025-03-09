# agents/monte_carlo_agent.py

import numpy as np
from collections import defaultdict
from config import MC_GAMMA

class MonteCarloAgent:
    """
    On-policy First-Visit Monte Carlo for smaller state spaces.
    """
    def __init__(self, action_dim, epsilon=0.1):
        self.action_dim = action_dim
        self.Q = defaultdict(lambda: 0.0)
        self.returns = defaultdict(list)
        self.epsilon = epsilon
        self.gamma = MC_GAMMA

    def select_action(self, state):
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_vals = [self.Q[(state, a)] for a in range(self.action_dim)]
            return int(np.argmax(q_vals))

    def store_episode(self, episode):
        """
        episode: list of (state, action, reward) for one complete episode
        """
        visited = set()
        G = 0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = self.gamma * G + r
            if (s, a) not in visited:
                self.returns[(s, a)].append(G)
                self.Q[(s, a)] = np.mean(self.returns[(s, a)])
                visited.add((s, a))
