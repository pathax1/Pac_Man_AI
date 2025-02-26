import numpy as np
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, action_size, gamma=0.99, epsilon=0.1):
        self.q = defaultdict(lambda: np.zeros(action_size))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q[state]))
        return np.argmax(self.q[state])

    def update(self, episode):
        """Update Q-values using Monte Carlo returns."""
        G = 0
        visited_state_actions = set()

        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited_state_actions:
                self.returns_sum[(state, action)] += G
                self.returns_count[(state, action)] += 1
                self.q[state][action] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]
                visited_state_actions.add((state, action))
