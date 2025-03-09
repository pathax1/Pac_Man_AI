# agents/qlearning_agent.py

import numpy as np
from config import QL_ALPHA, QL_EPSILON, QL_EPSILON_MIN, QL_EPSILON_DECAY, GAMMA

class QLearningAgent:
    """
    Simple tabular Q-learning (for smaller state spaces).
    If the state space is too large, consider function approximation instead.
    """
    def __init__(self, state_shape, action_dim):
        # We'll store Q-values in a dictionary: Q[(state), action] = value
        self.Q = {}
        self.alpha = QL_ALPHA
        self.gamma = GAMMA

        self.epsilon = QL_EPSILON
        self.epsilon_min = QL_EPSILON_MIN
        self.epsilon_decay = QL_EPSILON_DECAY
        self.action_dim = action_dim

    def _get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def select_action(self, state):
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            # Pick action with highest Q
            q_vals = [self._get_q(state, a) for a in range(self.action_dim)]
            return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state, done):
        best_next_q = max([self._get_q(next_state, a) for a in range(self.action_dim)])
        current_q = self._get_q(state, action)

        target = reward + (0 if done else (self.gamma * best_next_q))
        new_q = current_q + self.alpha * (target - current_q)
        self.Q[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
