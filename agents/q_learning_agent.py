# agents/q_learning_agent.py

import numpy as np
import random


class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Dictionary-based Q-table
        self.q_table = {}

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        qs = self.get_qs(state)
        return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        qs = self.get_qs(state)
        next_qs = self.get_qs(next_state)

        best_next_action = np.argmax(next_qs)
        qs[action] += self.alpha * (
                reward + self.gamma * next_qs[best_next_action] - qs[action]
        )

        self.q_table[state] = qs

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
