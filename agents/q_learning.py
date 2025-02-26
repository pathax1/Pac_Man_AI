import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_space = state_space
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Create a Q-table with zeros
        self.q_table = np.zeros((*state_space, action_size))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values."""
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state][action] += self.alpha * (target - predict)

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
