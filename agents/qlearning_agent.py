# agents/qlearning_agent.py
import random
import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = [self.get_q(state, a) for a in range(self.n_actions)]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        max_next_q = max([self.get_q(next_state, a) for a in range(self.n_actions)]) if not done else 0.0
        current_q = self.get_q(state, action)
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
