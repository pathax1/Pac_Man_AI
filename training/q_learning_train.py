# training/q_learning_train.py

import numpy as np
from environment.game import PacmanGame
from agents.q_learning_agent import QLearningAgent

def train_q_learning(episodes=500):
    env = PacmanGame()
    state_size = len(env.get_state())
    action_size = 4
    agent = QLearningAgent(state_size=state_size, action_size=action_size)

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # env.render()  # optional
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f"[Q-Learning] Episode {e+1}/{episodes}, Reward={total_reward:.2f}, Epsilon={agent.epsilon:.2f}")

    env.close()
