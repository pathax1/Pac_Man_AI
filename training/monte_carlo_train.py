# training/monte_carlo_train.py

import numpy as np
from environment.game import PacmanGame
from agents.monte_carlo_agent import MonteCarloAgent

def train_monte_carlo(episodes=500):
    env = PacmanGame()
    state_size = len(env.get_state())
    action_size = 4
    agent = MonteCarloAgent(state_size=state_size, action_size=action_size)

    for e in range(episodes):
        agent.start_episode()
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # env.render()  # optional
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.record_transition(state, action, reward)
            state = next_state
            total_reward += reward

        agent.end_episode()
        print(f"[MonteCarlo] Episode {e+1}/{episodes}, Reward={total_reward:.2f}, Epsilon={agent.epsilon:.2f}")

    env.close()
