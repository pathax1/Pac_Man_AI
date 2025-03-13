# training/train_qlearning.py

import numpy as np
from environment.pacman_env import PacmanEnv
from agents.qlearning_agent import QLearningAgent
from config import NUM_EPISODES
import pickle

def train_qlearning(level="simple"):
    env = PacmanEnv(level=level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = QLearningAgent(state_shape=state_dim, action_dim=action_dim)
    episode_rewards = []

    best_reward = float('-inf')
    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        s_tuple = tuple(map(int, state))

        while not done:
            action = agent.select_action(s_tuple)
            next_state, reward, done, _ = env.step(action)
            ns_tuple = tuple(map(int, next_state))

            agent.update(s_tuple, action, reward, ns_tuple, done)
            total_reward += reward
            s_tuple = ns_tuple

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            with open(f"qlearning_{level}_best.pkl", "wb") as f:
                pickle.dump(agent.Q, f)

        # ✅ Log progress every 100 episodes
        if ep % 100 == 0:
            print(f"[Q-Learning] Episode {ep + 1}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Pellets Consumed: {env.game.pellets_consumed} | Survival Time: {env.game.survival_time} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    with open(f"qlearning_{level}_final.pkl", "wb") as f:
        pickle.dump(agent.Q, f)

    return episode_rewards
