# training/train_monte_carlo.py

from environment.pacman_env import PacmanEnv
from agents.monte_carlo_agent import MonteCarloAgent
from config import MC_EPISODES
import pickle

def train_monte_carlo(level="simple"):
    env = PacmanEnv(level=level)
    action_dim = env.action_space.n

    agent = MonteCarloAgent(action_dim=action_dim)
    episode_rewards = []

    best_reward = float('-inf')
    for ep in range(MC_EPISODES):
        state = env.reset()
        done = False
        episode = []
        total_reward = 0

        while not done:
            s_tuple = tuple(map(int, state))
            action = agent.select_action(s_tuple)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            episode.append((s_tuple, action, reward))
            state = next_state

        # Monte Carlo update
        agent.store_episode(episode)

        if total_reward > best_reward:
            best_reward = total_reward
            with open(f"mc_{level}_best.pkl", "wb") as f:
                pickle.dump(dict(agent.Q), f)

        episode_rewards.append(total_reward)

        # ✅ Log progress every 100 episodes
        if ep % 100 == 0:
            print(f"[MonteCarlo] Episode {ep + 1}/{MC_EPISODES} | Reward: {total_reward:.2f} | Pellets Consumed: {env.game.pellets_consumed} | Survival Time: {env.game.survival_time} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    with open(f"mc_{level}_final.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)

    return episode_rewards
