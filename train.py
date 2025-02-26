import torch
import pickle
import random
import numpy as np

from environment.pacman import PacManEnv
from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from agents.monte_carlo import MonteCarloAgent

# Device setup for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}", flush=True)

# Default number of episodes and max steps safeguard per episode
DEFAULT_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1000  # Prevents a single episode from running indefinitely


# ------------------------- DQN Training -------------------------
def train_dqn(env, episodes=DEFAULT_EPISODES):
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n
    agent = DQNAgent(state_dim, action_size)
    epsilon, epsilon_decay, min_epsilon = 1.0, 0.995, 0.01

    print(f"Starting DQN training for {episodes} episodes...", flush=True)
    for ep in range(episodes):
        state = env.reset()
        total_reward, done, step_count = 0, False, 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store((state, action, reward, next_state, done))

            if step_count % 4 == 0:  # Train every 4 steps for efficiency
                agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

            # Intermediate logging every 100 steps
            if step_count % 100 == 0:
                print(f"[DQN] Episode {ep + 1} progress: step {step_count}", flush=True)

        if step_count >= MAX_STEPS_PER_EPISODE:
            #print(f"[DQN] Episode {ep + 1} reached max steps limit ({MAX_STEPS_PER_EPISODE}). Forcing termination.",flush=True)
            done = True

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(
            f"[DQN] Episode {ep + 1}/{episodes} | Reward: {total_reward} | Epsilon: {epsilon:.3f} | Steps: {step_count} | Episodes left: {episodes - (ep + 1)}",
            flush=True)

        if (ep + 1) % 50 == 0:
            torch.save(agent.model.state_dict(), f"dqn_model_ep{ep + 1}.pth")
            print(f"[DQN] Checkpoint saved at episode {ep + 1}.", flush=True)

    if episodes % 50 != 0:
        torch.save(agent.model.state_dict(), "dqn_model_final.pth")
        print("[DQN] Final model saved as dqn_model_final.pth", flush=True)


# --------------------- Q-Learning Training ----------------------
def train_q_learning(env, episodes=DEFAULT_EPISODES):
    state_space = (env.grid_size, env.grid_size)
    action_size = env.action_space.n
    agent = QLearningAgent(state_space, action_size)
    epsilon, epsilon_decay, min_epsilon = 1.0, 0.995, 0.01

    print(f"Starting Q-Learning training for {episodes} episodes...", flush=True)
    for ep in range(episodes):
        state = tuple(env.pacman_pos)
        total_reward, done, step_count = 0, False, 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, tuple(env.pacman_pos), done)
            state = tuple(env.pacman_pos)
            total_reward += reward
            step_count += 1

            if step_count % 100 == 0:
                print(f"[Q-Learning] Episode {ep + 1} progress: step {step_count}", flush=True)

        if step_count >= MAX_STEPS_PER_EPISODE:
            print(
                f"[Q-Learning] Episode {ep + 1} reached max steps limit ({MAX_STEPS_PER_EPISODE}). Forcing termination.",
                flush=True)
            done = True

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(
            f"[Q-Learning] Episode {ep + 1}/{episodes} | Reward: {total_reward} | Epsilon: {epsilon:.3f} | Steps: {step_count} | Episodes left: {episodes - (ep + 1)}",
            flush=True)

        if (ep + 1) % 50 == 0:
            with open(f"q_learning_model_ep{ep + 1}.pkl", "wb") as f:
                pickle.dump(agent.q_table, f)
            print(f"[Q-Learning] Checkpoint saved at episode {ep + 1}.", flush=True)

    if episodes % 50 != 0:
        with open("q_learning_model_final.pkl", "wb") as f:
            pickle.dump(agent.q_table, f)
        print("[Q-Learning] Final model saved as q_learning_model_final.pkl", flush=True)


# ------------------ Monte Carlo Training ------------------------
def train_monte_carlo(env, episodes=DEFAULT_EPISODES):
    action_size = env.action_space.n
    agent = MonteCarloAgent(action_size)
    epsilon, epsilon_decay, min_epsilon = 1.0, 0.995, 0.01

    print(f"Starting Monte Carlo training for {episodes} episodes...", flush=True)
    for ep in range(episodes):
        state = tuple(env.pacman_pos)
        episode_data, total_reward, done, step_count = [], 0, False, 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = tuple(env.pacman_pos)
            total_reward += reward
            step_count += 1

            if step_count % 100 == 0:
                print(f"[Monte Carlo] Episode {ep + 1} progress: step {step_count}", flush=True)

        if step_count >= MAX_STEPS_PER_EPISODE:
            print(
                f"[Monte Carlo] Episode {ep + 1} reached max steps limit ({MAX_STEPS_PER_EPISODE}). Forcing termination.",
                flush=True)
            done = True

        agent.update(episode_data)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(
            f"[Monte Carlo] Episode {ep + 1}/{episodes} | Reward: {total_reward} | Steps: {step_count} | Episodes left: {episodes - (ep + 1)}",
            flush=True)

        if (ep + 1) % 50 == 0:
            with open(f"monte_carlo_model_ep{ep + 1}.pkl", "wb") as f:
                pickle.dump(agent.q, f)
            print(f"[Monte Carlo] Checkpoint saved at episode {ep + 1}.", flush=True)

    if episodes % 50 != 0:
        with open("monte_carlo_model_final.pkl", "wb") as f:
            pickle.dump(agent.q, f)
        print("[Monte Carlo] Final model saved as monte_carlo_model_final.pkl", flush=True)


# -------------------------- Main -------------------------------
if __name__ == "__main__":
    print("Select the agent you want to train:", flush=True)
    print("Options: dqn, q_learning, monte_carlo", flush=True)
    agent_choice = input("Enter agent type: ").strip().lower()

    env = PacManEnv()

    if agent_choice == "dqn":
        train_dqn(env)
    elif agent_choice == "q_learning":
        train_q_learning(env)
    elif agent_choice == "monte_carlo":
        train_monte_carlo(env)
    else:
        print("Invalid agent selection! Please choose from dqn, q_learning, or monte_carlo.", flush=True)
