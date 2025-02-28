import gym
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from config import NUM_EPISODES, TARGET_UPDATE, DEVICE

# Path to save the trained model
MODEL_PATH = "TrainedModel/dqn_pacman.pth"


def preprocess_observation(obs):
    """
    Convert an observation (H, W, 3) or (H, W)
    into a normalized grayscale array with shape (C, H, W).
    """
    obs = obs.astype(np.float32) / 255.0  # Normalize pixel values to [0,1]

    # Convert RGB image (H, W, 3) to grayscale (H, W, 1)
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        obs = obs.mean(axis=2, keepdims=True)  # Convert to grayscale
    elif len(obs.shape) == 2:
        obs = obs[..., None]  # Already grayscale, add channel dim

    # Rearrange dimensions to (C, H, W) for CNN
    obs = obs.transpose(2, 0, 1)
    return obs


def train_dqn():
    """
    Train a DQN agent to play Pac-Man.
    """
    env = gym.make("ALE/MsPacman-v5")  # ✅ Works with the latest Gym versions
    n_actions = env.action_space.n

    # Get input shape from the environment
    initial_obs = env.reset()
    processed_obs = preprocess_observation(initial_obs)
    input_shape = processed_obs.shape  # (C, H, W)

    # Create DQN Agent
    agent = DQNAgent(input_shape, n_actions)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        obs = preprocess_observation(obs)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess_observation(next_obs)

            # Store transition and optimize the model
            agent.store_transition(obs, action, reward, next_obs, done)
            _ = agent.optimize_model()

            obs = next_obs
            total_reward += reward

        episode_rewards.append(total_reward)

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        print(f"Episode {episode} - Reward: {total_reward}")

    # Save the trained model
    agent.save_model(MODEL_PATH)
    print(f"✅ Training complete. Model saved at {MODEL_PATH}")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    train_dqn()
