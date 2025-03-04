import torch
import gym
import numpy as np
from agents.dqn_agent import DQNAgent
from training.dqn_train import preprocess_observation  # ✅ Now available!

# Path to the trained model
MODEL_PATH = "dqn_pacman.pth"


def play_trained_agent():
    """
    Load the trained model and play the game using the learned policy.
    """
    env = gym.make("MsPacman-v0")
    n_actions = env.action_space.n

    # Get input shape from environment
    initial_obs = env.reset()
    processed_obs = preprocess_observation(initial_obs)  # ✅ Now works!
    input_shape = processed_obs.shape

    # Create agent and load trained model
    agent = DQNAgent(input_shape, n_actions)
    agent.load_model(MODEL_PATH)  # Load trained weights

    total_reward = 0
    obs = env.reset()
    obs = preprocess_observation(obs)  # ✅ Now works!
    done = False

    while not done:
        env.render()  # Show the game screen
        action = agent.select_action(obs)  # Select action using trained model
        next_obs, reward, done, _ = env.step(action)
        next_obs = preprocess_observation(next_obs)  # ✅ Now works!
        obs = next_obs
        total_reward += reward

    print(f"🎮 Game Over! Final Score: {total_reward}")
    env.close()


if __name__ == "__main__":
    play_trained_agent()
