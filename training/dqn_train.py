# training/dqn_train.py
import os
import torch
from environment.game import PacmanGame
from agents.dqn_agent import DQNAgent
from config import NUM_EPISODES, TARGET_UPDATE, DEVICE

# Set directory and model path
MODEL_DIR = r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\TrainedModel"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_pacman.pth")

def train_dqn():
    game = PacmanGame()
    n_actions = 4  # UP, DOWN, LEFT, RIGHT
    state = game.get_state()
    state_dim = len(state)

    agent = DQNAgent(state_dim, n_actions, device=DEVICE)
    rewards_per_episode = []

    for episode in range(NUM_EPISODES):
        game.reset()
        state = game.get_state()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = game.step(action)
            # Update state from game (in case game internally changes it)
            next_state = game.get_state()
            agent.store_transition(state, action, reward, next_state, done)
            _ = agent.optimize_model()
            state = next_state
            total_reward += reward
            # Optionally, render the game:
            # game.render()
        rewards_per_episode.append(total_reward)
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        print(f"DQN Episode {episode} Reward: {total_reward}")
    agent.save_model(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
    game.close()
    return rewards_per_episode

if __name__ == "__main__":
    train_dqn()
