# training/dqn_train.py

import numpy as np
import os
from environment.game import PacmanGame
from agents.dqn_agent import DQNAgent

def train_dqn(episodes=500):
    env = PacmanGame()
    state_size = len(env.get_state())  # For example, 12
    action_size = 4
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0

        while not done:
            env.render()  # Render the GUI so you can see the game
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        agent.update_target_network()
        print(f"[DQN] Episode {e + 1}/{episodes}, Reward={total_reward:.2f}, Epsilon={agent.epsilon:.2f}")

    env.close()

    # Use the absolute path for your source root folder
    source_root = r"C:\Users\Autom\PycharmProjects\Pac_Man_AI"
    if not os.path.exists(source_root):
        os.makedirs(source_root)
    save_path = os.path.join(source_root, "dqn_pacman.pth")
    print("Attempting to save model to:", save_path)
    try:
        agent.save(save_path)
    except Exception as e:
        print("Error encountered during model saving:", e)

    # Confirm the file exists
    if os.path.exists(save_path):
        print("Model saved successfully at:", save_path)
    else:
        print("Error: Model file not found after saving. Check your agent.save() method and file permissions.")

if __name__ == "__main__":
    train_dqn(episodes=500)
