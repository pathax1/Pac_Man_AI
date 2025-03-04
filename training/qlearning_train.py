# training/qlearning_train.py
import os
import pickle
from environment.game import PacmanGame
from agents.qlearning_agent import QLearningAgent
from config import NUM_EPISODES

# Set directory and file path for Q-table
MODEL_DIR = r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\TrainedModel"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
QTABLE_PATH = os.path.join(MODEL_DIR, "qlearning_qtable.pkl")

def train_qlearning():
    game = PacmanGame()
    n_actions = 4
    agent = QLearningAgent(n_actions)
    rewards_per_episode = []

    for episode in range(NUM_EPISODES):
        game.reset()
        state = game.get_state()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = game.step(action)
            # Update state from game after step
            next_state = game.get_state()
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # Optionally render:
            # game.render()
        rewards_per_episode.append(total_reward)
        print(f"Q-Learning Episode {episode} Reward: {total_reward}")
    # Save Q-table
    with open(QTABLE_PATH, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Q-table saved at {QTABLE_PATH}")
    game.close()
    return rewards_per_episode

if __name__ == "__main__":
    train_qlearning()
