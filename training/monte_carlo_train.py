# training/monte_carlo_train.py
import os
import pickle
from environment.game import PacmanGame
from agents.monte_carlo_agent import MonteCarloAgent
from config import NUM_EPISODES

# Set directory and file path for Monte Carlo Q-values
MODEL_DIR = r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\TrainedModel"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MC_QTABLE_PATH = os.path.join(MODEL_DIR, "monte_carlo_qtable.pkl")

def train_monte_carlo():
    game = PacmanGame()
    n_actions = 4
    agent = MonteCarloAgent(n_actions)
    rewards_per_episode = []

    for episode in range(NUM_EPISODES):
        game.reset()
        state = game.get_state()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = game.step(action)
            agent.store_transition(state, action, reward)
            state = game.get_state()
            total_reward += reward
            # Optionally render:
            # game.render()
        agent.update()
        rewards_per_episode.append(total_reward)
        print(f"Monte Carlo Episode {episode} Reward: {total_reward}")
    # Save Monte Carlo Q-values
    with open(MC_QTABLE_PATH, "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print(f"Monte Carlo Q-values saved at {MC_QTABLE_PATH}")
    game.close()
    return rewards_per_episode

if __name__ == "__main__":
    train_monte_carlo()
