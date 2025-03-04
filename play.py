# play.py
import sys
import os
import pickle
import torch
from environment.game import PacmanGame
from agents.dqn_agent import DQNAgent
from agents.qlearning_agent import QLearningAgent
from agents.monte_carlo_agent import MonteCarloAgent
from config import DEVICE

# Directory where trained models (or Q-tables) are saved.
MODEL_DIR = r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\TrainedModel"

def load_dqn_agent(game):
    MODEL_PATH = os.path.join(MODEL_DIR, "dqn_pacman.pth")
    n_actions = 4  # UP, DOWN, LEFT, RIGHT
    state = game.get_state()
    state_dim = len(state)
    agent = DQNAgent(state_dim, n_actions, device=DEVICE)
    agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    agent.policy_net.eval()
    return agent

def load_qlearning_agent(game):
    QTABLE_PATH = os.path.join(MODEL_DIR, "qlearning_qtable.pkl")
    n_actions = 4
    with open(QTABLE_PATH, "rb") as f:
        q_table = pickle.load(f)
    agent = QLearningAgent(n_actions)
    agent.q_table = q_table
    return agent

def load_montecarlo_agent(game):
    MC_QTABLE_PATH = os.path.join(MODEL_DIR, "monte_carlo_qtable.pkl")
    n_actions = 4
    with open(MC_QTABLE_PATH, "rb") as f:
        Q = pickle.load(f)
    agent = MonteCarloAgent(n_actions)
    agent.Q = Q
    return agent

def main():
    if len(sys.argv) < 2:
        print("Usage: python play.py [dqn|qlearning|montecarlo]")
        return

    mode = sys.argv[1].lower()
    game = PacmanGame()

    # Load the appropriate agent based on user input.
    if mode == "dqn":
        agent = load_dqn_agent(game)
    elif mode == "qlearning":
        agent = load_qlearning_agent(game)
    elif mode == "montecarlo":
        agent = load_montecarlo_agent(game)
    else:
        print("Unknown mode. Use one of: dqn, qlearning, montecarlo")
        return

    done = False
    state = game.get_state()
    total_reward = 0

    # Main game loop.
    while not done:
        if mode == "dqn":
            # Use the DQN network to select a greedy action.
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()
        elif mode == "qlearning":
            # Use the loaded Q-table to select a greedy action.
            q_values = [agent.q_table.get((state, a), 0.0) for a in range(4)]
            action = int(max(range(4), key=lambda a: q_values[a]))
        elif mode == "montecarlo":
            # Use the Monte Carlo Q-values for greedy action selection.
            q_values = [agent.Q.get((state, a), 0.0) for a in range(4)]
            action = int(max(range(4), key=lambda a: q_values[a]))

        # Execute action in the environment.
        next_state, reward, done, _ = game.step(action)
        state = game.get_state()
        total_reward += reward

        # Render the maze so you can see the game.
        game.render()

    print("Total Score:", game.score)
    game.close()

if __name__ == "__main__":
    main()
