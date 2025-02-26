from environment.pacman import PacManEnv
from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from agents.monte_carlo import MonteCarloAgent
import torch
import argparse

def evaluate_dqn(env, model_path):
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n
    agent = DQNAgent(state_dim, action_size)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    state, total_reward, done = env.reset(), 0, False
    while not done:
        action = agent.choose_action(state, epsilon=0)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"[DQN] Final Score: {total_reward}")

def evaluate_q_learning(env, q_table_path):
    import pickle
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)

    state, total_reward, done = tuple(env.pacman_pos), 0, False
    while not done:
        action = int(np.argmax(q_table[state]))
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"[Q-Learning] Final Score: {total_reward}")

def evaluate_monte_carlo(env, q_values_path):
    import pickle
    with open(q_values_path, "rb") as f:
        q = pickle.load(f)

    state, total_reward, done = tuple(env.pacman_pos), 0, False
    while not done:
        action = int(np.argmax(q[state])) if state in q else env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"[Monte Carlo] Final Score: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Pac-Man agents.")
    parser.add_argument("--agent", choices=["dqn", "q_learning", "monte_carlo"], required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model.")
    args = parser.parse_args()

    env = PacManEnv()

    if args.agent == "dqn":
        evaluate_dqn(env, args.model_path)
    elif args.agent == "q_learning":
        evaluate_q_learning(env, args.model_path)
    elif args.agent == "monte_carlo":
        evaluate_monte_carlo(env, args.model_path)
