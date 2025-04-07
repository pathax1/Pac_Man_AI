import argparse
import torch
import numpy as np
import pickle
from environment.pacman_env import PacmanEnv
from agents.dqn_agent import DQNNetwork
from agents.qlearning_agent import QNetwork
from config import DEVICE

def play_dqn(level, model_path):
    env = PacmanEnv(level=level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQNNetwork(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = model(state_t)
        action = q_values.argmax(dim=1).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"DQN Game finished! Total reward: {total_reward:.2f}")
    env.close()

def play_qlearning(level, model_path):
    env = PacmanEnv(level=level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = QNetwork(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = model(state_t)
        action = q_values.argmax(dim=1).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Q-Learning Game finished! Total rewards: {total_reward:.2f}")
    env.close()

def play_montecarlo(level, model_path):
    env = PacmanEnv(level=level)
    with open(model_path, "rb") as f:
        Q = pickle.load(f)

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        state_tuple = tuple(map(int, state))
        q_vals = [Q.get((state_tuple, a), 0) for a in range(env.action_space.n)]
        action = int(np.argmax(q_vals))
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Monte Carlo Game finished! Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="simple", choices=["simple", "medium", "complex"])
    parser.add_argument("--agent", type=str, required=True, choices=["dqn", "qlearning", "montecarlo"])
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    if args.agent == "dqn":
        play_dqn(args.level, args.model_path)
    elif args.agent == "qlearning":
        play_qlearning(args.level, args.model_path)
    elif args.agent == "montecarlo":
        play_montecarlo(args.level, args.model_path)
