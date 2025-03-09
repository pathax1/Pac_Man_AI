# play.py

import argparse
import torch
import numpy as np
from environment.pacman_env import PacmanEnv
from agents.dqn_agent import DQNNetwork
from config import DEVICE

def play_dqn(level="simple", model_path="dqn_simple_best.pth"):
    env = PacmanEnv(level=level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load model
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

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Game finished! Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="simple", choices=["simple", "medium", "complex"])
    parser.add_argument("--model_path", type=str, default="dqn_simple_best.pth")
    args = parser.parse_args()

    play_dqn(level=args.level, model_path=args.model_path)
