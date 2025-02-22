import torch
import numpy as np
from models.dqn import DQN
from environment.pacman_env import PacmanEnv

def evaluate(model, env, episodes=10):
    total_reward = 0
    for episode in range(episodes):
        state = env.reset().flatten()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state.flatten()
    print(f"Average Reward over {episodes} episodes: {total_reward/episodes:.2f}")

if __name__ == "__main__":
    env = PacmanEnv()
    input_dim = np.prod(env.observation_space.shape)
    output_dim = env.action_space.n
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load("models/dqn_model.pth", map_location=torch.device('cpu')))
    evaluate(model, env)
