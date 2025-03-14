from environment.pacman_env import PacmanEnv
from agents.qlearning_agent import QLearningAgent
from config import NUM_EPISODES
import pickle
import torch

def train_qlearning(level="simple"):
    env = PacmanEnv(level=level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = QLearningAgent(state_dim=state_dim, action_dim=action_dim)

    best_reward = float('-inf')
    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_net.state_dict(), f"qlearning_{level}_best.pth")

        if ep % 100 == 0:
            print(f"[Q-Learning] Episode {ep}/{NUM_EPISODES} | Reward: {total_reward} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    torch.save(agent.q_net.state_dict(), f"qlearning_{level}_final.pth")
