# training/train_dqn.py

import torch
from environment.pacman_env import PacmanEnv
from agents.dqn_agent import DQNAgent
from config import TARGET_UPDATE, NUM_EPISODES

def train_dqn(level="simple"):
    print(f"Initializing environment for {level.capitalize()} maze...")
    env = PacmanEnv(level=level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Initializing DQN agent for {level.capitalize()} maze...")
    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []

    best_reward = float('-inf')
    for ep in range(NUM_EPISODES):
        print(f"Starting episode {ep + 1}/{NUM_EPISODES} for {level.capitalize()} maze...")
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.update_epsilon()

        # Update target network
        if ep % TARGET_UPDATE == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)

        # Save the best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_net.state_dict(), f"dqn_{level}_best.pth")
            print(f"New best model saved for {level.capitalize()} maze with reward: {best_reward:.2f}")

        # Log progress every 100 episodes
        if ep % 100 == 0:
            print(f"[DQN] Episode {ep + 1}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Pellets Consumed: {env.game.pellets_consumed} | Survival Time: {env.game.survival_time} | Epsilon: {agent.epsilon:.3f}")

    # Final save
    torch.save(agent.q_net.state_dict(), f"dqn_{level}_final.pth")
    print(f"Completed training for {level.capitalize()} maze. Final model saved.")

    env.close()
    return episode_rewards