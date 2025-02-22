import pygame

from environment.pacman_env import PacmanEnv
from agents.q_learning import QLearningAgent

if __name__ == "__main__":
    # Prompt user to select an agent type
    agent_type = input("Select agent (dqn/q_learning/monte_carlo): ").strip().lower()

    # Initialize the environment with the selected agent type
    env = PacmanEnv(agent_type=agent_type)

    state = env.reset()
    done = False

    print("\n🎮 Starting Pac-Man with AI agent:\n")

    while not done:
        env.render()  # Display the map

        # Use random actions for now (replace with model prediction after training)
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if reward > 0:
            print(f"🍪 Pellet collected! Current Score: {env.score}")

    print(f"🏆 Game Over! Final Score: {env.score}")