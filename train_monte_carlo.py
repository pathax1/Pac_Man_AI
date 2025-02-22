from environment.pacman_env import PacmanEnv
from agents.monte_carlo import MonteCarloAgent

if __name__ == "__main__":
    env = PacmanEnv()
    agent = MonteCarloAgent(actions=[0, 1, 2, 3])

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        episode_history = []
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_history.append((state, action, reward))
            state = next_state
            env.render()

        agent.update(episode_history)
        print(f"Episode {episode + 1}/{episodes} completed.")