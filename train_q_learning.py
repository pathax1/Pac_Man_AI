from environment.pacman_env import PacmanEnv
from agents.q_learning import QLearningAgent

if __name__ == "__main__":
    env = PacmanEnv()
    agent = QLearningAgent(actions=[0, 1, 2, 3])

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            env.render()

        print(f"Episode {episode + 1}/{episodes} completed.")