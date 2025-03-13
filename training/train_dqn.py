import torch
from environment.pacman_env import PacmanEnv
from agents.dqn_agent import DQNAgent
from config import TARGET_UPDATE, NUM_EPISODES

def train_dqn(level="simple"):
    print(f"\n🔄 Initializing environment for {level.capitalize()} maze...")

    # Check if CUDA is causing failures
    if torch.cuda.is_available():
        print(f"🖥️ Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"🔹 Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"🔹 Memory Reserved: {torch.cuda.memory_reserved()} bytes")

    env = PacmanEnv(level=level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"🤖 Initializing DQN agent for {level.capitalize()} maze...")
    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []
    best_reward = float('-inf')

    try:
        print(f"🚀 Starting DQN training for {level.capitalize()} maze with {NUM_EPISODES} episodes")
        for ep in range(NUM_EPISODES):
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

            # Log progress every 100 episodes
            if ep % 100 == 0:
                print(f"[DQN] Episode {ep + 1}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Pellets Consumed: {env.game.pellets_consumed} | Survival Time: {env.game.survival_time} | Epsilon: {agent.epsilon:.3f}")

        # Final save
        torch.save(agent.q_net.state_dict(), f"dqn_{level}_final.pth")
        print(f"✅ Completed training for {level.capitalize()} maze. Final model saved.")

    except Exception as e:
        print(f"[ERROR] DQN Training for {level.capitalize()} Maze failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"🧹 Cleaning up environment for {level.capitalize()} maze and forcing garbage collection.")
        env.close()
        del env
        del agent
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    return episode_rewards
