import sys
import os
import numpy as np  # ✅ Import NumPy
from environment.game import PacmanGame
from agents.dqn_agent import DQNAgent  # ✅ Import DQNAgent

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def train_dqn(episodes=50):
    env = PacmanGame()
    state_size = len(env.get_state())
    action_size = 4
    agent = DQNAgent(state_size, action_size)

    print("[DEBUG] Starting DQN Training...")

    for e in range(episodes):
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        agent.update_target_network()
        print(f"[DQN] Episode {e + 1}/{episodes}, Reward={total_reward:.2f}, Epsilon={agent.epsilon:.2f}")

    env.close()

    print("[DEBUG] Training finished, preparing to save model...")

    # Ensure directory exists before saving
    save_dir = os.path.join(os.getcwd(), "TrainedModel")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "dqn_pacman.pth")
    print(f"[DEBUG] Attempting to call agent.save() with path: {save_path}")

    try:
        print("[DEBUG] Manually calling save() now...")
        agent.save(save_path)
        print("[DEBUG] agent.save() function executed.")
    except Exception as e:
        print("[ERROR] Error while saving model:", e)

    # Manually list files in TrainedModel directory
    print("[DEBUG] Checking if the file exists in TrainedModel/")
    print("Files in TrainedModel:", os.listdir(save_dir))

    if os.path.exists(save_path):
        print(f"[SUCCESS] Model saved successfully at: {save_path}")
    else:
        print("[ERROR] Model file not found after saving.")

    # Test writing a file to check permissions
    test_path = os.path.join(save_dir, "test_file.txt")
    with open(test_path, "w") as f:
        f.write("Testing file permissions")
    print(f"[DEBUG] Test file written successfully: {test_path}")

if __name__ == "__main__":
    train_dqn()
