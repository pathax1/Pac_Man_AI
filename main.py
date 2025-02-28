# main.py

from training.dqn_train import train_dqn

def main():
    print("[DEBUG] Starting DQN Training...")
    train_dqn(episodes=50)

if __name__ == "__main__":
    print("[DEBUG] Starting main script...")
    main()
    print("[DEBUG] Training function completed.")
