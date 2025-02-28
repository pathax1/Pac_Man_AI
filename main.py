# main.py

from training.dqn_train import train_dqn
# If you want to train with Q-Learning or Monte Carlo, import and call them similarly.

def main():
    print("Starting DQN Training...")
    train_dqn(episodes=50)

if __name__ == "__main__":
    main()
