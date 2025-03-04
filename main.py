# main.py
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [dqn|qlearning|montecarlo]")
        return
    mode = sys.argv[1].lower()
    if mode == "dqn":
        from training.dqn_train import train_dqn
        train_dqn()
    elif mode == "qlearning":
        from training.qlearning_train import train_qlearning
        train_qlearning()
    elif mode == "montecarlo":
        from training.monte_carlo_train import train_monte_carlo
        train_monte_carlo()
    else:
        print("Unknown mode. Use one of: dqn, qlearning, montecarlo")

if __name__ == "__main__":
    main()
