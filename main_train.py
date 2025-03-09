# main_train.py

import argparse
from training.train_dqn import train_dqn
from training.train_qlearning import train_qlearning
from training.train_monte_carlo import train_monte_carlo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "qlearning", "monte_carlo"])
    parser.add_argument("--level", type=str, default="simple", choices=["simple", "medium", "complex"])
    args = parser.parse_args()

    if args.agent == "dqn":
        train_dqn(level=args.level)
    elif args.agent == "qlearning":
        train_qlearning(level=args.level)
    else:
        train_monte_carlo(level=args.level)
