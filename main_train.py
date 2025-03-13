import torch
import sys
from training.train_dqn import train_dqn
from training.train_qlearning import train_qlearning
from training.train_monte_carlo import train_monte_carlo

def clean_up():
    """Helper function to clean up resources and memory."""
    import gc
    torch.cuda.empty_cache()  # Clears GPU memory (if CUDA is being used)
    gc.collect()  # Force garbage collection

if __name__ == "__main__":
    levels = ["simple", "medium", "complex"]

    try:

        # Train Monte Carlo model for all maze complexities
        for level in levels:
            print(f"\n=== Preparing Monte Carlo Training for {level.capitalize()} Maze ===")
            try:
                train_monte_carlo(level=level)
                print(f"=== Completed Monte Carlo Training for {level.capitalize()} Maze ===\n")
            except Exception as e:
                print(f"[ERROR] Monte Carlo Training for {level.capitalize()} Maze failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"Cleaning up after Monte Carlo Training for {level.capitalize()} Maze.")
                clean_up()

        # Train Q-Learning model for all maze complexities
        for level in levels:
            print(f"\n=== Preparing Q-Learning Training for {level.capitalize()} Maze ===")
            try:
                train_qlearning(level=level)
                print(f"=== Completed Q-Learning Training for {level.capitalize()} Maze ===\n")
            except Exception as e:
                print(f"[ERROR] Q-Learning Training for {level.capitalize()} Maze failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"Cleaning up after Q-Learning Training for {level.capitalize()} Maze.")
                clean_up()

        # Train DQN model for all maze complexities
        for level in levels:
            print(f"\n=== Preparing DQN Training for {level.capitalize()} Maze ===")
            try:
                train_dqn(level=level)
                print(f"=== Completed DQN Training for {level.capitalize()} Maze ===\n")
            except Exception as e:
                print(f"[ERROR] DQN Training for {level.capitalize()} Maze failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"Cleaning up after DQN Training for {level.capitalize()} Maze.")
                clean_up()

        print("✅ All trainings completed successfully!")

    except SystemExit as e:
        print(f"[ERROR] SystemExit detected: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)  # Ensure the script exits gracefully
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Main script execution completed.")
