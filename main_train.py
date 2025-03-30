import torch
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import traceback
from tqdm import tqdm

# Use relative paths for training functions
from training.train_dqn import train_dqn
from training.train_qlearning import train_qlearning
from training.train_monte_carlo import train_monte_carlo

def clean_up():
    import gc
    torch.cuda.empty_cache()
    gc.collect()

def run_agent(agent_fn, agent_name, level):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{agent_name}_{level}.log")
    with open(log_path, "w") as log_file:
        log_file.write(f"[{agent_name.upper()}] Training started on {level} maze at {datetime.now()}\n")
        try:
            import builtins
            original_print = print

            def patched_print(*args, **kwargs):
                original_print(*args, **kwargs)
                if 'file' not in kwargs:
                    print(*args, **kwargs, file=log_file)

            builtins.print = patched_print

            total_episodes = 100000
            with tqdm(total=total_episodes, desc=f"{agent_name.upper()} - {level}", file=sys.stdout, dynamic_ncols=True) as pbar:
                def print_hook(*args, **kwargs):
                    msg = " ".join(map(str, args))
                    if "Episode" in msg:
                        try:
                            current = int(msg.split("Episode")[-1].split("/")[0].strip())
                            pbar.n = current
                            pbar.refresh()
                        except:
                            pass
                    patched_print(*args, **kwargs)

                builtins.print = print_hook
                agent_fn(level=level)

            builtins.print = original_print
            log_file.write(f"[{agent_name.upper()}] Training completed on {level} maze at {datetime.now()}\n")
        except Exception as e:
            log_file.write(f"[ERROR] {agent_name.upper()} Training failed: {e}\n")
            traceback.print_exc(file=log_file)
        finally:
            clean_up()

if __name__ == "__main__":
    levels = ["simple", "medium", "complex"]
    jobs = [
        #(train_monte_carlo, "montecarlo", level) for level in levels
    ] + [
       # (train_qlearning, "qlearning", level) for level in levels
    ] + [
        (train_dqn, "dqn", level) for level in levels
    ]

    try:
        print("Starting multiprocessing training with logs...")
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_agent, fn, name, level) for fn, name, level in jobs]
            for f in futures:
                f.result()
        print("All trainings completed successfully!")

    except SystemExit as e:
        print(f"[ERROR] SystemExit detected: {e}")
        traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        print("Main script execution completed.")
