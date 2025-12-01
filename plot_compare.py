import pandas as pd
import matplotlib.pyplot as plt

# Paths to your experiments
paths = {
    "baseline (lr=1e-3, bs=64, γ=0.99)": "runs/dqn_base_100k/train_metrics.csv",
    "lr=1e-4": "runs/dqn_lr1e4/train_metrics.csv",
    "batch_size=32": "runs/dqn_batch32/train_metrics.csv",
    "gamma=0.90": "runs/dqn_gamma90/train_metrics.csv",
    "batch_size=128": "runs/dqn_batch128/train_metrics.csv",
    "lr=3e-4": "runs/dqn_lr3e4/train_metrics.csv",
    "gamma=0.995": "runs/dqn_gamma995/train_metrics.csv",
}

plt.figure(figsize=(10, 6))

for label, path in paths.items():
    df = pd.read_csv(path)
    # convert to numeric just in case
    df["moving_return"] = pd.to_numeric(df["moving_return"], errors="coerce")
    plt.plot(df["episode"], df["moving_return"], label=label)

plt.xlabel("Episode")
plt.ylabel("50-episode Moving Return")
plt.title("FlappyBird DQN – Hyperparameter Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
