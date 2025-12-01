import pandas as pd
import matplotlib.pyplot as plt

paths = {
    "DQN Baseline": "runs/dqn_batch32/train_metrics.csv",
    "Double DQN": "runs/double_dqn_baseline/train_metrics.csv"
}

plt.figure(figsize=(9,5))

for label, path in paths.items():
    df = pd.read_csv(path)
    plt.plot(df["episode"], df["moving_return"], label=label)

plt.xlabel("Episode")
plt.ylabel("50-episode moving return")
plt.title("DQN vs Double DQN Training Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

