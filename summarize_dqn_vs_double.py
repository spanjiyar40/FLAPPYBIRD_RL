import pandas as pd

paths = {
    "DQN Baseline": "runs/dqn_batch32/train_metrics.csv",
    "Double DQN": "runs/double_dqn_baseline/train_metrics.csv"
}

rows = []

for name, path in paths.items():
    df = pd.read_csv(path)

    final = df["moving_return"].iloc[-1]
    best = df["moving_return"].min()  # most negative? or max if reward positive?

    # For Flappy Bird, higher is better → use max
    best = df["moving_return"].max()

    rows.append([name, final, best])

summary = pd.DataFrame(rows, columns=["Agent", "Final Return", "Best Return"])
print(summary)
