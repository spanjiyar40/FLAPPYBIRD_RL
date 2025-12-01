import pandas as pd

experiments = {
    "baseline (lr=1e-3, bs=64, γ=0.99)": "runs/dqn_base_100k/train_metrics.csv",
    "lr=1e-4": "runs/dqn_lr1e4/train_metrics.csv",
    "batch_size=32": "runs/dqn_batch32/train_metrics.csv",
    "gamma=0.90": "runs/dqn_gamma90/train_metrics.csv",
}

rows = []

for name, path in experiments.items():
    df = pd.read_csv(path)
    df["moving_return"] = pd.to_numeric(df["moving_return"], errors="coerce")

    final_mavg = df["moving_return"].iloc[-1]
    best_mavg = df["moving_return"].max()

    rows.append({
        "experiment": name,
        "final_moving_return": round(final_mavg, 3),
        "best_moving_return": round(best_mavg, 3),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
