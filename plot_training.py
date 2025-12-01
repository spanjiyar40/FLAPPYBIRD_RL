import pandas as pd
import matplotlib.pyplot as plt

# Change to your new run directory
df = pd.read_csv("runs/dqn_base_100k/train_metrics.csv")

plt.figure(figsize=(10, 5))

# Plot moving average (main signal)
plt.plot(df["episode"], df["moving_return"], label="50-episode moving return", linewidth=2)

# Plot raw episode returns (optional but useful)
plt.plot(df["episode"], df["ep_return"], label="Episode Return", alpha=0.4)

plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("FlappyBird DQN Training Curve (300k Steps)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
