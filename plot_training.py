import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/dqn_flappy_w5/train_metrics.csv")

plt.figure(figsize=(8,4))
plt.plot(df["episode"], df["moving_return"], label="50-episode moving return")
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.title("FlappyBird DQN Training Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
