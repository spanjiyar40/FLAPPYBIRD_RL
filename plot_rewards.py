import numpy as np
import matplotlib.pyplot as plt

# Load rewards
rewards = np.load("episode_rewards.npy")

# Line plot: Rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(rewards, marker='o', linestyle='-')
plt.title("Rewards per Episode - Random Agent")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("rewards_per_episode.png")
plt.show()


# Histogram: Reward distribution
plt.figure(figsize=(8, 5))
plt.hist(rewards, bins=10, color='skyblue', edgecolor='black')
plt.title("Reward Distribution (Histogram)")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.savefig("reward_histogram.png")
plt.show()
