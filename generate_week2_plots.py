import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("episode_scores.npy")
trajectories = np.load("state_trajectories.npy", allow_pickle=True)


plt.figure(figsize=(10, 5))
plt.plot(rewards, marker='o', linestyle='-')
plt.title("Rewards per Episode - Random Agent")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("rewards_per_episode.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(rewards, bins=10, color='skyblue', edgecolor='black')
plt.title("Reward Distribution (Histogram)")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.savefig("score_distribution.png")
plt.show()


# Identify shortest and longest episodes
episode_lengths = [len(ep) for ep in trajectories]
short_idx = np.argmin(episode_lengths)
long_idx = np.argmax(episode_lengths)

plt.figure(figsize=(10, 5))
plt.plot(trajectories[short_idx][:, 0], label="Short Episode", color="red")   # Bird height
plt.plot(trajectories[long_idx][:, 0], label="Long Episode", color="green")  # Bird height
plt.title("State Trajectories: Short vs Long Episodes")
plt.xlabel("Timestep")
plt.ylabel("Bird Height (feature 0)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("state_trajectories_short_vs_long.png")
plt.show()

print("\nAll 3 plots generated:")
print(" - rewards_per_episode.png")
print(" - score_distribution.png")
print(" - state_trajectories_short_vs_long.png")
