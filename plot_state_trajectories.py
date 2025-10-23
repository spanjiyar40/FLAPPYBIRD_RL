import numpy as np
import matplotlib.pyplot as plt


trajectories = np.load("state_trajectories.npy", allow_pickle=True)

# Compute episode lengths
episode_lengths = [len(ep) for ep in trajectories]

# Identify shortest and longest episode
short_idx = np.argmin(episode_lengths)
long_idx = np.argmax(episode_lengths)

# Plotting vertical position (example: feature 0 = bird height)
plt.figure(figsize=(10, 5))
plt.plot(trajectories[short_idx][:, 0], label="Short Episode", color="red")
plt.plot(trajectories[long_idx][:, 0], label="Long Episode", color="green")
plt.title("State Trajectories: Short vs Long Episodes")
plt.xlabel("Timestep")
plt.ylabel("Bird Height (or relevant state)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save figure
plt.savefig("state_trajectories_short_vs_long.png", dpi=300)
plt.show()
