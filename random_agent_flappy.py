"""
Flappy Bird - Random Agent
Week 1: Environment Setup & Random Actions
"""

import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np

# Parameters
N_EPISODES = 50  # Number of episodes to run

# Create environment
env = gym.make("FlappyBird-v0", render_mode="human")  # "human" mode shows the game window
episode_rewards = []  # List to store total rewards per episode

# Run random episodes
for episode in range(N_EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Random action: 0=no flap, 1=flap
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()  # Display the game

    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}/{N_EPISODES} - Reward: {total_reward}")

# Close environment
env.close()

# Save rewards to file
np.save("episode_rewards.npy", episode_rewards)
print("All episodes completed. Rewards saved to 'episode_rewards.npy'")
print("Average Reward:", np.mean(episode_rewards))
