import os
import csv
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium  # make sure this is installed

# make folders
os.makedirs("logs", exist_ok=True)

def make_env():
    """Create and return the FlappyBird environment."""
    env = gym.make("FlappyBird-v0", render_mode=None)
    print("Environment created successfully!")
    return env

def extract_features(obs):
    """Extract relevant state features: bird_y, velocity, distance_to_pipe"""
    obs = np.array(obs).flatten()
    # crude extraction – adjust later once we see the CSV
    bird_y = float(np.mean(obs[0:60]))
    bird_vel = float(np.mean(obs[60:120]))
    dist_pipe = float(np.mean(obs[120:180]))
    return bird_y, bird_vel, dist_pipe

def run_random_episodes(env, num_episodes=10, max_steps=500):
    episode_scores = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        # prepare log
        csv_path = f"logs/episode_{ep:03d}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step", "bird_y", "bird_vel", "dist_pipe", "reward", "done"])

            for t in range(max_steps):
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                bird_y, bird_vel, dist_pipe = extract_features(obs)
                writer.writerow([ep, t, bird_y, bird_vel, dist_pipe, reward, done])

                total_reward += reward
                obs = next_obs
                steps += 1

                if done:
                    break

        episode_scores.append(total_reward)
        print(f"[Episode {ep:03d}] Steps={steps}  Score={total_reward:.2f}")

    np.save("episode_scores.npy", np.array(episode_scores))
    print("\Finished all episodes!")
    print("Saved logs in 'logs/' and rewards in 'episode_scores.npy'")

def main():
    env = make_env()
    run_random_episodes(env, num_episodes=10)
    env.close()

if __name__ == "__main__":
    main()
