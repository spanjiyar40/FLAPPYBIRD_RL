import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import random
from collections import defaultdict

from utils.discretization import discretize, choose_action, Q, set_seed
from utils.plotting import plot_avg_reward, plot_epsilon_decay

# Q-Learning configuration
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
EPISODES = 10000
PRINT_EVERY = 500

rewards_history = []
epsilon_history = []

# Q-learning update rule
def q_update(state, action, reward, next_state, done):
    current = Q[state][action]
    target = reward
    if not done:
        target += GAMMA * np.max(Q[next_state])
    Q[state][action] += ALPHA * (target - current)

# Training loop
def train(env):
    set_seed(42)
    epsilon = EPSILON_START

    for ep in range(EPISODES):
        obs = env.reset()
        state = discretize(obs)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = choose_action(state, epsilon)
            next_obs, reward, done, info = env.step(action)
            next_state = discretize(next_obs)

            q_update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        epsilon_history.append(epsilon)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (ep + 1) % PRINT_EVERY == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(f"Episode {ep+1:5d} | Avg Reward: {avg_r:.2f} | ε={epsilon:.3f}")

    return rewards_history, epsilon_history

# Quick test with dummy environment
if __name__ == "__main__":
    class DummyEnv:
        def reset(self):
            return {'y': random.uniform(0, 512),
                    'vel': random.uniform(-8, 8),
                    'pipe_dist': random.uniform(0, 600)}

        def step(self, action):
            next_obs = {'y': random.uniform(0, 512),
                        'vel': random.uniform(-8, 8),
                        'pipe_dist': random.uniform(0, 600)}
            reward = random.choice([0, 1])
            done = random.random() < 0.05
            info = {}
            return next_obs, reward, done, info

    env = DummyEnv()
    rewards, epsilons = train(env)

    # Save plots
    os.makedirs("results", exist_ok=True)
    plot_avg_reward(rewards, save_path="results/avg_reward_curve.png")
    plot_epsilon_decay(epsilons, save_path="results/epsilon_decay_curve.png")

    print("\nTraining complete — plots saved in /results/:")
    print("  • results/avg_reward_curve.png")
    print("  • results/epsilon_decay_curve.png")
