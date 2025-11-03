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

N_ACTIONS = 2

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

# ----------------------------
# Session 3: Real environment training, evaluation, and baseline
# ----------------------------
if __name__ == "__main__":
    try:
        from test_env import make_env  # adjust if your env factory is named differently
        def create_env():
            return make_env()
        print("Using make_env() from test_env.py")
    except Exception:
        print("No real env found — using DummyEnv")
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
                return next_obs, reward, done, {}
        def create_env():
            return DummyEnv()

    # ----------------------------
    # Train the agent
    # ----------------------------
    env = create_env()
    print("Starting training...")
    rewards, epsilons = train(env)

    os.makedirs("results", exist_ok=True)
    np.save("results/episode_rewards.npy", np.array(rewards))
    np.save("results/epsilon_history.npy", np.array(epsilons))

    plot_avg_reward(rewards, save_path="results/avg_reward_curve.png")
    plot_epsilon_decay(epsilons, save_path="results/epsilon_decay_curve.png")

    # ----------------------------
    # Evaluate learned policy (greedy)
    # ----------------------------
    def evaluate_policy(env, n_episodes=100):
        all_rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            state = discretize(obs)
            done = False
            total = 0
            while not done:
                qvals = Q[state]
                max_idxs = np.flatnonzero(qvals == qvals.max())
                action = int(np.random.choice(max_idxs))
                obs, r, done, _ = env.step(action)
                state = discretize(obs)
                total += r
            all_rewards.append(total)
        return all_rewards

    env_eval = create_env()
    learned_rewards = evaluate_policy(env_eval)
    np.save("results/learned_policy_rewards.npy", np.array(learned_rewards))

    # ----------------------------
    # Random baseline
    # ----------------------------
    def random_policy_eval(env, n_episodes=100):
        all_rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total = 0
            while not done:
                action = random.randrange(N_ACTIONS)
                obs, r, done, _ = env.step(action)
                total += r
            all_rewards.append(total)
        return all_rewards

    env_rand = create_env()
    random_rewards = random_policy_eval(env_rand)
    np.save("results/random_baseline_rewards.npy", np.array(random_rewards))

    # ----------------------------
    # Save summary stats
    # ----------------------------
    import json
    summary = {
        "training_last_1000_mean": float(np.mean(rewards[-1000:])),
        "training_last_1000_std": float(np.std(rewards[-1000:])),
        "learned_policy_mean": float(np.mean(learned_rewards)),
        "learned_policy_std": float(np.std(learned_rewards)),
        "random_baseline_mean": float(np.mean(random_rewards)),
        "random_baseline_std": float(np.std(random_rewards))
    }
    with open("results/summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n Session 3 complete — all results saved in results/:")
    print("  • avg_reward_curve.png")
    print("  • epsilon_decay_curve.png")
    print("  • episode_rewards.npy")
    print("  • epsilon_history.npy")
    print("  • learned_policy_rewards.npy")
    print("  • random_baseline_rewards.npy")
    print("  • summary_stats.json")
