import os, numpy as np, matplotlib.pyplot as plt, torch
from tqdm import trange
from envs.flappy_env import make_env
from agents.dqn import DQNAgent, DQNConfig


def as_obs(o):
    """Flatten observation from env."""
    return np.asarray(o, dtype=np.float32).flatten()


def evaluate(agent, env, episodes=200, render=False):
    """Run the trained agent for evaluation."""
    rewards = []
    for ep in trange(episodes, desc="Evaluating"):
        o, info = env.reset()
        s = as_obs(o)
        total_r = 0.0
        done = trunc = False
        while not (done or trunc):
            a = agent.act(s)
            o2, r, done, trunc, info = env.step(a)
            s = as_obs(o2)
            total_r += r
            if render:
                env.render()
        rewards.append(total_r)
    return np.array(rewards, dtype=np.float32)


def evaluate_random(env, episodes=200):
    """Evaluate a random baseline for comparison."""
    rewards = []
    for ep in trange(episodes, desc="Random Baseline"):
        o, info = env.reset()
        total_r = 0.0
        done = trunc = False
        while not (done or trunc):
            a = env.action_space.sample()
            o2, r, done, trunc, info = env.step(a)
            total_r += r
        rewards.append(total_r)
    return np.array(rewards, dtype=np.float32)


def main():
    model_path = "outputs/models/dqn_final.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun train_dqn.py first.")

    # environment setup
    env = make_env(seed=42)
    nA = env.action_space.n
    o, info = env.reset()
    obs_dim = as_obs(o).shape[0]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # recreate and load trained agent
    agent = DQNAgent(DQNConfig(
        obs_dim=obs_dim,
        n_actions=nA,
        hidden=(128, 128),
        gamma=0.99,
        eps_start=0.0, 
        eps_end=0.0,
        device=device
    ))
    agent.load(model_path)

    # evaluate trained DQN
    dqn_rewards = evaluate(agent, env, episodes=200)
    np.save("outputs/logs/dqn_eval_rewards.npy", dqn_rewards)

    # evaluate random policy
    random_rewards = evaluate_random(env, episodes=200)
    np.save("outputs/logs/random_eval_rewards.npy", random_rewards)

    # plot comparison
    plt.figure()
    plt.plot(np.cumsum(dqn_rewards)/np.arange(1, len(dqn_rewards)+1), label="DQN Agent")
    plt.plot(np.cumsum(random_rewards)/np.arange(1, len(random_rewards)+1), label="Random Agent")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("DQN vs Random Baseline")
    plt.legend()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/eval_dqn_vs_random.png", dpi=150, bbox_inches="tight")

    print("\n Saved: outputs/plots/eval_dqn_vs_random.png")
    print(f"DQN mean reward: {np.mean(dqn_rewards):.2f}")
    print(f"Random mean reward: {np.mean(random_rewards):.2f}")


if __name__ == "__main__":
    main()
