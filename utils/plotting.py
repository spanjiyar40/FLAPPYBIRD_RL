import matplotlib.pyplot as plt
import numpy as np
import os

def plot_avg_reward(rewards, save_path=None, window=100):
    """
    Plot average reward per N episodes.
    """
    plt.figure(figsize=(8, 5))
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f"Average reward (window={window})")
    else:
        plt.plot(rewards, label="Reward per episode")

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per 100 Episodes")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_epsilon_decay(epsilons, save_path=None):
    """
    Plot epsilon (exploration rate) decay curve.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, color='orange', label="Epsilon decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Decay over Training")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
