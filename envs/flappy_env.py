import gymnasium as gym
import flappy_bird_gymnasium 

def make_env(seed: int = 2025):
    """Environment for training (no rendering)."""
    env = gym.make("FlappyBird-v0", render_mode=None)
    env.reset(seed=seed)
    return env

def make_env_render(seed: int = 2025):
    """Environment for recording or visualizing (returns RGB frames)."""
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env.reset(seed=seed)
    return env
