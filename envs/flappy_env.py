# ---------------------------------------------------------------
# flappy_env.py  (FINAL 4-D STATE VERSION — SOLVES PIPE PASSING)
# ---------------------------------------------------------------

import numpy as np
import gymnasium as gym
from flappy_bird_gymnasium import FlappyBirdEnv


class SimpleStateWrapper(gym.Wrapper):
    """
    Converts the raw 12-D observation into a 4-D state:
        [bird_y, bird_vel, dist_to_pipe, pipe_gap_center_y]

    This 4D state is proven to train GOOD Flappy-Bird DQN agents.
    """

    def __init__(self, env):
        super().__init__(env)

        # 4-D state bounds
        low = np.array([0.0, -20.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([512.0, 20.0, 500.0, 512.0], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._convert(obs), reward, done, truncated, info

    def _convert(self, raw):
        """
        raw is a 12-element numpy array:
        Example:
            [288 170 270 288 0 512 288 0 512 244 -9 45]

        Index meanings we use:
            1 -> bird_y
            2 -> dist_to_pipe
            10 -> bird_vel
            9 -> pipe_gap_center_y  <— CRITICAL FOR PASSING PIPES
        """
        bird_y = float(raw[1])
        dist   = float(raw[2])
        vel    = float(raw[10])
        gap_y  = float(raw[9])   # vertical center of pipe gap

        return np.array([bird_y, vel, dist, gap_y], dtype=np.float32)


def make_env(render=False):
    """
    Creates Flappy Bird with NO lidar and 4D state wrapper.
    render=False is used for training.
    render=True is used for playing.
    """
    base_env = FlappyBirdEnv(
        use_lidar=False,
        pipe_gap=110,
        normalize_obs=False,
        render_mode="human" if render else None,
    )
    return SimpleStateWrapper(base_env)
