import gymnasium as gym

class FlappyRewardWrapper(gym.Wrapper):
    """
    Fix reward shaping for Flappy Bird.
    Default env reward is NOT good for RL.
    This wrapper gives:
        +1 every step you survive
        +10 for passing a pipe
        -10 for dying
    """

    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_score = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # -------- Correct reward shaping --------
        shaped_reward = 1.0  # stay alive reward

        # pipe passed? (score increased)
        score = info.get("score", 0)
        if score > self.prev_score:
            shaped_reward += 10.0
        self.prev_score = score

        # death penalty
        if terminated or truncated:
            shaped_reward -= 10.0

        return obs, shaped_reward, terminated, truncated, info
