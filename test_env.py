import gymnasium as gym
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", render_mode=None)
obs, info = env.reset()
print("✅ Flappy Bird Gymnasium environment loaded successfully!")
print("Observation sample:", obs)
env.close()
