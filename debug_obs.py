from envs.flappy_env import make_env

env = make_env()
obs, info = env.reset()
print("Observation sample:", obs)
print("Length:", len(obs))
