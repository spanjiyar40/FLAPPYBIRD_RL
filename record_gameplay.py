import os, numpy as np, torch, imageio.v2 as imageio
from envs.flappy_env import make_env_render
from agents.dqn import DQNAgent, DQNConfig

def as_obs(o):
    return np.asarray(o, dtype=np.float32).flatten()

def main():
    model_path = "outputs/models/dqn_final.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

    os.makedirs("outputs/video", exist_ok=True)

    # env that returns RGB frames
    env = make_env_render(seed=42)
    nA = env.action_space.n
    o, info = env.reset()
    obs_dim = as_obs(o).shape[0]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # rebuild agent and load weights (greedy eval: eps=0)
    agent = DQNAgent(DQNConfig(
        obs_dim=obs_dim, n_actions=nA,
        hidden=(128,128), gamma=0.99,
        eps_start=0.0, eps_end=0.0,
        device=device
    ))
    agent.load(model_path)

    # roll one episode and capture frames
    frames = []
    total_r = 0.0
    done = trunc = False
    steps_cap = 3000  
    step = 0

    while not (done or trunc) and step < steps_cap:
        # render current frame
        frame = env.render()  # ndarray HxWx3 (RGB)
        if frame is not None:
            frames.append(frame)

        # act greedily
        a = agent.act(as_obs(o))
        o, r, done, trunc, info = env.step(a)
        total_r += r
        step += 1

    # write MP4 (smaller, better for reports)
    mp4_path = "outputs/video/dqn_play.mp4"
    with imageio.get_writer(mp4_path, fps=30, codec="libx264", quality=7) as w:
        for f in frames:
            w.append_data(f)

    # optional small GIF (bigger file, downsample frames)
    gif_path = "outputs/video/dqn_play.gif"
    imageio.mimsave(gif_path, frames[::2], fps=15)

    print(f"Saved: {mp4_path}")
    print(f"Saved: {gif_path}")
    print(f"Episode reward: {total_r:.2f}, frames: {len(frames)}")

if __name__ == "__main__":
    import os
    main()
