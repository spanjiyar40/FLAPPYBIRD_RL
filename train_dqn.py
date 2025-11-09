import os, json, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from envs.flappy_env import make_env
from agents.dqn import DQNAgent, DQNConfig

def ema(x, w=0.97):
    """Exponential moving average for smoothing curves."""
    y, last = [], 0.0
    for i, v in enumerate(x):
        last = v if i == 0 else last * w + (1 - w) * v
        y.append(last)
    return np.array(y, dtype=np.float32)

def as_obs(np_obs):
    """Convert FlappyBirdGymnasium observation to flat float32 vector."""
    return np.asarray(np_obs, dtype=np.float32).flatten()

def main():
    cfg = {
        "episodes": 5000,            
        "max_steps_per_ep": 500,     
        "seed": 2025,
        "checkpoint_every": 1000    
    }

    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    with open("outputs/logs/train_config_dqn.json", "w") as f:
        json.dump(cfg, f, indent=2)

    #ENV SETUP
    env = make_env(seed=cfg["seed"])
    nA = env.action_space.n

  
    o, info = env.reset()
    obs_dim = as_obs(o).shape[0]
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # DQN AGENT
    agent = DQNAgent(DQNConfig(
        obs_dim=obs_dim,
        n_actions=nA,
        hidden=(128, 128),           
        lr=5e-4,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,    
        buffer_size=50_000,
        batch_size=128,
        target_tau=0.01,
        device=device,
        seed=cfg["seed"]
    ))

    rewards, losses, epsilons, steps_list = [], [], [], []
    t0 = time.time()

    # WARMUP (fill replay buffer)
    warmup = 1000
    s = as_obs(o)
    for _ in range(warmup):
        a = np.random.randint(nA)
        o2, r, done, trunc, info = env.step(a)
        s2 = as_obs(o2)
        agent.push(s, a, r, s2, done or trunc)
        s = s2
        if done or trunc:
            o, info = env.reset()
            s = as_obs(o)

    # MAIN TRAINING LOOP
    for ep in trange(1, cfg["episodes"] + 1, desc="Training"):
        o, info = env.reset()
        s = as_obs(o)
        ep_r, steps = 0.0, 0

        for _ in range(cfg["max_steps_per_ep"]):
            a = agent.act(s)

            #repeat same action for 3 frames
            total_r = 0
            for _ in range(3):
                o2, r, done, trunc, info = env.step(a)
                total_r += r
                if done or trunc:
                    break
            r = total_r

            s2 = as_obs(o2)
            agent.push(s, a, r, s2, done or trunc)
            loss = agent.train_step()
            if loss:
                losses.append(loss)

            ep_r += float(r)
            steps += 1
            s = s2
            if done or trunc:
                break

        rewards.append(ep_r)
        epsilons.append(agent.epsilon())
        steps_list.append(steps)

        if ep % 500 == 0:
            avg100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Ep {ep}/{cfg['episodes']}  R:{ep_r:.2f}  avg100:{avg100:.2f}  eps:{epsilons[-1]:.3f}  steps:{steps}")

        if ep % cfg["checkpoint_every"] == 0:
            agent.save(f"outputs/models/dqn_ep{ep}.pt")
            np.save("outputs/logs/rewards_dqn.npy",  np.array(rewards, dtype=np.float32))
            np.save("outputs/logs/epsilons_dqn.npy", np.array(epsilons, dtype=np.float32))
            np.save("outputs/logs/steps_dqn.npy",    np.array(steps_list, dtype=np.float32))
            np.save("outputs/logs/losses_dqn.npy",   np.array(losses, dtype=np.float32))

    
    agent.save("outputs/models/dqn_final.pt")
    np.save("outputs/logs/rewards_dqn.npy",  np.array(rewards, dtype=np.float32))
    np.save("outputs/logs/epsilons_dqn.npy", np.array(epsilons, dtype=np.float32))
    np.save("outputs/logs/steps_dqn.npy",    np.array(steps_list, dtype=np.float32))
    np.save("outputs/logs/losses_dqn.npy",   np.array(losses, dtype=np.float32))

    sm = ema(rewards, 0.97)
    plt.figure()
    plt.plot(rewards, label="reward")
    plt.plot(sm, label="smoothed")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("DQN — Reward per Episode (smoothed)")
    plt.legend()
    plt.savefig("outputs/plots/learning_curve_dqn.png", dpi=150, bbox_inches="tight")
    print(" Saved: outputs/plots/learning_curve_dqn.png")
    print(f" Done in {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()
