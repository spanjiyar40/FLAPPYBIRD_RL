import os
import csv
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from agents.dqn import DQN, DQNAgent, EpsilonGreedyConfig, hard_update, count_parameters
from agents.replay_buffer import ReplayBuffer, Transition

try:
    from envs.flappy_env import make_env 
except Exception:
    make_env = None


def default_make_env():
    raise RuntimeError(
        "Please implement envs/flappy_env.py:make_env() that returns your Flappy env "
        "with obs shape (3,) or any numeric vector, and actions {0,1}."
    )


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_policy(env, agent: DQNAgent, device, n_episodes=5, render=False):
    """Evaluate policy greedily (eps≈0)."""
    total = 0.0
    for _ in range(n_episodes):
        obs_out = env.reset()
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

        done = False
        ep_ret = 0.0
        step = 10**9
        while not done:
            a = agent.act(obs, global_step=step, device=str(device))
            step_out = env.step(a)
            if len(step_out) == 5:
                next_obs, r, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, r, done, info = step_out

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]

            obs = next_obs
            ep_ret += r
            if render and hasattr(env, "render"):
                env.render()
        total += ep_ret
    return total / n_episodes


def train_loop(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "train_metrics.csv"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    env_fn = make_env if make_env is not None else default_make_env
    env = env_fn()

    if hasattr(env, "seed"):
        env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"[Device] {device}")

    obs_out = env.reset()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
    state_dim = np.array(obs).shape[0]
    print(f"[Env] Detected state_dim={state_dim}")

    q_net = DQN(state_dim=state_dim, action_dim=2,
                hidden_sizes=(args.h1, args.h2, args.h3)).to(device)
    target_net = DQN(state_dim=state_dim, action_dim=2,
                     hidden_sizes=(args.h1, args.h2, args.h3)).to(device)
    hard_update(target_net, q_net)
    target_net.eval()

    print(f"[Model] Trainable params: {count_parameters(q_net):,}")
    with open(out_dir / "model_summary.txt", "w") as f:
        f.write(str(q_net) + "\n")
        f.write(f"Trainable params: {count_parameters(q_net):,}\n")

    eps_cfg = EpsilonGreedyConfig(
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
    )
    agent = DQNAgent(q_net, action_dim=2, eps_cfg=eps_cfg)
    buffer = ReplayBuffer(capacity=args.replay_capacity, state_dim=state_dim)

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    header = ["step", "episode", "ep_return", "moving_return", "loss", "epsilon"]
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    returns_window = deque(maxlen=50)
    global_step = 0
    best_mavg = -float("inf")

    for ep in range(1, args.episodes + 1):
        obs_out = env.reset()
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.act(obs, global_step=global_step, device=str(device))

            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_out

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]

            obs_np = np.array(obs, dtype=np.float32).flatten()
            next_obs_np = np.array(next_obs, dtype=np.float32).flatten()

            buffer.push(Transition(
                state=obs_np,
                action=int(action),
                reward=float(reward),
                next_state=next_obs_np,
                done=bool(done)
            ))

            obs = next_obs
            ep_ret += reward
            global_step += 1

            if len(buffer) >= args.batch_size and global_step > args.learn_starts and (global_step % args.train_every == 0):
                loss_val = optimize_step(
                    q_net, target_net, buffer, optimizer, loss_fn,
                    args.batch_size, args.gamma, device, args.max_grad_norm
                )
            else:
                loss_val = None

            if global_step % args.target_update_freq == 0:
                hard_update(target_net, q_net)

        returns_window.append(ep_ret)
        moving = np.mean(returns_window)
        eps_now = agent.epsilon(global_step)

        if ep % args.eval_every == 0:
            try:
                eval_ret = evaluate_policy(env, agent, device, n_episodes=3)
                print(f"[Eval] episode={ep} avg_return={eval_ret:.2f}")
            except Exception as e:
                print(f"[Eval skipped] {e}")

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                global_step, ep, f"{ep_ret:.4f}", f"{moving:.4f}",
                (f"{loss_val:.6f}" if loss_val is not None else ""), f"{eps_now:.4f}"
            ])


        if moving > best_mavg:
            best_mavg = moving
            torch.save(q_net.state_dict(), ckpt_dir / "best.pt")

        if ep % args.ckpt_every == 0:
            torch.save(q_net.state_dict(), ckpt_dir / f"ep{ep:05d}.pt")

        print(f"[Ep {ep:04d}] return={ep_ret:.2f}  moving50={moving:.2f}  eps={eps_now:.3f}"
              + (f"  loss={loss_val:.4f}" if loss_val is not None else ""))


    torch.save(q_net.state_dict(), ckpt_dir / "final.pt")
    print(f"Training done. Metrics -> {metrics_csv}  Checkpoints -> {ckpt_dir}")


def optimize_step(q_net, target_net, buffer, optimizer, loss_fn, batch_size, gamma, device, max_grad_norm):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

    q_values = q_net(states_t).gather(1, actions_t)
    with torch.no_grad():
        next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
        target = rewards_t + gamma * (1.0 - dones_t) * next_q

    loss = loss_fn(q_values, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm > 0:
        nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.item())


def parse_args():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--h1", type=int, default=128)
    p.add_argument("--h2", type=int, default=128)
    p.add_argument("--h3", type=int, default=64)
    # rl
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--replay_capacity", type=int, default=100_000)
    p.add_argument("--learn_starts", type=int, default=1_000)
    p.add_argument("--train_every", type=int, default=4)
    p.add_argument("--target_update_freq", type=int, default=1_000)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    # eps
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay_steps", type=int, default=50_000)
    # run
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="runs/dqn_session2")
    p.add_argument("--ckpt_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
