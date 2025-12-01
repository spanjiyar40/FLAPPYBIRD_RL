# ---------------------------------------------------------
# train_dqn.py (FINAL WORKING VERSION FOR 4D STATE)
# ---------------------------------------------------------

import os
import csv
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn import (
    DQN,
    DQNAgent,
    EpsilonGreedyConfig,
    hard_update,
    count_parameters,
)

from agents.replay_buffer import ReplayBuffer


# -----------------------------
# Device
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Double DQN optimize step
# -----------------------------
def optimize_step(q_net, target_net, buffer, optimizer, loss_fn,
                  batch_size, gamma, device, max_grad_norm):

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    # Q(s,a)
    q_values = q_net(states).gather(1, actions)

    # Double DQN target
    with torch.no_grad():
        best_act = q_net(next_states).argmax(1, keepdim=True)
        target_q = target_net(next_states).gather(1, best_act)
        target = rewards + gamma * (1 - dones) * target_q

    loss = loss_fn(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.item())


# -----------------------------
# Evaluate policy
# -----------------------------
@torch.no_grad()
def evaluate_policy(env, agent, device, n_episodes=3):
    scores = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total = 0
        while not done:
            action = agent.act(state, global_step=10**9, device=device)
            state, reward, done, _, _ = env.step(action)
            total += reward
        scores.append(total)
    return float(np.mean(scores))


# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--min_buffer", type=int, default=1000)
    parser.add_argument("--target_update", type=int, default=1000)
    parser.add_argument("--eps_decay_steps", type=int, default=100000)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"[Device] {device}")

    # Import the env after seed
    from envs.flappy_env import make_env
    env = make_env(render=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"[Env] state_dim={state_dim}, action_dim={action_dim}")

    # Networks
    q_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    hard_update(target_net, q_net)

    print(f"[Model] Trainable params: {count_parameters(q_net):,}")

    # Agent + Buffer
    agent = DQNAgent(q_net, action_dim, EpsilonGreedyConfig(args.eps_decay_steps))
    buffer = ReplayBuffer(args.buffer_size)

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    # Output files
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "moving50", "epsilon", "loss"])

    global_step = 0
    episode = 0
    recent_returns = []

    # ---------------------------
    # DEBUG: Print first state
    # ---------------------------
    s, _ = env.reset()
    print("[DEBUG] example state:", s)

    print("\n>>> TRAINING STARTED...\n")

    # ---------------------------
    # TRAIN LOOP
    # ---------------------------
    while global_step < args.max_steps:
        state, _ = env.reset()
        done = False
        total_reward = 0
        loss_val = None

        while not done and global_step < args.max_steps:
            action = agent.act(state, global_step, device)
            next_state, reward, done, _, _ = env.step(action)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= args.min_buffer:
                loss_val = optimize_step(
                    q_net, target_net, buffer, optimizer, loss_fn,
                    args.batch_size, args.gamma, device, args.max_grad_norm
                )

            if global_step % args.target_update == 0:
                hard_update(target_net, q_net)

            global_step += 1

        episode += 1
        recent_returns.append(total_reward)
        moving = np.mean(recent_returns[-50:])
        eps_now = agent.epsilon(global_step)

        print(f"[Ep {episode:04d}] return={total_reward:.1f}  moving50={moving:.2f}  eps={eps_now:.3f}"
              + (f"  loss={loss_val:.3f}" if loss_val else ""))

        # Save CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, total_reward, moving, eps_now,
                loss_val if loss_val is not None else ""
            ])

        # Evaluate
        if episode % args.eval_every == 0:
            avg_r = evaluate_policy(env, agent, device)
            print(f"[Eval] episode={episode}  avg_return={avg_r:.2f}")

        # Save model
        if episode % 200 == 0:
            ckpt = out_dir / "checkpoints"
            ckpt.mkdir(exist_ok=True)
            torch.save(q_net.state_dict(), ckpt / f"ep{episode}.pth")


if __name__ == "__main__":
    main()
