# agents/dqn.py
from dataclasses import dataclass
from typing import Tuple
import random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden: Tuple[int, int] = (256, 256)
    gamma: float = 0.99
    lr: float = 1e-3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000
    buffer_size: int = 200_000
    batch_size: int = 256
    target_tau: float = 0.005  # soft update rate
    device: str = "cpu"
    seed: int = 2025

class QNet(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=(256,256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], n_actions),
        )
    def forward(self, x): return self.net(x)

class Replay:
    def __init__(self, cap, obs_dim, seed=0):
        self.cap = cap
        self.obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.next = np.zeros((cap, obs_dim), dtype=np.float32)
        self.act = np.zeros((cap,), dtype=np.int64)
        self.rew = np.zeros((cap,), dtype=np.float32)
        self.done = np.zeros((cap,), dtype=np.float32)
        self.idx = 0; self.full = False
        random.seed(seed); np.random.seed(seed)
    def add(self, s, a, r, s2, d):
        i = self.idx
        self.obs[i] = s; self.act[i] = a; self.rew[i] = r
        self.next[i] = s2; self.done[i] = float(d)
        self.idx = (i + 1) % self.cap
        self.full = self.full or self.idx == 0
    def size(self): return self.cap if self.full else self.idx
    def sample(self, bs):
        n = self.size()
        idxs = np.random.randint(0, n, size=bs)
        return (self.obs[idxs], self.act[idxs], self.rew[idxs],
                self.next[idxs], self.done[idxs])

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
        self.cfg = cfg
        self.q = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(cfg.device)
        self.t = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(cfg.device)
        self.t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = Replay(cfg.buffer_size, cfg.obs_dim, seed=cfg.seed)
        self.step = 0

    def epsilon(self):
        f = min(1.0, self.step / max(1, self.cfg.eps_decay_steps))
        return float(self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * f)

    @torch.no_grad()
    def act(self, s_np):
        self.step += 1
        if random.random() < self.epsilon():
            return random.randrange(self.cfg.n_actions)
        s = torch.from_numpy(s_np).to(self.cfg.device).unsqueeze(0)  # [1, obs]
        q = self.q(s)  # [1, nA]
        return int(q.argmax(dim=1).item())

    def push(self, s, a, r, s2, d): self.replay.add(s, a, r, s2, d)

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.t.parameters()):
                tp.data.mul_(1.0 - self.cfg.target_tau).add_(self.cfg.target_tau * p.data)

    def train_step(self):
        if self.replay.size() < self.cfg.batch_size: return 0.0
        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)
        dev = self.cfg.device
        s = torch.from_numpy(s).to(dev)
        a = torch.from_numpy(a).to(dev)
        r = torch.from_numpy(r).to(dev)
        s2= torch.from_numpy(s2).to(dev)
        d = torch.from_numpy(d).to(dev)

        qsa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
        with torch.no_grad():
            # Double DQN selection
            a2 = self.q(s2).argmax(dim=1, keepdim=True)
            q_next = self.t(s2).gather(1, a2).squeeze(1)
            target = r + (1.0 - d) * self.cfg.gamma * q_next
        loss = torch.nn.functional.smooth_l1_loss(qsa, target)

        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())

    def save(self, path): torch.save(self.q.state_dict(), path)
    def load(self, path):  self.q.load_state_dict(torch.load(path, map_location=self.cfg.device))
