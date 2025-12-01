from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


# ------------------------------------------------------------------------------------
# Q-Network (unchanged)
# ------------------------------------------------------------------------------------
class DQN(nn.Module):
    """
    Simple MLP for Q(s, a).
    """
    def __init__(self, state_dim: int = 3, action_dim: int = 2,
                 hidden_sizes: Tuple[int, int, int] = (128, 128, 64)):
        super().__init__()
        h1, h2, h3 = hidden_sizes
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


# ------------------------------------------------------------------------------------
# Epsilon-greedy schedule
# ------------------------------------------------------------------------------------
@dataclass
class EpsilonGreedyConfig:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

class DQNAgent:
    """
    Handles epsilon-greedy actions.
    """
    def __init__(self, q_net: DQN, action_dim: int, eps_cfg: EpsilonGreedyConfig):
        self.q_net = q_net
        self.action_dim = action_dim
        self.eps_cfg = eps_cfg

    def epsilon(self, global_step: int) -> float:
        frac = max(0.0, 1.0 - (global_step / max(1, self.eps_cfg.eps_decay_steps)))
        return self.eps_cfg.eps_end + (self.eps_cfg.eps_start - self.eps_cfg.eps_end) * frac

    @torch.no_grad()
    def act(self, state, global_step: int, device: str = "cpu") -> int:
        eps = self.epsilon(global_step)
        if random.random() < eps:
            return random.randrange(self.action_dim)
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        q = self.q_net(state.unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())


# ------------------------------------------------------------------------------------
# Hard update
# ------------------------------------------------------------------------------------
def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


# ------------------------------------------------------------------------------------
# Count params
# ------------------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------------
# Transition + ReplayBuffer (same as your old file)
# ------------------------------------------------------------------------------------
class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


# ------------------------------------------------------------------------------------
# DOUBLE DQN optimize step  (MAIN UPDATE YOU NEEDED)
# ------------------------------------------------------------------------------------
def optimize_step(
    q_net, target_net, buffer, optimizer, loss_fn,
    batch_size, gamma, device, max_grad_norm
):
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    # Current Q-values
    q_values = q_net(states_t).gather(1, actions_t)

    # ----------- DOUBLE DQN -----------
    with torch.no_grad():
        # Step 1: online net selects next action
        next_actions = q_net(next_states_t).argmax(dim=1, keepdim=True)

        # Step 2: target net evaluates chosen action
        next_q_target = target_net(next_states_t).gather(1, next_actions)

        target_q_values = rewards_t + gamma * (1 - dones_t) * next_q_target
    # -----------------------------------

    loss = loss_fn(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()

    if max_grad_norm > 0:
        nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)

    optimizer.step()

    return float(loss.item())
