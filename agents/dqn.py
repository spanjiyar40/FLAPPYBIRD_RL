from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQN(nn.Module):
    """
    Simple MLP for Q(s, a).
    Input: state vector [y, vel, pipe_dist] -> R^3
    Output: Q-values for each discrete action (e.g., 2: [no-flap, flap])
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
    

@dataclass
class EpsilonGreedyConfig:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000 


class DQNAgent:
    """
    Wraps a DQN with ε-greedy policy.
    Use .act(state, global_step) to pick an action.
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
        """
        state: list/np.array/torch.Tensor shape (3,)
        returns action int in [0, action_dim-1]
        """
        eps = self.epsilon(global_step)
        if random.random() < eps:
            return random.randrange(self.action_dim)
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        q = self.q_net(state.unsqueeze(0)) 
        return int(torch.argmax(q, dim=1).item())

def hard_update(target: nn.Module, source: nn.Module):
    """Copy weights from source -> target (in-place)."""
    target.load_state_dict(source.state_dict())

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

