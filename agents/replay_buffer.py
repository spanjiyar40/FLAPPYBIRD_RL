from typing import NamedTuple, Tuple
import numpy as np
import random

class Transition(NamedTuple):
    state: np.ndarray     
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    """
    Fixed-size FIFO buffer for (s, a, r, s', done).
    Uses numpy for speed; returns batches as numpy arrays.
    """
    def __init__(self, capacity: int = 100_000, state_dim: int = 3):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, transition: Transition):
        idx = self.ptr
        self.states[idx] = transition.state
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.next_states[idx] = transition.next_state
        self.dones[idx] = transition.done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        assert self.size >= batch_size, "Not enough samples to draw a batch."
        idxs = random.sample(range(self.size), batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size
