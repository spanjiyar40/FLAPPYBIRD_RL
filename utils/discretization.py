import numpy as np
import random
from collections import defaultdict

VEL_MIN, VEL_MAX = -8, 8
VEL_BINS = VEL_MAX - VEL_MIN + 1     
Y_BINS = 20                         
DIST_BINS = 10                       
N_ACTIONS = 2                        

# Observation ranges (approximate defaults — adjust if needed)
Y_MIN, Y_MAX = 0.0, 512.0
DIST_MIN, DIST_MAX = 0.0, 600.0

def make_bins(n_bins, low, high):
    """Return evenly spaced cutpoints for np.digitize."""
    if n_bins <= 1:
        return np.array([])
    return np.linspace(low, high, n_bins + 1)[1:-1]

# Precompute bin cutpoints
y_bins = make_bins(Y_BINS, Y_MIN, Y_MAX)
dist_bins = make_bins(DIST_BINS, DIST_MIN, DIST_MAX)


def discretize(obs):
    """
    Convert an observation (dict or array-like) into a discrete state tuple.
    Returns (vel_idx, y_idx, dist_idx).
    """

    if isinstance(obs, dict):
        vel_raw = obs.get('vel', obs.get('velocity', 0.0))
        y_raw = obs.get('y', obs.get('player_y', 0.0))
        dist_raw = obs.get('pipe_dist', obs.get('pipe_dx', 0.0))
    else:
        arr = np.asarray(obs)
        y_raw, vel_raw, dist_raw = float(arr[0]), float(arr[1]), float(arr[2])

    # velocity: round + clip + shift to index
    vel_idx = int(np.clip(round(vel_raw), VEL_MIN, VEL_MAX)) - VEL_MIN

    # y position and distance via binning
    y_idx = int(np.digitize(x=y_raw, bins=y_bins))
    dist_idx = int(np.digitize(x=dist_raw, bins=dist_bins))

    # ensure safe bounds
    vel_idx = int(np.clip(vel_idx, 0, VEL_BINS - 1))
    y_idx = int(np.clip(y_idx, 0, Y_BINS - 1))
    dist_idx = int(np.clip(dist_idx, 0, DIST_BINS - 1))

    return (vel_idx, y_idx, dist_idx)


Q = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))


def choose_action(state, epsilon):
    """
    ε-greedy policy: random action with prob ε, else greedy.
    Returns (action, is_exploratory_flag).
    """
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS), True

    qvals = Q[state]
    max_idxs = np.flatnonzero(qvals == qvals.max())
    return int(np.random.choice(max_idxs)), False

# Optional: random seed helper
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

# Quick test (manual)
if __name__ == "__main__":
    print("Testing discretization...")
    samples = [
        {'y': 250.0, 'vel': -3.2, 'pipe_dist': 120.0},
        {'y': 10.0, 'vel': 5.7, 'pipe_dist': 400.0},
        np.array([200.0, 0.0, 50.0]),
    ]
    for s in samples:
        st = discretize(s)
        print("obs:", s, "-> state:", st)
        a, ex = choose_action(st, epsilon=0.5)
        print("  chosen action:", a, "exploratory?", ex)
