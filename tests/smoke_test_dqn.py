import torch
import numpy as np
from agents.dqn import DQN, DQNAgent, EpsilonGreedyConfig
from agents.replay_buffer import ReplayBuffer, Transition

def test_forward_pass():
    net = DQN(state_dim=3, action_dim=2)
    x = torch.randn(5, 3)  
    q = net(x)
    assert q.shape == (5, 2)
    print("Forward pass OK:", q.shape)

def test_agent_action():
    net = DQN(state_dim=3, action_dim=2)
    agent = DQNAgent(net, action_dim=2, eps_cfg=EpsilonGreedyConfig())
    action = agent.act([0.1, -0.2, 0.5], global_step=0)
    assert action in [0, 1]
    print("Agent action OK:", action)

def test_replay_buffer():
    buf = ReplayBuffer(capacity=100, state_dim=3)
    for _ in range(32):
        s = np.random.randn(3).astype(np.float32)
        a = np.random.randint(0, 2)
        r = float(np.random.randn())
        ns = np.random.randn(3).astype(np.float32)
        d = bool(np.random.rand() < 0.1)
        buf.push(Transition(s, a, r, ns, d))

    assert len(buf) == 32
    batch = buf.sample(16)
    states, actions, rewards, next_states, dones = batch
    assert states.shape == (16, 3)
    assert actions.shape == (16,)
    print("Replay buffer OK: batch shapes", states.shape, actions.shape)

if __name__ == "__main__":
    test_forward_pass()
    test_agent_action()
    test_replay_buffer()
    print("All smoke tests passed.")
