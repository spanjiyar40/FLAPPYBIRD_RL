import torch
import numpy as np
import argparse

from agents.dqn import DQN, DQNAgent
from envs.flappy_env import make_env   # <-- USE YOUR SAME ENV WRAPPER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    # Same env as training (render enabled)
    env = make_env(render=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"[PLAY] state_dim={state_dim}, action_dim={action_dim}")

    device = torch.device("cpu")

    # Load Q-network
    q_net = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    q_net.eval()

    agent = DQNAgent(q_net=q_net, action_dim=action_dim, eps_cfg=None)

    for ep in range(1, args.episodes + 1):
        print(f"\n[PLAY] Episode {ep} starting...")
        state, _ = env.reset()

        done = False
        total_r = 0
        steps = 0

        while not done:
            env.render()

            # GREEDY action (no epsilon)
            q_values = q_net(torch.tensor(state, dtype=torch.float32)).detach().numpy()
            action = int(np.argmax(q_values))

            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            total_r += reward
            steps += 1

        print(f"[GAME OVER] Episode {ep}  Score: {total_r:.2f}  Steps: {steps}")

    env.close()


if __name__ == "__main__":
    main()
