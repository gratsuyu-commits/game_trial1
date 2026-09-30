"""
Evaluate a trained DQN agent with ε=0 (greedy).

Usage:
    python rl/evaluate.py                            # uses best_model.pth, 20 episodes
    python rl/evaluate.py --no-render                # results only, no pygame window
    python rl/evaluate.py --model rl/models/checkpoint_100000.pth --episodes 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from env import CrabAdventureEnv
from agent import DQNAgent


def evaluate(args):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    render_mode = None if args.no_render else 'human'
    env   = CrabAdventureEnv(render_mode=render_mode)
    agent = DQNAgent(device=device)
    agent.load(args.model)
    agent.eps = 0.0     # greedy — no random actions

    cleared_count = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        env.hud_info = {
            'episode':     ep,
            'total_steps': 0,
            'ep_reward':   0.0,
            'eps':         0.0,
        }
        ep_reward = 0.0
        done      = False
        info      = {}

        while not done:
            if not args.no_render:
                env.hud_info['ep_reward'] = ep_reward
                env.render()

            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        if info.get('cleared'):
            cleared_count += 1

        status = 'CLEAR' if info.get('cleared') else 'fail '
        print(f"EP {ep+1:3d} [{status}]  reward={ep_reward:.2f}")

    print(f"\nClear rate: {cleared_count}/{args.episodes} "
          f"= {cleared_count/args.episodes:.1%}")
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument(
        '--model', default='rl/models/best_model.pth', help='Path to model weights'
    )
    parser.add_argument(
        '--episodes', type=int, default=20, help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--no-render', action='store_true', help='Disable pygame rendering'
    )
    evaluate(parser.parse_args())
