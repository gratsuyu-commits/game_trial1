"""
Train a Double-DQN agent to clear Crab Adventure.

Usage:
    python rl/train.py                 # with pygame visualisation every 50 episodes
    python rl/train.py --no-render     # headless, faster training
    python rl/train.py --seed 123      # custom random seed
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from env import CrabAdventureEnv
from agent import DQNAgent

_RL_DIR    = Path(__file__).parent
_MODEL_DIR = _RL_DIR / 'models'
_LOG_DIR   = _RL_DIR / 'logs'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  seed: {args.seed}  |  render: {not args.no_render}")

    render_mode = None if args.no_render else 'human'
    env   = CrabAdventureEnv(render_mode=render_mode)
    agent = DQNAgent(device=device)

    log_path = _LOG_DIR / 'train_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['episode', 'total_steps', 'episode_reward',
             'episode_length', 'cleared', 'epsilon']
        )

    best_clear_rate    = 0.0
    recent_clears: list[int] = []
    last_ckpt_step     = 0
    episode            = 0

    while agent.total_steps < 5_000_000:
        should_render = (not args.no_render) and (episode % 50 == 0)
        obs, _ = env.reset()

        ep_reward = 0.0
        ep_len    = 0
        cleared   = False

        if should_render:
            env.hud_info = {
                'episode':     episode,
                'total_steps': agent.total_steps,
                'ep_reward':   0.0,
                'eps':         agent.eps,
            }

        while True:
            if should_render:
                env.hud_info['ep_reward'] = ep_reward
                env.hud_info['eps']       = agent.eps
                env.render()

            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.push(obs, action, reward, next_obs, float(done))
            agent.learn()

            obs       = next_obs
            ep_reward += reward
            ep_len    += 1
            if info.get('cleared'):
                cleared = True
            if done:
                break

        # Track clear rate over last 100 episodes
        recent_clears.append(1 if cleared else 0)
        if len(recent_clears) > 100:
            recent_clears.pop(0)
        clear_rate = sum(recent_clears) / len(recent_clears)

        # Save best model when clear rate improves (need at least 100 episodes)
        if len(recent_clears) == 100 and clear_rate > best_clear_rate:
            best_clear_rate = clear_rate
            agent.save(str(_MODEL_DIR / 'best_model.pth'))
            print(f"  ★ New best clear rate {best_clear_rate:.1%}  (ep {episode})")

        # Periodic checkpoint every 50 000 agent steps
        step = agent.total_steps
        if step - last_ckpt_step >= 50_000:
            agent.save(str(_MODEL_DIR / f'checkpoint_{step}.pth'))
            last_ckpt_step = step

        # CSV log
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                episode, step,
                f'{ep_reward:.2f}', ep_len,
                int(cleared), f'{agent.eps:.4f}',
            ])

        if episode % 10 == 0:
            print(
                f"[EP {episode:5d}] steps={step:7d}  "
                f"reward={ep_reward:8.2f}  len={ep_len:5d}  "
                f"cleared={int(cleared)}  ε={agent.eps:.3f}  "
                f"rate={clear_rate:.1%}"
            )

        episode += 1

    env.close()
    print(f"\nTraining complete. Best clear rate: {best_clear_rate:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN agent for Crab Adventure')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable pygame visualisation (faster training)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train(parser.parse_args())
