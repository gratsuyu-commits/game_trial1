"""
Visualise training progress from rl/logs/train_log.csv.

Usage:
    python rl/plot_training.py
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_PATH = Path(__file__).parent / 'logs' / 'train_log.csv'

if not LOG_PATH.exists():
    print(f"ERROR: {LOG_PATH} not found. Run training first.")
    raise SystemExit(1)

# ── Load ──────────────────────────────────────────────────────────────────────
episodes, steps, rewards, cleared, epsilons = [], [], [], [], []

with open(LOG_PATH, newline='') as f:
    for row in csv.DictReader(f):
        episodes.append(int(row['episode']))
        steps.append(int(row['total_steps']))
        rewards.append(float(row['episode_reward']))
        cleared.append(int(row['cleared']))
        epsilons.append(float(row['epsilon']))

# ── Rolling average ───────────────────────────────────────────────────────────
def rolling(data, window=100):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i+1]) / (i - lo + 1))
    return out

clear_rate  = [v * 100 for v in rolling(cleared,  100)]
reward_avg  = rolling(rewards, 100)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Crab Adventure — DQN Training Curve', fontsize=13, fontweight='bold')

# Clear rate
axes[0].plot(episodes, clear_rate, color='#2196F3', linewidth=1.5)
axes[0].axhline(50, color='gray', linestyle='--', linewidth=0.8, label='50%')
axes[0].set_ylabel('Clear Rate % (100-ep avg)')
axes[0].set_ylim(0, 105)
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
axes[0].legend(loc='upper left', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Episode reward
axes[1].plot(episodes, reward_avg, color='#4CAF50', linewidth=1.5)
axes[1].set_ylabel('Episode Reward (100-ep avg)')
axes[1].grid(True, alpha=0.3)

# Epsilon
axes[2].plot(episodes, epsilons, color='#FF5722', linewidth=1.2)
axes[2].set_ylabel('Epsilon (ε)')
axes[2].set_xlabel('Episode')
axes[2].set_ylim(0, 1.05)
axes[2].grid(True, alpha=0.3)

# Step count on top x-axis
ax_top = axes[0].twiny()
ax_top.set_xlim(axes[0].get_xlim())
step_ticks = list(range(0, max(steps) + 1, max(steps) // 5))
ep_at_step = []
for s in step_ticks:
    idx = next((i for i, st in enumerate(steps) if st >= s), len(episodes) - 1)
    ep_at_step.append(episodes[idx])
ax_top.set_xticks(ep_at_step)
ax_top.set_xticklabels([f'{s//1000}k' for s in step_ticks], fontsize=8)
ax_top.set_xlabel('Total Steps', fontsize=9)

plt.tight_layout()
out_path = Path(__file__).parent / 'logs' / 'training_curve.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()
