import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from model import DQN


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 4,
        device: str = 'cpu',
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        warmup: int = 1_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 100_000,
        target_update_freq: int = 1_000,
        max_grad_norm: float = 10.0,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm

        self.online_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(buffer_size)
        self.total_steps: int = 0
        self._update_count: int = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_net(t).argmax(dim=1).item())

    # ── Learning ──────────────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        frac = min(self.total_steps / self.eps_decay_steps, 1.0)
        self.eps = self.eps_start + frac * (self.eps_end - self.eps_start)

    def learn(self) -> float | None:
        if len(self.buffer) < self.warmup:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states     = torch.tensor(states,     device=self.device)
        actions    = torch.tensor(actions,    device=self.device)
        rewards    = torch.tensor(rewards,    device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones      = torch.tensor(dones,      device=self.device)

        # Double DQN: online net selects action, target net evaluates Q
        with torch.no_grad():
            next_acts = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q    = self.target_net(next_states).gather(1, next_acts).squeeze(1)
            target_q  = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str):
        self.online_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
