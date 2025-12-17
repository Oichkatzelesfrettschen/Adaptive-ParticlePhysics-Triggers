# rl/dqn_agent.py
"""
Minimal DQN module (PyTorch) for threshold-control tasks.

Exports:
  - DQNAgent
  - make_obs(...)
  - shield_delta(...)
  - compute_reward(...)

No domain-specific code here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import random
import numpy as np

# --- torch import guarded so main script can error nicely if missing ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required.\nInstall: pip install torch\n\n"
        f"Import error: {e}"
    )

# ------------------------ replay buffer ------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = int(capacity)
        self.data = []
        self.i = 0

    def push(self, s, a, r, sp, done: bool):
        item = (
            np.asarray(s, np.float32),
            int(a),
            float(r),
            np.asarray(sp, np.float32),
            float(done),
        )
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.i] = item
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size: int = 128):
        batch = random.sample(self.data, batch_size)
        s, a, r, sp, done = zip(*batch)
        return (
            np.stack(s),
            np.asarray(a, np.int64),
            np.asarray(r, np.float32),
            np.stack(sp),
            np.asarray(done, np.float32),
        )

    def __len__(self):
        return len(self.data)

# ------------------------ networks ------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------ agent ------------------------
@dataclass
class DQNConfig:
    lr: float = 5e-4
    gamma: float = 0.95
    batch_size: int = 128
    target_update: int = 200
    buffer_capacity: int = 50_000
    grad_clip: float = 5.0

class DQNAgent:
    """
    Vanilla Double-DQN with:
      - SmoothL1 (Huber)
      - target network
      - replay buffer
      - epsilon-greedy action selection
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 0,
        device: Optional[str] = None,
        cfg: Optional[DQNConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg or DQNConfig()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.q = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.tgt = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buf = ReplayBuffer(capacity=self.cfg.buffer_capacity)

        self.train_steps = 0

    def act(self, obs: np.ndarray, eps: float = 0.1) -> int:
        """Epsilon-greedy."""
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(x)[0]
            return int(torch.argmax(qvals).item())

    def train_step(self) -> Optional[float]:
        """One gradient step. Returns loss or None if not enough data."""
        bs = self.cfg.batch_size
        if len(self.buf) < bs:
            return None

        s, a, r, sp, done = self.buf.sample(bs)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        sp = torch.tensor(sp, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        # Double DQN target
        with torch.no_grad():
            a_star = torch.argmax(self.q(sp), dim=1, keepdim=True)
            q_sp = self.tgt(sp).gather(1, a_star)
            y = r + (1.0 - done) * self.cfg.gamma * q_sp

        loss = nn.SmoothL1Loss()(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

# ------------------------ observation + reward helpers ------------------------
def make_obs(
    bg_rate: float,
    prev_bg_rate: float,
    cut: float,
    cut_mid: float,
    cut_span: float,
    target: float,
) -> np.ndarray:
    """
    Default 3D observation used in your scripts:
      [ normalized_error, normalized_delta_error, normalized_cut ]
    """
    cut_span = max(1e-12, float(cut_span))
    target = max(1e-12, float(target))
    x_rate = (float(bg_rate) - target) / target
    x_drate = (float(bg_rate) - float(prev_bg_rate)) / target
    x_cut = (float(cut) - float(cut_mid)) / cut_span
    return np.array([x_rate, x_drate, x_cut], dtype=np.float32)

def shield_delta(
    bg_rate: float,
    target: float,
    tol: float,
    max_delta: float,
) -> Optional[float]:
    """
    If you're far from target, force a strong move in the correct direction.
      - bg too high => increase cut (positive delta)
      - bg too low  => decrease cut (negative delta)
    """
    if bg_rate > target + tol:
        return +float(max_delta)
    if bg_rate < target - tol:
        return -float(max_delta)
    return None

def compute_reward(
    bg_rate: float,
    target: float,
    tol: float,
    sig_rate_1: float,
    sig_rate_2: float,
    delta_applied: float,
    max_delta: float,
    alpha: float = 0.2,
    beta: float = 0.02,
    clip: Tuple[float, float] = (-10.0, 10.0),
) -> float:
    """
    sig_rate_1: first signal rate (e.g. TTbar)
    sig_rate_2: second signal rate (e.g. HToAATo4B)

    Generic reward:
      + in-band tracking bonus (encourages holding)
      - out-of-band penalty grows smoothly
    #   - background penalty: |bg-target|/tol
      + signal bonus: alpha * mean(sig)/100
      - movement penalty: beta * |delta|/max_delta
    """
    tol = max(1e-12, float(tol))
    max_delta = max(1e-12, float(max_delta))

    # normalized error
    e = (float(bg_rate) - float(target)) / tol
    ae = abs(e)

    # Tracking: reward being within tolerance, penalize being outside
    if ae <= 1.0:
        # max +1 at center; smoothly decreases to 0 at band edge
        track = 1.0 - ae**2
    else:
        # linear penalty outside band, continuous at ae=1
        track = - (ae - 1.0)


    bg_pen = abs(float(bg_rate) - float(target)) / tol
    sig_term = 0.5 * (float(sig_rate_1) + float(sig_rate_2)) / 100.0

    move_pen = abs(float(delta_applied)) / max_delta

    # r = -bg_pen + alpha * sig_term - beta * move_pen
    r = track + alpha * sig_term - beta * move_pen
    lo, hi = clip
    return float(np.clip(r, lo, hi))
