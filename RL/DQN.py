# DQN using background samples SingleTrigger
#!/usr/bin/env python3
"""
demo_single_trigger_dqn_ht.py

Minimal DQN test for *single HT trigger* on Trigger_food_MC.h5 workflow.

- One RL step = one data batch (chunk_size events)
- Agent action = discrete nudge to Ht_cut for the *next* batch
- State = [rate_error, rate_trend, normalized_cut]
- Reward = keep bg rate near target (0.25%) + small bonus for signal efficiency
- Compares: Constant Menu vs PD Controller vs DQN

Run:
  python demo_single_trigger_dqn_ht.py

Requires:
  pip install torch h5py hdf5plugin matplotlib numpy
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # noqa: F401
import argparse
from pathlib import Path
# ------------------------- output -------------------------
OUTDIR = "outputs/demo_sing_dqn"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------- reproducibility -------------------------
SEED = 20251213
random.seed(SEED)
np.random.seed(SEED)

# ------------------------- your helpers -------------------------
def PD_controller1(r_, pre_, cut_):
    Kp = 30
    Kd = 5
    target = 0.25  # (%)
    error = r_ - target
    delta = error - pre_
    newcut_ = cut_ + Kp * error + Kd * delta
    return newcut_, error

def Sing_Trigger(x_, cut_):
    """Returns acceptance in percent (%)."""
    num_ = x_.shape[0]
    accepted_ = np.sum(x_ >= cut_)
    r_ = 100.0 * accepted_ / max(1, num_)
    return float(r_)

def read_data(h5_file_path, score_dim=2):
    """
    V2 as we only save dim=2 for both MC and RealData.
    Supports:
      - new Trigger_food_v2: mc_*_score02 (or score{dim:02d})
      - legacy Trigger_food: mc_*_score01/04
    """
    score_key = f"score{int(score_dim):02d}"  # e.g. score02

    with h5py.File(h5_file_path, "r") as h5:
        keys = set(h5.keys())

        # --- HT/NPV always needed for HT-only trigger ---
        Bht_tot = h5["mc_bkg_ht"][:]
        B_npvs  = h5["mc_bkg_Npv"][:]

        Sht_tot1 = h5["mc_tt_ht"][:]
        S_npvs1  = h5["tt_Npv"][:]

        Sht_tot2 = h5["mc_aa_ht"][:]
        S_npvs2  = h5["aa_Npv"][:]

        # --- Scores: optional (keep for future AS triggers) ---
        def _read_score(prefix):
            # new
            k_new = f"{prefix}_{score_key}"
            if k_new in keys:
                return h5[k_new][:]

            # legacy fallbacks
            k01 = f"{prefix}_score01"
            k04 = f"{prefix}_score04"
            if k01 in keys:
                return h5[k01][:]
            if k04 in keys:
                return h5[k04][:]

            return None

        Bas_tot  = _read_score("mc_bkg")
        Stt_tot  = _read_score("mc_tt")
        Saa_tot  = _read_score("mc_aa")

    return {
        "Bht": Bht_tot, "Bnpv": B_npvs,
        "Tht": Sht_tot1, "Tnpv": S_npvs1,
        "Aht": Sht_tot2, "Anpv": S_npvs2,
        "Bas": Bas_tot,  "Stt_as": Stt_tot, "Saa_as": Saa_tot,
    }


# ------------------------- minimal DQN (PyTorch) -------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise SystemExit(
        "PyTorch is required for this script.\n"
        "Install: pip install torch\n\n"
        f"Import error: {e}"
    )

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.capacity = int(capacity)
        self.data = []
        self.i = 0

    def push(self, s, a, r, sp, done):
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

    def sample(self, batch_size=128):
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

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.95, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(obs_dim, n_actions).to(self.device)
        self.tgt = QNet(obs_dim, n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = ReplayBuffer()
        self.gamma = float(gamma)
        self.n_actions = int(n_actions)
        self.train_steps = 0

    def act(self, obs, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(x)[0]
            return int(torch.argmax(qvals).item())

    def train_step(self, batch_size=128, target_update=200):
        if len(self.buf) < batch_size:
            return None

        s, a, r, sp, done = self.buf.sample(batch_size)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        sp = torch.tensor(sp, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        with torch.no_grad():
            max_q_sp = self.tgt(sp).max(dim=1, keepdim=True)[0]
            y = r + (1.0 - done) * self.gamma * max_q_sp

        loss = nn.SmoothL1Loss()(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())
    
def shield_delta(bg_rate, target, tol, max_delta):
    # if bg is too high, raise cut; if too low, lower cut
    if bg_rate > target + tol:
        return +max_delta
    if bg_rate < target - tol:
        return -max_delta
    return None

def make_obs(bg_rate, prev_bg_rate, ht_cut, cut_mid, cut_span, target):
    # normalize rate error/trend by target (dimensionless)
    x_rate = (bg_rate - target) / max(1e-6, target)
    x_drate = (bg_rate - prev_bg_rate) / max(1e-6, target)
    x_cut = (ht_cut - cut_mid) / max(1e-6, cut_span)
    return np.array([x_rate, x_drate, x_cut], dtype=np.float32)

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"])
    ap.add_argument("--trigger-food", default=None, help="Path to Trigger_food_*.h5")
    ap.add_argument("--score-dim", type=int, default=2, help="AE score dim used in file, e.g. 2 -> score02")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # pick default file based on control
    if args.trigger_food is None:
        path = "Data/Trigger_food_Data.h5" if args.control == "RealData" else "Data/Trigger_food_MC.h5"
    else:
        path = args.trigger_food

    tag = "realdata" if args.control == "RealData" else "mc"
    outdir = args.outdir or f"outputs/demo_sing_dqn_{tag}"
    os.makedirs(outdir, exist_ok=True)

    data = read_data(path, score_dim=args.score_dim)

    Bht_tot = data["Bht"]
    B_npvs  = data["Bnpv"]
    Sht_tot1, S_npvs1 = data["Tht"], data["Tnpv"]   # TT
    Sht_tot2, S_npvs2 = data["Aht"], data["Anpv"]   # AA

    N = len(B_npvs)

    # -------- batching settings --------
    
    chunk_size = 50_000
    start_event = chunk_size * 10

    # -------- fixed menu cut --------
    fixed_Ht_cut = float(np.percentile(Bht_tot[start_event:start_event + 100_000], 99.75))
    print("fixed_Ht_cut =", fixed_Ht_cut)
    print("np.percentile(Bht_tot[start_event:],99.75) =", np.percentile(Bht_tot[start_event:], 99.75))

    # define a reasonable cut range for clipping & normalization
    ht_lo = float(np.percentile(Bht_tot[start_event:], 95.0))
    ht_hi = float(np.percentile(Bht_tot[start_event:], 99.99))
    cut_mid = 0.5 * (ht_lo + ht_hi)
    cut_span = max(1.0, (ht_hi - ht_lo))

    # -------- DQN hyperparams --------
    target = 0.25  # percent
    tol = 0.03     # percent (0.22-0.28 band ~ +/-0.03)

    # reward shaping weights
    alpha = 0.2    # signal bonus weight
    beta = 0.02    # cut-change penalty

    # discrete HT cut steps (GeV)
    HT_DELTAS = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    MAX_DELTA = float(np.max(np.abs(HT_DELTAS)))

    agent = DQNAgent(obs_dim=3, n_actions=len(HT_DELTAS), lr=1e-3, gamma=0.95)

    # -------- controllers initial cuts --------
    Ht_cut_pd = fixed_Ht_cut
    Ht_cut_dqn = fixed_Ht_cut

    pre_pd_err = 0.0

    # DQN memory for (s,a) chosen for NEXT batch
    prev_obs = None
    prev_action = None
    prev_bg_rate = None
    last_delta = 0.0

    # -------- logs --------
    steps = []
    bg_const, bg_pd, bg_dqn = [], [], []
    ht_pd_hist, ht_dqn_hist = [], []

    # signal efficiencies (%)
    tt_const, tt_pd, tt_dqn = [], [], []
    aa_const, aa_pd, aa_dqn = [], [], []

    losses = []

    # iterate by batches directly (much faster than looping all events)
    batch_starts = list(range(start_event, N, chunk_size))
    print(f"Processing {len(batch_starts)} batches from event {start_event} to {N} (chunk={chunk_size}).")

    for t, I in enumerate(batch_starts):
        start_idx = I
        end_idx = min(I + chunk_size, N)
        idx = np.arange(start_idx, end_idx)

        bht = Bht_tot[idx]
        b_npvs = B_npvs[idx]

        npv_min = np.min(b_npvs)
        npv_max = np.max(b_npvs)

        # match signal to the npv band
        sig_mask1 = (S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
        sig_mask2 = (S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)

        sht_tt = Sht_tot1[sig_mask1]
        sht_aa = Sht_tot2[sig_mask2]

        # -------- compute rates for each strategy --------
        bg_r_const = Sing_Trigger(np.asarray(bht), fixed_Ht_cut)
        bg_r_pd_   = Sing_Trigger(np.asarray(bht), Ht_cut_pd)
        bg_r_dqn_  = Sing_Trigger(np.asarray(bht), Ht_cut_dqn)

        # signal efficiencies (% accepted)
        tt_r_const = Sing_Trigger(np.asarray(sht_tt), fixed_Ht_cut)
        tt_r_pd_   = Sing_Trigger(np.asarray(sht_tt), Ht_cut_pd)
        tt_r_dqn_  = Sing_Trigger(np.asarray(sht_tt), Ht_cut_dqn)

        aa_r_const = Sing_Trigger(np.asarray(sht_aa), fixed_Ht_cut)
        aa_r_pd_   = Sing_Trigger(np.asarray(sht_aa), Ht_cut_pd)
        aa_r_dqn_  = Sing_Trigger(np.asarray(sht_aa), Ht_cut_dqn)

        # -------- PD update for next batch --------
        Ht_cut_pd, pre_pd_err = PD_controller1(bg_r_pd_, pre_pd_err, Ht_cut_pd)
        Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))

        # -------- DQN update (reward for previous action becomes available now) --------
        # build current observation for DQN based on DQN-controlled rate
        if prev_bg_rate is None:
            prev_bg_rate = bg_r_dqn_

        obs = make_obs(bg_r_dqn_, prev_bg_rate, Ht_cut_dqn, cut_mid, cut_span, target)

        if prev_obs is not None and prev_action is not None:
            bg_pen = abs(bg_r_dqn_ - target) / tol
            sig_term = 0.5 * (tt_r_dqn_ + aa_r_dqn_) / 100.0
            reward = -bg_pen + alpha * sig_term - beta * (abs(last_delta) / max(1e-9, MAX_DELTA))

            agent.buf.push(prev_obs, prev_action, reward, obs, done=False)
            loss = agent.train_step(batch_size=128, target_update=200)
            if loss is not None:
                losses.append(loss)

        # choose action for NEXT batch
        # epsilon schedule: start high, decay quickly over batches
        eps = max(0.05, 1.0 * (0.97 ** t))
        action = agent.act(obs, eps=eps)
        delta = float(HT_DELTAS[action])
        sd = shield_delta(bg_r_dqn_, target, tol, MAX_DELTA)
        if sd is not None:
            delta = float(sd)

        prev_obs = obs
        prev_action = action
        prev_bg_rate = bg_r_dqn_
        last_delta = delta

        # apply cut update for NEXT batch
        Ht_cut_dqn = float(np.clip(Ht_cut_dqn + delta, ht_lo, ht_hi))

        # -------- record --------
        steps.append(t)
        bg_const.append(bg_r_const)
        bg_pd.append(bg_r_pd_)
        bg_dqn.append(bg_r_dqn_)

        ht_pd_hist.append(Ht_cut_pd)
        ht_dqn_hist.append(Ht_cut_dqn)

        tt_const.append(tt_r_const)
        tt_pd.append(tt_r_pd_)
        tt_dqn.append(tt_r_dqn_)

        aa_const.append(aa_r_const)
        aa_pd.append(aa_r_pd_)
        aa_dqn.append(aa_r_dqn_)

        if t % 5 == 0:
            last_loss = losses[-1] if losses else None
            print(
                f"[batch {t:4d}] "
                f"bg% const={bg_r_const:.3f} pd={bg_r_pd_:.3f} dqn={bg_r_dqn_:.3f} | "
                f"cut pd={ht_pd_hist[-1]:.1f} dqn={ht_dqn_hist[-1]:.1f} | "
                f"eps={eps:.3f} loss={last_loss}"
            )

    # ------------------------- plots -------------------------
    steps_arr = np.asarray(steps)

    # bg acceptance
    plt.figure(figsize=(12, 6))
    plt.plot(steps_arr, bg_const, label="Constant Menu", linewidth=3, linestyle="dashed")
    plt.plot(steps_arr, bg_pd,    label="PD Controller", linewidth=2.5, linestyle="solid")
    plt.plot(steps_arr, bg_dqn,   label="DQN", linewidth=2.5, linestyle="solid")
    plt.axhline(y=0.28, color="gray", linestyle="--", linewidth=1.5, label="Upper Tolerance (0.28)")
    plt.axhline(y=0.22, color="gray", linestyle="--", linewidth=1.5, label="Lower Tolerance (0.22)")
    plt.ylim(0, 0.7)
    plt.title("HT Trigger Background Acceptance: Constant vs PD vs DQN", fontsize=18)
    plt.xlabel(f"Time (batch = {chunk_size} events)", fontsize=18)
    plt.ylabel("Acceptance (%)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12, loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(outdir, "bht_rate_const_pd_dqn.pdf"))
    plt.close()

    # cut evolution
    plt.figure(figsize=(12, 5))
    plt.plot(steps_arr, ht_pd_hist, label="Ht_cut PD", linewidth=2.0)
    plt.plot(steps_arr, ht_dqn_hist, label="Ht_cut DQN", linewidth=2.0)
    plt.axhline(y=fixed_Ht_cut, linestyle="--", linewidth=1.5, label="fixed_Ht_cut")
    plt.title("HT Cut Evolution (PD vs DQN)", fontsize=16)
    plt.xlabel(f"Time (batch = {chunk_size} events)", fontsize=14)
    plt.ylabel("Ht_cut", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(outdir, "ht_cut_pd_vs_dqn.pdf"))
    plt.close()

    # cumulative signal efficiency (normalize by mean like your plots)
    def cummean(x):
        x = np.asarray(x, dtype=np.float64)
        c = np.cumsum(x)
        return c / (np.arange(len(x)) + 1.0)

    tt_c_const = cummean(tt_const)
    tt_c_pd    = cummean(tt_pd)
    tt_c_dqn   = cummean(tt_dqn)

    aa_c_const = cummean(aa_const)
    aa_c_pd    = cummean(aa_pd)
    aa_c_dqn   = cummean(aa_dqn)

    # normalize by each curve's mean (so shape comparisons match your style)
    def norm_by_mean(x):
        x = np.asarray(x, dtype=np.float64)
        m = np.mean(x) if len(x) else 1.0
        return x / max(1e-9, m)

    plt.figure(figsize=(12, 6))
    plt.plot(steps_arr, norm_by_mean(tt_c_const), label=f"Const ttbar (mean={np.mean(tt_c_const):.3f})", linestyle="dashed", linewidth=2.5)
    plt.plot(steps_arr, norm_by_mean(tt_c_pd),    label=f"PD    ttbar (mean={np.mean(tt_c_pd):.3f})",  linestyle="solid",  linewidth=2.5)
    plt.plot(steps_arr, norm_by_mean(tt_c_dqn),   label=f"DQN   ttbar (mean={np.mean(tt_c_dqn):.3f})", linestyle="solid",  linewidth=2.5)

    plt.plot(steps_arr, norm_by_mean(aa_c_const), label=f"Const HToAATo4B (mean={np.mean(aa_c_const):.3f})", linestyle="dotted", linewidth=2.5)
    plt.plot(steps_arr, norm_by_mean(aa_c_pd),    label=f"PD    HToAATo4B (mean={np.mean(aa_c_pd):.3f})",  linestyle="solid",  linewidth=2.0)
    plt.plot(steps_arr, norm_by_mean(aa_c_dqn),   label=f"DQN   HToAATo4B (mean={np.mean(aa_c_dqn):.3f})", linestyle="solid",  linewidth=2.0)

    plt.ylim(0.75, 1.6)
    plt.title("HT Trigger Cumulative Signal Efficiency: Constant vs PD vs DQN", fontsize=18)
    plt.xlabel(f"Time (batch = {chunk_size} events)", fontsize=18)
    plt.ylabel("Cumulative Efficiency / mean", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=10, loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(outdir, "sht_cumeff_const_pd_dqn.pdf"))
    plt.close()

    # loss curve (if any)
    if losses:
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(losses)), losses, linewidth=1.5)
        plt.title("DQN training loss (SmoothL1)", fontsize=14)
        plt.xlabel("Gradient step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(outdir, "dqn_loss.pdf"))
        plt.close()

    print(f"\nSaved plots to: {outdir}")
    print(" - bht_rate_const_pd_dqn.pdf")
    print(" - ht_cut_pd_vs_dqn.pdf")
    print(" - sht_cumeff_const_pd_dqn.pdf")
    if losses:
        print(" - dqn_loss.pdf")

if __name__ == "__main__":
    main()
