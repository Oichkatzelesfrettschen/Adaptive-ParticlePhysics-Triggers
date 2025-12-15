# DQN using background samples SingleTrigger
#!/usr/bin/env python3
"""
demo_single_trigger_dqn_ht.py  (SingleTrigger: Constant vs PD vs DQN)

Goal: DQN on HT trigger (background samples) + PD + Constant menu,
and generate the *same family + styling* of plots as Data_SingleTrigger.py:

HT plots (Const vs PD vs DQN):
  - paper/bht_rate_pidData_dqn.pdf/.png          (background rate [kHz])
  - paper/ht_cut_pidData_dqn.pdf/.png            (Ht_cut evolution)
  - paper/sht_rate_pidData2data_dqn.pdf/.png     (cumulative signal eff, relative to t0)
  - paper/L_sht_rate_pidData2data_dqn.pdf/.png   (local signal eff, relative to t0)
  - paper/dqn_loss_ht.pdf/.png                   (if any)

AS plots (Const vs PD only, dim=1 by default; generated only if score exists):
  - paper/bas_rate_pidData.pdf/.png              (background rate [kHz])
  - paper/sas_rate_pidData2data.pdf/.png         (cumulative signal eff, relative to t0)
  - paper/L_sas_rate_pidData2data.pdf/.png       (local signal eff, relative to t0)

Notes:
- Rates are computed in percent (%) and then scaled to kHz by *400, matching your Data_SingleTrigger.py.
- For "Matched_data_2016.h5" style files, signals are matched by index (same chunk indices).
- For Trigger_food_MC.h5 style files, signals are matched by NPV band per chunk (like your earlier DQN script).

"""

import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import h5py
import hdf5plugin  # noqa: F401
import argparse
from pathlib import Path
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
import mplhep as hep
from utils import cummean, rel_to_t0, add_cms_header, save_pdf_png  

hep.style.use("CMS")

# ------------------------- reproducibility -------------------------
SEED = 20251213
random.seed(SEED)
np.random.seed(SEED)

# ------------------------- output -------------------------
OUTDIR = "outputs/demo_sing_dqn"
os.makedirs(OUTDIR, exist_ok=True)




def read_data(h5_file_path, score_dim=2):
    """
    Supports:
      - new Trigger_food_v2: mc_*_score{dim:02d} (e.g. mc_bkg_score02)
      - legacy Trigger_food: mc_*_score01/04
    """
    score_key = f"score{int(score_dim):02d}"  # e.g. score02

    with h5py.File(h5_file_path, "r") as h5:
        keys = set(h5.keys())

        Bht = h5["mc_bkg_ht"][:]
        Bnpv = h5["mc_bkg_Npv"][:]

        Tht = h5["mc_tt_ht"][:]
        Tnpv = h5["tt_Npv"][:]

        Aht = h5["mc_aa_ht"][:]
        Anpv = h5["aa_Npv"][:]

        def _read_score(prefix):
            k_new = f"{prefix}_{score_key}"
            if k_new in keys:
                return h5[k_new][:]
            k01 = f"{prefix}_score01"
            k04 = f"{prefix}_score04"
            if k01 in keys:
                return h5[k01][:]
            if k04 in keys:
                return h5[k04][:]
            return None

        Bas = _read_score("mc_bkg")
        T_as = _read_score("mc_tt")
        A_as = _read_score("mc_aa")

    return dict(
        Bht=Bht, Bnpv=Bnpv,
        Tht=Tht, Tnpv=Tnpv,
        Aht=Aht, Anpv=Anpv,
        Bas=Bas, Tas=T_as, Aas=A_as,
    )


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
    def __init__(self, capacity=50000):
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
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

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
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        sp = torch.tensor(sp, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        # Double-DQN target
        with torch.no_grad():
            a_star = torch.argmax(self.q(sp), dim=1, keepdim=True)
            q_sp = self.tgt(sp).gather(1, a_star)
            y = r + (1.0 - done) * self.gamma * q_sp

        loss = nn.SmoothL1Loss()(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

def make_obs(bg_rate, prev_bg_rate, cut, cut_mid, cut_span, target):
    x_rate  = (bg_rate - target) / max(1e-6, target)
    x_drate = (bg_rate - prev_bg_rate) / max(1e-6, target)
    x_cut   = (cut - cut_mid) / max(1e-6, cut_span)
    return np.array([x_rate, x_drate, x_cut], dtype=np.float32)


def shield_delta(bg_rate, target, tol, max_delta):
    if bg_rate > target + tol:
        return +max_delta
    if bg_rate < target - tol:
        return -max_delta
    return None

def _first_present(h5_keys, candidates):
    for k in candidates:
        if k in h5_keys:
            return k
    return None

def read_any_h5(path, score_dim_hint=1):
    """
    Returns:
      dict with unified keys:
        Bht, Bnpv, Bas1, Bas4
        Tht, Tnpv, Tas1, Tas4
        Aht, Anpv, Aas1, Aas4
      plus:
        meta['matched_by_index']  True for Matched_data_*.h5
    """
    with h5py.File(path, "r") as h5:
        keys = set(h5.keys())

        # ---------- Matched data format (Data_SingleTrigger.py) ----------
        has_data = ("data_ht" in keys) or ("data_Npv" in keys) or any(k.startswith("data_") for k in keys)
        has_tt   = any(k.startswith("matched_tt_") for k in keys)
        has_aa   = any(k.startswith("matched_aa_") for k in keys)
        is_matched = has_data and has_tt and has_aa
        if is_matched:
            Bht  = h5["data_ht"][:]
            Bnpv = h5["data_Npv"][:]

            Bas1 = h5[_first_present(keys, ["data_scores01", "data_score01", "data_scores1"])] if _first_present(keys, ["data_scores01", "data_score01", "data_scores1"]) else None
            Bas4 = h5[_first_present(keys, ["data_scores04", "data_score04", "data_scores4"])] if _first_present(keys, ["data_scores04", "data_score04", "data_scores4"]) else None
            Bas1 = Bas1[:] if Bas1 is not None else None
            Bas4 = Bas4[:] if Bas4 is not None else None

            Tht  = h5["matched_tt_ht"][:]
            Tnpv = h5[_first_present(keys, ["matched_tt_npvs", "matched_tt_Npv", "matched_tt_npv"])] if _first_present(keys, ["matched_tt_npvs", "matched_tt_Npv", "matched_tt_npv"]) else None
            Tnpv = Tnpv[:] if Tnpv is not None else np.zeros_like(Tht, dtype=np.float32)

            Tas1 = h5[_first_present(keys, ["matched_tt_scores01", "matched_tt_score01"])] if _first_present(keys, ["matched_tt_scores01", "matched_tt_score01"]) else None
            Tas4 = h5[_first_present(keys, ["matched_tt_scores04", "matched_tt_score04"])] if _first_present(keys, ["matched_tt_scores04", "matched_tt_score04"]) else None
            Tas1 = Tas1[:] if Tas1 is not None else None
            Tas4 = Tas4[:] if Tas4 is not None else None

            Aht  = h5["matched_aa_ht"][:]
            Anpv = h5[_first_present(keys, ["matched_aa_npvs", "matched_aa_Npv", "matched_aa_npv"])] if _first_present(keys, ["matched_aa_npvs", "matched_aa_Npv", "matched_aa_npv"]) else None
            Anpv = Anpv[:] if Anpv is not None else np.zeros_like(Aht, dtype=np.float32)

            Aas1 = h5[_first_present(keys, ["matched_aa_scores01", "matched_aa_score01"])] if _first_present(keys, ["matched_aa_scores01", "matched_aa_score01"]) else None
            Aas4 = h5[_first_present(keys, ["matched_aa_scores04", "matched_aa_score04"])] if _first_present(keys, ["matched_aa_scores04", "matched_aa_score04"]) else None
            Aas1 = Aas1[:] if Aas1 is not None else None
            Aas4 = Aas4[:] if Aas4 is not None else None

            return dict(
                Bht=Bht, Bnpv=Bnpv, Bas1=Bas1, Bas4=Bas4,
                Tht=Tht, Tnpv=Tnpv, Tas1=Tas1, Tas4=Tas4,
                Aht=Aht, Anpv=Anpv, Aas1=Aas1, Aas4=Aas4,
                meta=dict(matched_by_index=True),
            )

        # ---------- MC Trigger_food_* format ----------
        Bht  = h5["mc_bkg_ht"][:]
        Bnpv = h5["mc_bkg_Npv"][:]

        Tht  = h5["mc_tt_ht"][:]
        Tnpv = h5[_first_present(keys, ["tt_Npv", "mc_tt_Npv", "mc_tt_npv"])] if _first_present(keys, ["tt_Npv", "mc_tt_Npv", "mc_tt_npv"]) else None
        Tnpv = Tnpv[:] if Tnpv is not None else np.zeros_like(Tht, dtype=np.float32)

        Aht  = h5["mc_aa_ht"][:]
        Anpv = h5[_first_present(keys, ["aa_Npv", "mc_aa_Npv", "mc_aa_npv"])] if _first_present(keys, ["aa_Npv", "mc_aa_Npv", "mc_aa_npv"]) else None
        Anpv = Anpv[:] if Anpv is not None else np.zeros_like(Aht, dtype=np.float32)

        # scores: support legacy 01/04 and newer score{dim:02d}
        def read_score(prefix, which):
            # which: "01", "04", or f"{score_dim_hint:02d}"
            k = f"{prefix}_score{which}"
            return h5[k][:] if k in keys else None

        Bas1 = read_score("mc_bkg", "01")
        Bas4 = read_score("mc_bkg", "04")
        Tas1 = read_score("mc_tt",  "01")
        Tas4 = read_score("mc_tt",  "04")
        Aas1 = read_score("mc_aa",  "01")
        Aas4 = read_score("mc_aa",  "04")

        # if no 01/04 exist, try hinted dim (e.g. score02)
        if (Bas1 is None) and (Bas4 is None):
            hint = f"{int(score_dim_hint):02d}"
            Bas1 = read_score("mc_bkg", hint)
            Tas1 = read_score("mc_tt",  hint)
            Aas1 = read_score("mc_aa",  hint)

        return dict(
            Bht=Bht, Bnpv=Bnpv, Bas1=Bas1, Bas4=Bas4,
            Tht=Tht, Tnpv=Tnpv, Tas1=Tas1, Tas4=Tas4,
            Aht=Aht, Anpv=Anpv, Aas1=Aas1, Aas4=Aas4,
            meta=dict(matched_by_index=False),
        )


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data/Trigger_food_MC.h5",
                    help="Trigger_food_*.h5 (MC) or Matched_data_2016.h5 (data)")

    ap.add_argument("--outdir", default="outputs/demo_sing_dqn", help="output root")
    # ap.add_argument("--paper-subdir", default="paper", help="subdir under outdir for plots")

    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--start-event", type=int, default=None,
                    help="start event index (default: chunk_size * start_batches)")
    ap.add_argument("--start-batches", type=int, default=10,
                    help="if start-event not set, start_event = chunk_size * start_batches")

    ap.add_argument("--score-dim-hint", type=int, default=2,
                    help="if file has only scoreXX, use this dim (e.g. 2 -> score02)")

    ap.add_argument("--as-dim", type=int, default=1, choices=[1, 4],
                    help="which AS dimension to plot/control (1 uses *_score01; 4 uses *_score04)")

    # DQN action set for HT cut (GeV)
    ap.add_argument("--ht-deltas", type=str, default="-2,-1,0,1,2",
                    help="comma-separated HT deltas in GeV, e.g. -3,-1,0,1,3")

    ap.add_argument("--cms-run-label", default="Run 283408",
                    help='right-side header text, e.g. "Run 283408"')
    
    ap.add_argument("--force-matched", action="store_true",
                help="Force matched-by-index (real data Matched_data_*.h5) mode")
    ap.add_argument("--data-chunk-size", type=int, default=20_000,
                help="Chunk size for matched data (default 20000)")
    ap.add_argument("--data-start-event", type=int, default=200_000,
                help="Start event for matched data (default 200000)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    paper_dir = outdir 
    paper_dir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5(args.input, score_dim_hint=args.score_dim_hint)
    meta = d["meta"]
    matched_by_index = bool(meta.get("matched_by_index", False))

    Bht, Bnpv = d["Bht"], d["Bnpv"]
    Tht, Tnpv = d["Tht"], d["Tnpv"]
    Aht, Anpv = d["Aht"], d["Anpv"]

    # AS choose dim
    if args.as_dim == 1:
        Bas, Tas, Aas = d["Bas1"], d["Tas1"], d["Aas1"]
    else:
        Bas, Tas, Aas = d["Bas4"], d["Tas4"], d["Aas4"]

    have_as = (Bas is not None) and (Tas is not None) and (Aas is not None)

    N = len(Bht)
    if matched_by_index:
        chunk_size  = int(args.data_chunk_size)      # 20000
        start_event = int(args.data_start_event)     # 200000

        # If file is shorter than expected, fall back safely
        if start_event >= N:
            start_event = 0
    else:
        chunk_size  = int(args.chunk_size)
        start_event = int(args.start_event) if args.start_event is not None else chunk_size * int(args.start_batches)

    start_event = max(0, (start_event // chunk_size) * chunk_size)
    if start_event + chunk_size > N:
        start_event = max(0, ((N - chunk_size) // chunk_size) * chunk_size)

    # -------- fixed cuts (use a window) --------
    if matched_by_index:
        win_lo = min(200_000, N-1)
        win_hi = min(210_000, N)
        if win_hi - win_lo < 1000:
            win_lo = 0
            win_hi = min(10_000, N)
    else:
        win_lo = start_event
        win_hi = min(N, start_event + 10_000)

    fixed_Ht_cut = float(np.percentile(Bht[win_lo:win_hi], 99.75))
    if have_as:
        fixed_AS_cut = float(np.percentile(Bas[win_lo:win_hi], 99.75))

    # clip ranges for stability
    ht_lo = float(np.percentile(Bht[start_event:], 95.0))
    ht_hi = float(np.percentile(Bht[start_event:], 99.99))
    ht_mid = 0.5 * (ht_lo + ht_hi)
    ht_span = max(1.0, ht_hi - ht_lo)

    if have_as:
        as_lo = float(np.percentile(Bas[start_event:], 95.0))
        as_hi = float(np.percentile(Bas[start_event:], 99.99))

    print(f"[INFO] matched_by_index = {matched_by_index}")
    print(f"[HT] fixed_Ht_cut={fixed_Ht_cut:.3f} clip=({ht_lo:.3f},{ht_hi:.3f}) start_event={start_event} chunk={chunk_size}")
    if have_as:
        print(f"[AS dim={args.as_dim}] fixed_AS_cut={fixed_AS_cut:.6f} clip=({as_lo:.6f},{as_hi:.6f})")
    else:
        print("[AS] score missing -> skipping AS plots")

    # -------- controllers init --------
    Ht_cut_pd = fixed_Ht_cut
    Ht_cut_dqn = fixed_Ht_cut
    pre_pd_err = 0.0

    if have_as:
        AS_cut_pd = fixed_AS_cut
        pre_as_err = 0.0

    # -------- DQN config --------
    target = 0.25  # %
    tol = 0.03     # %

    alpha = 0.2    # signal bonus
    beta  = 0.02   # delta penalty

    HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
    MAX_DELTA = float(np.max(np.abs(HT_DELTAS)))

    agent = DQNAgent(obs_dim=3, n_actions=len(HT_DELTAS), lr=5e-4, gamma=0.95)

    prev_obs = None
    prev_action = None
    prev_bg_rate = None
    last_delta = 0.0
    losses_ht = []

    # -------- logs (HT) --------
    R_const, R_pd, R_dqn = [], [], []          # background (%)
    Ht_pd_hist, Ht_dqn_hist = [], []

    L_tt_const, L_tt_pd, L_tt_dqn = [], [], []  # local sig tt (%)
    L_aa_const, L_aa_pd, L_aa_dqn = [], [], []  # local sig aa (%)

    # -------- logs (AS, Const vs PD) --------
    E_const, E_pd = [], []                    # background (%)
    L_tt_as_const, L_tt_as_pd = [], []
    L_aa_as_const, L_aa_as_pd = [], []

    # -------- main batching loop --------
    batch_starts = list(range(start_event, N, chunk_size))
    print(f"[INFO] matched_by_index={matched_by_index} N={N} chunk={chunk_size} start_event={start_event}")
    print(f"[INFO] batches={len(batch_starts)} first5={batch_starts[:5]}")
    print(f"[INFO] fixed window=[{win_lo}:{win_hi}]")
    

    for t, I in enumerate(batch_starts):
        idx = np.arange(I, min(I + chunk_size, N))

        bht = Bht[idx]
        bnpv = Bnpv[idx]

        # signals:
        if matched_by_index:
            # matched arrays should be at least as long; clamp if not
            idx_s_tt = idx[idx < len(Tht)]
            idx_s_aa = idx[idx < len(Aht)]
            sht_tt = Tht[idx_s_tt]
            sht_aa = Aht[idx_s_aa]

            if have_as:
                sas_tt = Tas[idx_s_tt]
                sas_aa = Aas[idx_s_aa]
        else:
            # MC: match signals by NPV band
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)

            sht_tt = Tht[mask_tt]
            sht_aa = Aht[mask_aa]
            if have_as:
                sas_tt = Tas[mask_tt]
                sas_aa = Aas[mask_aa]

        # ---------- HT rates ----------
        bg_r_const = Sing_Trigger(bht, fixed_Ht_cut)
        bg_r_pd    = Sing_Trigger(bht, Ht_cut_pd)
        bg_r_dqn   = Sing_Trigger(bht, Ht_cut_dqn)

        tt_r_const = Sing_Trigger(sht_tt, fixed_Ht_cut)
        tt_r_pd    = Sing_Trigger(sht_tt, Ht_cut_pd)
        tt_r_dqn   = Sing_Trigger(sht_tt, Ht_cut_dqn)

        aa_r_const = Sing_Trigger(sht_aa, fixed_Ht_cut)
        aa_r_pd    = Sing_Trigger(sht_aa, Ht_cut_pd)
        aa_r_dqn   = Sing_Trigger(sht_aa, Ht_cut_dqn)

        # PD update
        Ht_cut_pd, pre_pd_err = PD_controller1(bg_r_pd, pre_pd_err, Ht_cut_pd)
        Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))

        # DQN update
        if prev_bg_rate is None:
            prev_bg_rate = bg_r_dqn

        obs = make_obs(bg_r_dqn, prev_bg_rate, Ht_cut_dqn, ht_mid, ht_span, target)

        if (prev_obs is not None) and (prev_action is not None):
            bg_pen = abs(bg_r_dqn - target) / tol
            sig_term = 0.5 * (tt_r_dqn + aa_r_dqn) / 100.0
            reward = -bg_pen + alpha * sig_term - beta * (abs(last_delta) / max(1e-9, MAX_DELTA))
            reward = float(np.clip(reward, -10.0, 10.0))
            agent.buf.push(prev_obs, prev_action, reward, obs, done=False)
            loss = agent.train_step(batch_size=128, target_update=200)
            if loss is not None:
                losses_ht.append(loss)

        # choose action for next batch
        eps = max(0.05, 1.0 * (0.98 ** t))
        action = agent.act(obs, eps=eps)
        delta = float(HT_DELTAS[action])

        sd = shield_delta(bg_r_dqn, target, tol, MAX_DELTA)
        if sd is not None:
            delta = float(sd)

        prev_obs = obs
        prev_action = action
        prev_bg_rate = bg_r_dqn
        last_delta = delta

        Ht_cut_dqn = float(np.clip(Ht_cut_dqn + delta, ht_lo, ht_hi))

        # record HT
        R_const.append(bg_r_const)
        R_pd.append(bg_r_pd)
        R_dqn.append(bg_r_dqn)

        Ht_pd_hist.append(Ht_cut_pd)
        Ht_dqn_hist.append(Ht_cut_dqn)

        L_tt_const.append(tt_r_const)
        L_tt_pd.append(tt_r_pd)
        L_tt_dqn.append(tt_r_dqn)

        L_aa_const.append(aa_r_const)
        L_aa_pd.append(aa_r_pd)
        L_aa_dqn.append(aa_r_dqn)

        # ---------- AS (Const vs PD only) ----------
        if have_as:
            bas = Bas[idx] if idx[-1] < len(Bas) else Bas[idx[idx < len(Bas)]]

            bg_as_const = Sing_Trigger(bas, fixed_AS_cut)
            bg_as_pd    = Sing_Trigger(bas, AS_cut_pd)

            tt_as_const = Sing_Trigger(sas_tt, fixed_AS_cut)
            tt_as_pd    = Sing_Trigger(sas_tt, AS_cut_pd)

            aa_as_const = Sing_Trigger(sas_aa, fixed_AS_cut)
            aa_as_pd    = Sing_Trigger(sas_aa, AS_cut_pd)

            AS_cut_pd, pre_as_err = PD_controller2(bg_as_pd, pre_as_err, AS_cut_pd)
            AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))

            E_const.append(bg_as_const)
            E_pd.append(bg_as_pd)

            L_tt_as_const.append(tt_as_const)
            L_tt_as_pd.append(tt_as_pd)
            L_aa_as_const.append(aa_as_const)
            L_aa_as_pd.append(aa_as_pd)

        if t % 5 == 0:
            last_loss = losses_ht[-1] if losses_ht else None
            print(f"[batch {t:4d}] bg% const={bg_r_const:.3f} pd={bg_r_pd:.3f} dqn={bg_r_dqn:.3f} "
                  f"| cut pd={Ht_cut_pd:.1f} dqn={Ht_cut_dqn:.1f} | eps={eps:.3f} loss={last_loss}")

    # ------------------------- convert to numpy -------------------------
    R_const = np.array(R_const); R_pd = np.array(R_pd); R_dqn = np.array(R_dqn)
    Ht_pd_hist = np.array(Ht_pd_hist); Ht_dqn_hist = np.array(Ht_dqn_hist)

    L_tt_const = np.array(L_tt_const); L_tt_pd = np.array(L_tt_pd); L_tt_dqn = np.array(L_tt_dqn)
    L_aa_const = np.array(L_aa_const); L_aa_pd = np.array(L_aa_pd); L_aa_dqn = np.array(L_aa_dqn)

    if have_as:
        E_const = np.array(E_const); E_pd = np.array(E_pd)
        L_tt_as_const = np.array(L_tt_as_const); L_tt_as_pd = np.array(L_tt_as_pd)
        L_aa_as_const = np.array(L_aa_as_const); L_aa_as_pd = np.array(L_aa_as_pd)

    # ------------------------- style knobs (match your Data script) -------------------------
    RATE_SCALE_KHZ = 400.0
    upper_tol_khz = 0.275 * RATE_SCALE_KHZ
    lower_tol_khz = 0.225 * RATE_SCALE_KHZ

    styles = {
        "Constant": {"linestyle": "dashed", "linewidth": 2.5},
        "PD":       {"linestyle": "solid",  "linewidth": 2.0},
        "DQN":      {"linestyle": "solid",  "linewidth": 2.0},
    }

    # time axis is fraction of run
    time = np.linspace(0, 1, len(R_const))

    # =========================================================
    # 1) HT background rate [kHz]
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, R_const * RATE_SCALE_KHZ, color="tab:blue", linewidth=3, linestyle="dashed")
    ax.plot(time, R_pd    * RATE_SCALE_KHZ, color="mediumblue", linewidth=2.5, linestyle="solid")
    ax.plot(time, R_dqn   * RATE_SCALE_KHZ, color="tab:purple", linewidth=2.5, linestyle="solid")

    ax.axhline(y=upper_tol_khz, color="gray", linestyle="--", linewidth=1.5)
    ax.axhline(y=lower_tol_khz, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Background Rate [kHz]", loc="center")
    ax.set_ylim(0, 200)
    ax.grid(True, linestyle="--", alpha=0.6)

    # main legend
    h_const = mlines.Line2D([], [], color="tab:blue", linestyle="dashed", linewidth=3)
    h_pd    = mlines.Line2D([], [], color="mediumblue", linestyle="solid", linewidth=2.5)
    h_dqn   = mlines.Line2D([], [], color="tab:purple", linestyle="solid", linewidth=2.5)
    leg_main = ax.legend([h_const, h_pd, h_dqn],
                         ["Constant Menu", "PD Controller", "DQN"],
                         title="HT Trigger", loc="upper left",
                         bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=14)
    ax.add_artist(leg_main)

    # tolerance legend
    upper_tol = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    lower_tol = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    ax.legend([upper_tol, lower_tol],
              ["Upper Tolerance (110)", "Lower Tolerance (90)"],
              title="Reference", loc="upper right",
              bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=14)

    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(paper_dir / "bht_rate_pidData_dqn"))
    plt.close(fig)

    # =========================================================
    # 2) HT cut evolution
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Ht_pd_hist, color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time, Ht_dqn_hist, color="tab:purple", linewidth=2.0, label="DQN")
    ax.axhline(y=fixed_Ht_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_Ht_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Ht_cut [GeV]", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="HT Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(paper_dir / "ht_cut_pidData_dqn"))
    plt.close(fig)

    # =========================================================
    # 3) HT cumulative signal efficiency (relative to t0)
    # =========================================================
    tt_c_const = cummean(L_tt_const)
    tt_c_pd    = cummean(L_tt_pd)
    tt_c_dqn   = cummean(L_tt_dqn)

    aa_c_const = cummean(L_aa_const)
    aa_c_pd    = cummean(L_aa_pd)
    aa_c_dqn   = cummean(L_aa_dqn)

    colors_ht = {"ttbar": "goldenrod", "HToAATo4B": "seagreen"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(tt_c_const), color=colors_ht["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_const), color=colors_ht["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")

    ax.plot(time, rel_to_t0(tt_c_pd), color=colors_ht["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")

    ax.plot(time, rel_to_t0(tt_c_dqn), color=colors_ht["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_dqn), color=colors_ht["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.85, 1.5)
    ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(paper_dir / "sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # =========================================================
    # 4) HT local signal efficiency (relative to t0)
    # =========================================================
    colors_ht_local = {"ttbar": "goldenrod", "HToAATo4B": "seagreen"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(L_tt_const), color=colors_ht_local["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_const), color=colors_ht_local["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_const[0]:.2f}\%$)")

    ax.plot(time, rel_to_t0(L_tt_pd), color=colors_ht_local["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_pd), color=colors_ht_local["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_pd[0]:.2f}\%$)")

    ax.plot(time, rel_to_t0(L_tt_dqn), color=colors_ht_local["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_dqn[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_dqn), color=colors_ht_local["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_dqn[0]:.2f}\%$)")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 1.6)
    ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(paper_dir / "L_sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # =========================================================
    # 5) DQN loss
    # =========================================================
    if losses_ht:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_ht)), losses_ht, linewidth=1.5)
        ax.set_title("HT DQN training loss (SmoothL1)")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=args.cms_run_label)
        save_pdf_png(fig, str(paper_dir / "dqn_loss_ht"))
        plt.close(fig)

    # =========================================================
    # AS plots (Const vs PD only), if available
    # =========================================================
    if have_as:
        time_as = np.linspace(0, 1, len(E_const))
        E_const_khz = np.array(E_const) * RATE_SCALE_KHZ
        E_pd_khz    = np.array(E_pd)    * RATE_SCALE_KHZ

        # (A1) AS background rate [kHz]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_as, E_const_khz, color="tab:blue", linewidth=3, linestyle="dotted")
        ax.plot(time_as, E_pd_khz,    color="mediumblue", linewidth=2.5, linestyle="solid")

        ax.axhline(y=upper_tol_khz, color="gray", linestyle="--", linewidth=1.5)
        ax.axhline(y=lower_tol_khz, color="gray", linestyle="--", linewidth=1.5)

        ax.set_xlabel("Time (Fraction of Run)", loc="center")
        ax.set_ylabel("Background Rate [kHz]", loc="center")
        ax.set_ylim(60, 200)
        ax.grid(True, linestyle="--", alpha=0.6)

        header_const = mlines.Line2D([], [], color="none", linestyle="none")
        header_pd    = mlines.Line2D([], [], color="none", linestyle="none")
        const_dim = mlines.Line2D([], [], color="tab:blue", linestyle="dotted", linewidth=3)
        pd_dim    = mlines.Line2D([], [], color="mediumblue", linestyle="solid", linewidth=2.5)

        leg_main = ax.legend(
            [header_const, const_dim, header_pd, pd_dim],
            ["Constant Menu", f"model dim={args.as_dim}", "PD Controller", f"model dim={args.as_dim}"],
            title="AD Trigger", fontsize=14, ncol=1, loc="upper left",
            bbox_to_anchor=(0.02, 0.98), frameon=True, handlelength=2
        )
        ax.add_artist(leg_main)

        upper_tol = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
        lower_tol = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
        ax.legend([upper_tol, lower_tol],
                  ["Upper Tolerance (110)", "Lower Tolerance (90)"],
                  title="Reference", fontsize=14, loc="upper right",
                  bbox_to_anchor=(0.98, 0.98), frameon=True, handlelength=2)

        add_cms_header(fig, run_label=args.cms_run_label)
        save_pdf_png(fig, str(paper_dir / "bas_rate_pidData"))
        plt.close(fig)

        # (A2) AS cumulative efficiency (relative to t0)
        tt_as_c_const = cummean(L_tt_as_const)
        tt_as_c_pd    = cummean(L_tt_as_pd)
        aa_as_c_const = cummean(L_aa_as_const)
        aa_as_c_pd    = cummean(L_aa_as_pd)

        colors_ad = {"ttbar": "goldenrod", "HToAATo4B": "limegreen"}
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_as, rel_to_t0(tt_as_c_const), color=colors_ad["ttbar"], **styles["Constant"],
                label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_as_c_const[0]:.2f}\%$)")
        ax.plot(time_as, rel_to_t0(aa_as_c_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
                label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_as_c_const[0]:.2f}\%$)")
        ax.plot(time_as, rel_to_t0(tt_as_c_pd), color=colors_ad["ttbar"], **styles["PD"],
                label=fr"PD Controller, ttbar ($\epsilon[t_0]={tt_as_c_pd[0]:.2f}\%$)")
        ax.plot(time_as, rel_to_t0(aa_as_c_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
                label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={aa_as_c_pd[0]:.2f}\%$)")

        ax.set_xlabel("Time (Fraction of Run)", loc="center")
        ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(0.98, 1.5)
        ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
        add_cms_header(fig, run_label=args.cms_run_label)
        save_pdf_png(fig, str(paper_dir / "sas_rate_pidData2data"))
        plt.close(fig)

        # (A3) AS local efficiency (relative to t0)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_as, rel_to_t0(L_tt_as_const), color=colors_ad["ttbar"], **styles["Constant"],
                label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_as_const[0]:.2f}\%$)")
        ax.plot(time_as, rel_to_t0(L_aa_as_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
                label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_as_const[0]:.2f}\%$)")
        ax.plot(time_as, rel_to_t0(L_tt_as_pd), color=colors_ad["ttbar"], **styles["PD"],
                label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_as_pd[0]:.2f}\%$)")
        ax.plot(time_as, rel_to_t0(L_aa_as_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
                label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_as_pd[0]:.2f}\%$)")

        ax.set_xlabel("Time (Fraction of Run)", loc="center")
        ax.set_ylabel("Relative Efficiency", loc="center")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(0.9, 1.7)
        ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
        add_cms_header(fig, run_label=args.cms_run_label)
        save_pdf_png(fig, str(paper_dir / "L_sas_rate_pidData2data"))
        plt.close(fig)

    print("\nSaved to:", paper_dir)
    for p in sorted(paper_dir.glob("*.pdf")):
        print(" -", p.name)

if __name__ == "__main__":
    main()