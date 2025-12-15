#!/usr/bin/env python3
"""
demo_single_trigger_dqn_htas.py

SingleTrigger: 
Constant vs PD vs DQN
- HT trigger: accept = (HT >= Ht_cut)
- AS trigger: accept = (AS >= AS_cut)

We train two independent DQNs:
  (1) DQN_HT controls Ht_cut using HT-only rates
  (2) DQN_AS controls AS_cut using AS-only rates

Outputs:

HT trigger outputs:
  - bht_rate_pidData_dqn.pdf/.png          (HT background rate [kHz])
  - ht_cut_pidData_dqn.pdf/.png            (Ht_cut evolution)
  - sht_rate_pidData2data_dqn.pdf/.png     (cumulative signal eff, relative to t0)
  - L_sht_rate_pidData2data_dqn.pdf/.png   (local signal eff, relative to t0)
  - dqn_loss_ht.pdf/.png                   (HT DQN loss)

AS trigger outputs:
  - bas_rate_pidData_dqn.pdf/.png          (AS background rate [kHz])
  - as_cut_pidData_dqn.pdf/.png            (AS_cut evolution)
  - sas_rate_pidData2data_dqn.pdf/.png     (cumulative signal eff, relative to t0)
  - L_sas_rate_pidData2data_dqn.pdf/.png   (local signal eff, relative to t0)
  - dqn_loss_as.pdf/.png                   (AS DQN loss)

"""

import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # noqa: F401

from pathlib import Path
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from RL.utils import cummean, rel_to_t0, add_cms_header, save_pdf_png, plot_rate_with_tolerance
from RL.dqn_agent import DQNAgent, make_obs, shield_delta, compute_reward, DQNConfig

# ------------------------- reproducibility -------------------------
SEED = 20251213
random.seed(SEED)
np.random.seed(SEED)

# ------------------------- H5 reading (the unified reader) -------------------------
def _first_present(h5_keys, candidates):
    for k in candidates:
        if k in h5_keys:
            return k
    return None

def _collect_datasets(h5):
    """Return dict: dataset_path -> h5py.Dataset (supports nested groups)."""
    dsets = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            dsets[name] = obj
    h5.visititems(visitor)
    return dsets

def _basename(x: str) -> str:
    return x.split("/")[-1]

def _find_key(keys, candidates):
    """
    Find a dataset key by trying:
      1) exact match
      2) basename match (for nested datasets)
    """
    for c in candidates:
        if c in keys:
            return c
    # basename match
    for c in candidates:
        for k in keys:
            if _basename(k) == c:
                return k
    return None

# def read_any_h5(path, score_dim_hint=2):
#     """
#     Returns unified dict keys:
#       Bht, Bnpv, Bas1, Bas2, Bas4
#       Tht, Tnpv, Tas1, Tas2, Tas4
#       Aht, Anpv, Aas1, Aas2, Aas4
#       meta['matched_by_index']
#     """
#     with h5py.File(path, "r") as h5:
#         dsets = _collect_datasets(h5)
#         keys = set(dsets.keys())

#         # ---------- detect Matched_data_* ----------
#         # allow nested names too (basename matching)
#         is_matched = (
#             _find_key(keys, ["data_ht"]) is not None and
#             _find_key(keys, ["data_Npv", "data_npv"]) is not None and
#             _find_key(keys, ["matched_tt_ht"]) is not None and
#             _find_key(keys, ["matched_aa_ht"]) is not None
#         )

#         if is_matched:
#             k_Bht  = _find_key(keys, ["data_ht"])
#             k_Bnpv = _find_key(keys, ["data_Npv", "data_npv"])
#             Bht  = dsets[k_Bht][:]
#             Bnpv = dsets[k_Bnpv][:]

#             # scores can appear as score/score02/scores02 etc.
#             Bas_k = _find_key(keys, [
#                 "data_score",
#                 "data_score02", "data_scores02", "data_score2", "data_scores2",
#                 "data_score01", "data_scores01", "data_score1", "data_scores1",
#                 "data_score04", "data_scores04", "data_score4", "data_scores4",
#             ])
#             Tas_k = _find_key(keys, [
#                 "matched_tt_score",
#                 "matched_tt_score02", "matched_tt_scores02", "matched_tt_score2", "matched_tt_scores2",
#                 "matched_tt_score01", "matched_tt_scores01", "matched_tt_score1", "matched_tt_scores1",
#                 "matched_tt_score04", "matched_tt_scores04", "matched_tt_score4", "matched_tt_scores4",
#             ])
#             Aas_k = _find_key(keys, [
#                 "matched_aa_score",
#                 "matched_aa_score02", "matched_aa_scores02", "matched_aa_score2", "matched_aa_scores2",
#                 "matched_aa_score01", "matched_aa_scores01", "matched_aa_score1", "matched_aa_scores1",
#                 "matched_aa_score04", "matched_aa_scores04", "matched_aa_score4", "matched_aa_scores4",
#             ])

#             Bas = dsets[Bas_k][:] if Bas_k else None
#             Tas = dsets[Tas_k][:] if Tas_k else None
#             Aas = dsets[Aas_k][:] if Aas_k else None

#             Tht = dsets[_find_key(keys, ["matched_tt_ht"])][:]
#             Aht = dsets[_find_key(keys, ["matched_aa_ht"])][:]

#             Tnpv_k = _find_key(keys, ["matched_tt_npvs", "matched_tt_Npv", "matched_tt_npv"])
#             Anpv_k = _find_key(keys, ["matched_aa_npvs", "matched_aa_Npv", "matched_aa_npv"])
#             Tnpv = dsets[Tnpv_k][:] if Tnpv_k else np.zeros_like(Tht, dtype=np.float32)
#             Anpv = dsets[Anpv_k][:] if Anpv_k else np.zeros_like(Aht, dtype=np.float32)

#             return dict(
#                 Bht=Bht, Bnpv=Bnpv,
#                 Bas1=Bas, Bas2=Bas, Bas4=Bas,
#                 Tht=Tht, Tnpv=Tnpv,
#                 Tas1=Tas, Tas2=Tas, Tas4=Tas,
#                 Aht=Aht, Anpv=Anpv,
#                 Aas1=Aas, Aas2=Aas, Aas4=Aas,
#                 meta=dict(matched_by_index=True),
#             )

#         # ---------- MC Trigger_food_* ----------
#         # Try many common variants + allow nested groups
#         def req(cands, label):
#             k = _find_key(keys, cands)
#             if k is None:
#                 # show a short list of available basenames for debugging
#                 avail = sorted({_basename(x) for x in keys})
#                 raise KeyError(f"[read_any_h5] Missing {label}. Tried {cands}. "
#                                f"Available dataset basenames (sample): {avail[:80]}")
#             return dsets[k][:]

#         def opt(cands):
#             k = _find_key(keys, cands)
#             return dsets[k][:] if k is not None else None

#         Bht  = req(["mc_bkg_ht", "bkg_ht", "mc_bkg_HT", "Bht", "bht", "mc/bkg/ht", "mc_bkg/ht"], "BHT (background HT)")
#         Bnpv = req(["mc_bkg_Npv", "mc_bkg_npv", "bkg_Npv", "bkg_npv", "Bnpv", "bnpv", "mc/bkg/npv", "mc_bkg/npv"], "Bnpv (background NPV)")

#         Tht  = req(["mc_tt_ht", "tt_ht", "mc_tt_HT", "Tht", "ttht", "mc/tt/ht", "mc_tt/ht"], "THT (tt HT)")
#         Tnpv = opt(["mc_tt_Npv", "mc_tt_npv", "tt_Npv", "tt_npv", "Tnpv", "mc/tt/npv", "mc_tt/npv"])
#         if Tnpv is None:
#             Tnpv = np.zeros_like(Tht, dtype=np.float32)

#         Aht  = req(["mc_aa_ht", "aa_ht", "mc_aa_HT", "Aht", "aaht", "mc/aa/ht", "mc_aa/ht"], "AHT (aa HT)")
#         Anpv = opt(["mc_aa_Npv", "mc_aa_npv", "aa_Npv", "aa_npv", "Anpv", "mc/aa/npv", "mc_aa/npv"])
#         if Anpv is None:
#             Anpv = np.zeros_like(Aht, dtype=np.float32)

#         # scores: support scoreXX, scoresXX, and non-zero-padded variants
#         def score(prefix, dim):
#             dim2 = str(int(dim))           # "2"
#             dim02 = f"{int(dim):02d}"      # "02"
#             return opt([
#                 f"{prefix}_score{dim02}", f"{prefix}_scores{dim02}",
#                 f"{prefix}_score{dim2}",  f"{prefix}_scores{dim2}",
#                 f"{prefix}/score{dim02}", f"{prefix}/scores{dim02}",
#                 f"{prefix}/score{dim2}",  f"{prefix}/scores{dim2}",
#             ])

#         # prefer explicit 01/04; otherwise use hint as "dim2 slot"
#         Bas1 = score("mc_bkg", 1) or score("bkg", 1) or score("mc/bkg", 1)
#         Bas4 = score("mc_bkg", 4) or score("bkg", 4) or score("mc/bkg", 4)
#         Tas1 = score("mc_tt",  1) or score("tt",  1) or score("mc/tt",  1)
#         Tas4 = score("mc_tt",  4) or score("tt",  4) or score("mc/tt",  4)
#         Aas1 = score("mc_aa",  1) or score("aa",  1) or score("mc/aa",  1)
#         Aas4 = score("mc_aa",  4) or score("aa",  4) or score("mc/aa",  4)

#         # If 01/04 aren't present, fill "dim2 slot" from score_dim_hint (default 2)
#         hint = int(score_dim_hint)
#         Bas_hint = score("mc_bkg", hint) or score("bkg", hint) or score("mc/bkg", hint)
#         Tas_hint = score("mc_tt",  hint) or score("tt",  hint) or score("mc/tt",  hint)
#         Aas_hint = score("mc_aa",  hint) or score("aa",  hint) or score("mc/aa",  hint)

#         # Assemble Bas2/Tas2/Aas2 from hinted dim (or fall back to Bas1 if that's all we have)
#         Bas2 = Bas_hint if Bas_hint is not None else Bas1
#         Tas2 = Tas_hint if Tas_hint is not None else Tas1
#         Aas2 = Aas_hint if Aas_hint is not None else Aas1

#         # if still nothing, hard fail with a good message
#         if (Bas1 is None and Bas2 is None and Bas4 is None):
#             avail = sorted({_basename(x) for x in keys})
#             raise KeyError("[read_any_h5] Could not find ANY background score dataset "
#                            "(tried mc_bkg_scoreXX / bkg_scoreXX variants). "
#                            f"Available basenames (sample): {avail[:120]}")

#         return dict(
#             Bht=Bht, Bnpv=Bnpv,
#             Bas1=Bas1, Bas2=Bas2, Bas4=Bas4,
#             Tht=Tht, Tnpv=Tnpv,
#             Tas1=Tas1, Tas2=Tas2, Tas4=Tas4,
#             Aht=Aht, Anpv=Anpv,
#             Aas1=Aas1, Aas2=Aas2, Aas4=Aas4,
#             meta=dict(matched_by_index=False),
#         )
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
    ap.add_argument("--input", default="Data/Matched_data_2016_dim2.h5",
                    help="Matched_data_*.h5 (data) or Trigger_food_*.h5 (MC)")
    ap.add_argument("--outdir", default="outputs/demo_sing_dqn_separate", help="output dir")
    ap.add_argument("--cms-run-label", default="Run 283408")

    ap.add_argument("--chunk-size", type=int, default=20000,
                    help="Chunk size (Matched_data default 20000; MC default maybe 50000).")
    ap.add_argument("--start-event", type=int, default=200000,
                    help="Start event index (Matched_data default 200000).")
    ap.add_argument("--score-dim-hint", type=int, default=2,
                    help="If file has only scoreXX, use this dim (e.g. 2 -> score02).")
    ap.add_argument("--as-dim", type=int, default=2, choices=[1, 4, 2],
                    help="Which AS dimension to use (1->score01, 4->score04).")

    ap.add_argument("--ht-deltas", type=str, default="-2,-1,0,1,2",
                    help="HT DQN deltas (in HT cut units, like your HT script).")
    ap.add_argument("--as-deltas", type=str, default="-1,-0.5,0,0.5,1",
                    help="AS DQN delta multipliers.")
    ap.add_argument("--as-step", type=float, default=0.02,
                    help="AS step: final delta = as_delta * as_step (tune to your AS scale).")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5(args.input, score_dim_hint=args.score_dim_hint)
    matched_by_index = bool(d["meta"].get("matched_by_index", False))

    Bht, Bnpv = d["Bht"], d["Bnpv"]
    Tht, Tnpv = d["Tht"], d["Tnpv"]
    Aht, Anpv = d["Aht"], d["Anpv"]

    # Pick AS dim
    if args.as_dim == 1:
        Bas, Tas, Aas = d["Bas1"], d["Tas1"], d["Aas1"]
    elif args.as_dim == 2:
        Bas, Tas, Aas = d["Bas2"], d["Tas2"], d["Aas2"]
    else:  # 4
        Bas, Tas, Aas = d["Bas4"], d["Tas4"], d["Aas4"]

    if Bas is None or Tas is None or Aas is None:
        raise SystemExit("AS arrays missing for requested --as-dim. Try --as-dim 1 and/or --score-dim-hint.")

    N = len(Bht)
    chunk_size = int(args.chunk_size)
    start_event = int(args.start_event)

    # Align start_event to chunk boundary
    start_event = max(0, (start_event // chunk_size) * chunk_size)
    if start_event + chunk_size > N:
        start_event = max(0, ((N - chunk_size) // chunk_size) * chunk_size)

    # Fixed cuts from a reference window (mimic your Data_SingleTrigger.py)
    win_lo = min(start_event, N - 1)
    win_hi = min(start_event + 10000, N)
    if win_hi - win_lo < 1000:
        win_lo = 0
        win_hi = min(10000, N)

    fixed_Ht_cut = float(np.percentile(Bht[win_lo:win_hi], 99.75))
    fixed_AS_cut = float(np.percentile(Bas[win_lo:win_hi], 99.75))

    # Clip ranges
    ht_lo = float(np.percentile(Bht[start_event:], 95.0))
    ht_hi = float(np.percentile(Bht[start_event:], 99.99))
    ht_mid = 0.5 * (ht_lo + ht_hi)
    ht_span = max(1.0, ht_hi - ht_lo)

    as_lo = float(np.percentile(Bas[start_event:], 95.0))
    as_hi = float(np.percentile(Bas[start_event:], 99.99))
    as_mid = 0.5 * (as_lo + as_hi)
    as_span = max(1e-6, as_hi - as_lo)

    print(f"[INFO] matched_by_index={matched_by_index} N={N} chunk={chunk_size} start_event={start_event}")
    print(f"[HT] fixed={fixed_Ht_cut:.3f} clip=({ht_lo:.3f},{ht_hi:.3f}) window=[{win_lo}:{win_hi}]")
    print(f"[AS dim={args.as_dim}] fixed={fixed_AS_cut:.6f} clip=({as_lo:.6f},{as_hi:.6f}) as_step={args.as_step}")

    # ------------------------- init cuts -------------------------
    # HT
    Ht_cut_pd  = fixed_Ht_cut
    Ht_cut_dqn = fixed_Ht_cut
    pre_ht_err = 0.0

    # AS
    AS_cut_pd  = fixed_AS_cut
    AS_cut_dqn = fixed_AS_cut
    pre_as_err = 0.0

    # ------------------------- DQN configs -------------------------
    target = 0.25  # %
    tol = 0.03     # %
    alpha = 0.2    # signal bonus
    beta  = 0.02   # move penalty

    HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
    AS_DELTAS = np.array([float(x) for x in args.as_deltas.split(",")], dtype=np.float32)
    AS_STEP = float(args.as_step)

    MAX_DELTA_HT = float(np.max(np.abs(HT_DELTAS)))
    MAX_DELTA_AS = float(np.max(np.abs(AS_DELTAS))) * AS_STEP

    cfg = DQNConfig(lr=5e-4, gamma=0.95, batch_size=128, target_update=200)
    agent_ht = DQNAgent(obs_dim=3, n_actions=len(HT_DELTAS), cfg=cfg, seed = SEED)
    agent_as = DQNAgent(obs_dim=3, n_actions=len(AS_DELTAS), cfg=cfg, seed = SEED)

    # state trackers (HT)
    prev_obs_ht = None
    prev_act_ht = None
    prev_bg_ht = None
    last_dht = 0.0
    losses_ht = []

    # state trackers (AS)
    prev_obs_as = None
    prev_act_as = None
    prev_bg_as = None
    last_das = 0.0
    losses_as = []

    # ------------------------- logs (HT) -------------------------
    R1_ht, R2_ht, R3_ht = [], [], []                  # background % (const, PD, DQN)
    Ht_pd_hist, Ht_dqn_hist = [], []
    L_tt_ht_const, L_tt_ht_pd, L_tt_ht_dqn = [], [], []
    L_aa_ht_const, L_aa_ht_pd, L_aa_ht_dqn = [], [], []

    # ------------------------- logs (AS) -------------------------
    R1_as, R2_as, R3_as = [], [], []                  # background % (const, PD, DQN)
    As_pd_hist, As_dqn_hist = [], []
    L_tt_as_const, L_tt_as_pd, L_tt_as_dqn = [], [], []
    L_aa_as_const, L_aa_as_pd, L_aa_as_dqn = [], [], []

    # ------------------------- batching loop -------------------------
    batch_starts = list(range(start_event, N, chunk_size))

    for t, I in enumerate(batch_starts):
        idx = np.arange(I, min(I + chunk_size, N))

        bht = Bht[idx]
        bas = Bas[idx] if idx[-1] < len(Bas) else Bas[idx[idx < len(Bas)]]
        bnpv = Bnpv[idx]

        # ---- signals per chunk ----
        if matched_by_index:
            idx_tt = idx[idx < len(Tht)]
            idx_aa = idx[idx < len(Aht)]

            sht_tt = Tht[idx_tt]
            sas_tt = Tas[idx_tt]
            sht_aa = Aht[idx_aa]
            sas_aa = Aas[idx_aa]
        else:
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)

            sht_tt = Tht[mask_tt]
            sas_tt = Tas[mask_tt]
            sht_aa = Aht[mask_aa]
            sas_aa = Aas[mask_aa]

        # =========================================================
        # HT trigger (separate)
        # =========================================================
        bg_const_ht = Sing_Trigger(bht, fixed_Ht_cut)
        bg_pd_ht    = Sing_Trigger(bht, Ht_cut_pd)
        bg_dqn_ht   = Sing_Trigger(bht, Ht_cut_dqn)

        tt_const_ht = Sing_Trigger(sht_tt, fixed_Ht_cut)
        tt_pd_ht    = Sing_Trigger(sht_tt, Ht_cut_pd)
        tt_dqn_ht   = Sing_Trigger(sht_tt, Ht_cut_dqn)

        aa_const_ht = Sing_Trigger(sht_aa, fixed_Ht_cut)
        aa_pd_ht    = Sing_Trigger(sht_aa, Ht_cut_pd)
        aa_dqn_ht   = Sing_Trigger(sht_aa, Ht_cut_dqn)

        # PD update HT
        Ht_cut_pd, pre_ht_err = PD_controller1(bg_pd_ht, pre_ht_err, Ht_cut_pd)
        Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))

        # DQN HT update (train on previous transition, choose next delta)
        if prev_bg_ht is None:
            prev_bg_ht = bg_dqn_ht
        obs_ht = make_obs(bg_dqn_ht, prev_bg_ht, Ht_cut_dqn, ht_mid, ht_span, target)

        if (prev_obs_ht is not None) and (prev_act_ht is not None):
            reward = compute_reward(
                bg_rate=bg_dqn_ht,
                target=target,
                tol=tol,
                sig_rate_1=tt_dqn_ht,
                sig_rate_2=aa_dqn_ht,
                delta_applied=last_dht,
                max_delta=MAX_DELTA_HT,
                alpha=alpha,
                beta=beta,
            )

            agent_ht.buf.push(prev_obs_ht, prev_act_ht, reward, obs_ht, done=False)
            loss = agent_ht.train_step()
            if loss is not None:
                losses_ht.append(loss)

        eps = max(0.05, 1.0 * (0.98 ** t))
        act_ht = agent_ht.act(obs_ht, eps=eps)
        dht = float(HT_DELTAS[act_ht])

        sd = shield_delta(bg_dqn_ht, target, tol, MAX_DELTA_HT)
        if sd is not None:
            dht = float(sd)

        prev_obs_ht = obs_ht
        prev_act_ht = act_ht
        prev_bg_ht = bg_dqn_ht
        last_dht = dht

        Ht_cut_dqn = float(np.clip(Ht_cut_dqn + dht, ht_lo, ht_hi))

        # record HT logs
        R1_ht.append(bg_const_ht)
        R2_ht.append(bg_pd_ht)
        R3_ht.append(bg_dqn_ht)
        Ht_pd_hist.append(Ht_cut_pd)
        Ht_dqn_hist.append(Ht_cut_dqn)
        L_tt_ht_const.append(tt_const_ht)
        L_tt_ht_pd.append(tt_pd_ht)
        L_tt_ht_dqn.append(tt_dqn_ht)
        L_aa_ht_const.append(aa_const_ht)
        L_aa_ht_pd.append(aa_pd_ht)
        L_aa_ht_dqn.append(aa_dqn_ht)

        # =========================================================
        # AD trigger (separate)
        # =========================================================
        bg_const_as = Sing_Trigger(bas, fixed_AS_cut)
        bg_pd_as    = Sing_Trigger(bas, AS_cut_pd)
        bg_dqn_as   = Sing_Trigger(bas, AS_cut_dqn)

        tt_const_as = Sing_Trigger(sas_tt, fixed_AS_cut)
        tt_pd_as    = Sing_Trigger(sas_tt, AS_cut_pd)
        tt_dqn_as   = Sing_Trigger(sas_tt, AS_cut_dqn)

        aa_const_as = Sing_Trigger(sas_aa, fixed_AS_cut)
        aa_pd_as    = Sing_Trigger(sas_aa, AS_cut_pd)
        aa_dqn_as   = Sing_Trigger(sas_aa, AS_cut_dqn)

        # PD update AS
        AS_cut_pd, pre_as_err = PD_controller2(bg_pd_as, pre_as_err, AS_cut_pd)
        AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))

        # DQN AS update
        if prev_bg_as is None:
            prev_bg_as = bg_dqn_as
        obs_as = make_obs(bg_dqn_as, prev_bg_as, AS_cut_dqn, as_mid, as_span, target)

        if (prev_obs_as is not None) and (prev_act_as is not None):
            reward = compute_reward(
                bg_rate=bg_dqn_as,
                target=target,
                tol=tol,
                sig_rate_1=tt_dqn_as,
                sig_rate_2=aa_dqn_as,
                delta_applied=last_das,
                max_delta=MAX_DELTA_AS,
                alpha=alpha,
                beta=beta,
            )

            agent_as.buf.push(prev_obs_as, prev_act_as, reward, obs_as, done=False)
            loss = agent_as.train_step()
            if loss is not None:
                losses_as.append(loss)

        act_as = agent_as.act(obs_as, eps=eps)
        das = float(AS_DELTAS[act_as] * AS_STEP)

        sd = shield_delta(bg_dqn_as, target, tol, MAX_DELTA_AS)
        if sd is not None:
            das = float(sd)

        prev_obs_as = obs_as
        prev_act_as = act_as
        prev_bg_as = bg_dqn_as
        last_das = das

        AS_cut_dqn = float(np.clip(AS_cut_dqn + das, as_lo, as_hi))

        # record AS logs
        R1_as.append(bg_const_as)
        R2_as.append(bg_pd_as)
        R3_as.append(bg_dqn_as)
        As_pd_hist.append(AS_cut_pd)
        As_dqn_hist.append(AS_cut_dqn)
        L_tt_as_const.append(tt_const_as)
        L_tt_as_pd.append(tt_pd_as)
        L_tt_as_dqn.append(tt_dqn_as)
        L_aa_as_const.append(aa_const_as)
        L_aa_as_pd.append(aa_pd_as)
        L_aa_as_dqn.append(aa_dqn_as)

        if t % 5 == 0:
            lh = losses_ht[-1] if losses_ht else None
            la = losses_as[-1] if losses_as else None
            print(f"[batch {t:4d}] eps={eps:.3f} "
                  f"HT bg% c={bg_const_ht:.3f} pd={bg_pd_ht:.3f} dqn={bg_dqn_ht:.3f} "
                  f"| ht_cut pd={Ht_cut_pd:.1f} dqn={Ht_cut_dqn:.1f} loss={lh} "
                  f"|| AS bg% c={bg_const_as:.3f} pd={bg_pd_as:.3f} dqn={bg_dqn_as:.3f} "
                  f"| as_cut pd={AS_cut_pd:.4f} dqn={AS_cut_dqn:.4f} loss={la}")

    # ------------------------- convert + scale -------------------------
    RATE_SCALE_KHZ = 400.0
    upper_tol_khz = 0.275 * RATE_SCALE_KHZ
    lower_tol_khz = 0.225 * RATE_SCALE_KHZ

    # HT
    R1_ht = np.array(R1_ht) * RATE_SCALE_KHZ
    R2_ht = np.array(R2_ht) * RATE_SCALE_KHZ
    R3_ht = np.array(R3_ht) * RATE_SCALE_KHZ
    Ht_pd_hist = np.array(Ht_pd_hist)
    Ht_dqn_hist = np.array(Ht_dqn_hist)
    L_tt_ht_const = np.array(L_tt_ht_const)
    L_tt_ht_pd    = np.array(L_tt_ht_pd)
    L_tt_ht_dqn   = np.array(L_tt_ht_dqn)
    L_aa_ht_const = np.array(L_aa_ht_const)
    L_aa_ht_pd    = np.array(L_aa_ht_pd)
    L_aa_ht_dqn   = np.array(L_aa_ht_dqn)

    # AS
    R1_as = np.array(R1_as) * RATE_SCALE_KHZ
    R2_as = np.array(R2_as) * RATE_SCALE_KHZ
    R3_as = np.array(R3_as) * RATE_SCALE_KHZ
    As_pd_hist = np.array(As_pd_hist)
    As_dqn_hist = np.array(As_dqn_hist)
    L_tt_as_const = np.array(L_tt_as_const)
    L_tt_as_pd    = np.array(L_tt_as_pd)
    L_tt_as_dqn   = np.array(L_tt_as_dqn)
    L_aa_as_const = np.array(L_aa_as_const)
    L_aa_as_pd    = np.array(L_aa_as_pd)
    L_aa_as_dqn   = np.array(L_aa_as_dqn)

    time = np.linspace(0, 1, len(R1_ht))
    plot_rate_with_tolerance(
        time, R1_ht, R2_ht, R3_ht,
        outbase=outdir / "bht_rate_pidData_dqn",
        run_label=args.cms_run_label,
        legend_title="HT Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        # pass your functions from utils import
        add_cms_header=add_cms_header,
        save_pdf_png=save_pdf_png,
    )

    # ------------------------- common styles -------------------------
    styles = {
        "Constant": {"linestyle": "dashed", "linewidth": 2.5},
        "PD":       {"linestyle": "solid",  "linewidth": 2.0},
        "DQN":      {"linestyle": "solid",  "linewidth": 2.0},
    }

    # =========================================================
    # HT plots
    # =========================================================
    # (2) HT cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Ht_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time, Ht_dqn_hist, color="tab:purple", linewidth=2.0, label="DQN")
    ax.axhline(y=fixed_Ht_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_Ht_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Ht_cut [GeV]", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="HT Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(outdir / "ht_cut_pidData_dqn"))
    plt.close(fig)

    # (3) HT cumulative eff (relative to t0)
    tt_c_const = cummean(L_tt_ht_const)
    tt_c_pd    = cummean(L_tt_ht_pd)
    tt_c_dqn   = cummean(L_tt_ht_dqn)
    aa_c_const = cummean(L_aa_ht_const)
    aa_c_pd    = cummean(L_aa_ht_pd)
    aa_c_dqn   = cummean(L_aa_ht_dqn)

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
    save_pdf_png(fig, str(outdir / "sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # (4) HT local eff (relative to t0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(L_tt_ht_const), color=colors_ht["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_ht_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_const), color=colors_ht["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_ht_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_tt_ht_pd), color=colors_ht["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_tt_ht_dqn), color=colors_ht["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_ht_dqn[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_dqn), color=colors_ht["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_ht_dqn[0]:.2f}\%$)")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 1.6)
    ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(outdir / "L_sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # (5) HT loss
    if losses_ht:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_ht)), losses_ht, linewidth=1.5)
        ax.set_title("HT DQN training loss (SmoothL1)")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=args.cms_run_label)
        save_pdf_png(fig, str(outdir / "dqn_loss_ht"))
        plt.close(fig)

    # =========================================================
    # AD plots
    # =========================================================
    time_as = np.linspace(0, 1, len(R1_as))
    plot_rate_with_tolerance(
        time_as, R1_as, R2_as, R3_as,
        outbase=outdir / "bas_rate_pidData_dqn",
        run_label=args.cms_run_label,
        legend_title="AD Trigger",
        ylim=(60, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", linestyle="dotted", linewidth=3.0),
        pd_style=dict(color="mediumblue", linestyle="solid", linewidth=2.5),
        dqn_style=dict(color="tab:purple", linestyle="solid", linewidth=2.5),
        add_cms_header=add_cms_header,
        save_pdf_png=save_pdf_png,
    )

    # (A2) AS cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, As_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time_as, As_dqn_hist, color="tab:purple", linewidth=2.0, label="DQN")
    ax.axhline(y=fixed_AS_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_AS_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("AS_cut", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(outdir / "as_cut_pidData_dqn"))
    plt.close(fig)

    # (A3) AD cumulative eff (relative to t0)
    tt_c_const = cummean(L_tt_as_const)
    tt_c_pd    = cummean(L_tt_as_pd)
    tt_c_dqn   = cummean(L_tt_as_dqn)
    aa_c_const = cummean(L_aa_as_const)
    aa_c_pd    = cummean(L_aa_as_pd)
    aa_c_dqn   = cummean(L_aa_as_dqn)

    colors_ad = {"ttbar": "goldenrod", "HToAATo4B": "limegreen"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(tt_c_const), color=colors_ad["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_pd), color=colors_ad["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_dqn), color=colors_ad["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_dqn), color=colors_ad["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.98, 1.5)
    ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(outdir / "sas_rate_pidData2data_dqn"))
    plt.close(fig)

    # (A4) AD local eff (relative to t0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(L_tt_as_const), color=colors_ad["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_as_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_as_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_pd), color=colors_ad["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_dqn), color=colors_ad["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_as_dqn[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_dqn), color=colors_ad["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_as_dqn[0]:.2f}\%$)")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.9, 1.7)
    ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=args.cms_run_label)
    save_pdf_png(fig, str(outdir / "L_sas_rate_pidData2data_dqn"))
    plt.close(fig)

    # (A5) AS loss
    if losses_as:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_as)), losses_as, linewidth=1.5)
        ax.set_title("AD DQN training loss (SmoothL1)")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=args.cms_run_label)
        save_pdf_png(fig, str(outdir / "dqn_loss_as"))
        plt.close(fig)

    print("\nSaved to:", outdir)
    for p in sorted(outdir.glob("*.pdf")):
        print(" -", p.name)

if __name__ == "__main__":
    main()