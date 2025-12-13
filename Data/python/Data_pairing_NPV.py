import numpy as np
import h5py

def read_h5_data(file_path):
    """Read datasets from HDF5 input file (both score01 and score04)."""
    with h5py.File(file_path, 'r') as h5_file:
        # Background
        data_ht      = h5_file['mc_bkg_ht'][:]
        data_scores01 = h5_file['mc_bkg_score01'][:]
        data_scores04 = h5_file['mc_bkg_score04'][:]   # NEW
        data_npvs    = h5_file['mc_bkg_Npv'][:]
        data_njets   = h5_file['mc_bkg_njets'][:]

        # TTbar
        tt_ht       = h5_file['mc_tt_ht'][:]
        tt_scores01 = h5_file['mc_tt_score01'][:]
        tt_scores04 = h5_file['mc_tt_score04'][:]      # NEW
        tt_npvs     = h5_file['tt_Npv'][:]
        tt_njets    = h5_file['mc_tt_njets'][:]

        # H→AA→4b
        aa_ht       = h5_file['mc_aa_ht'][:]
        aa_scores01 = h5_file['mc_aa_score01'][:]
        aa_scores04 = h5_file['mc_aa_score04'][:]      # NEW
        aa_npvs     = h5_file['aa_Npv'][:]
        aa_njets    = h5_file['mc_aa_njets'][:]

    return (data_ht, data_scores01, data_scores04, data_npvs, data_njets,
            tt_ht, tt_scores01, tt_scores04, tt_npvs, tt_njets,
            aa_ht, aa_scores01, aa_scores04, aa_npvs, aa_njets)

def match_to_data(data_npvs, sig_ht, sig_scores, sig_npvs, sig_njets):
    """Match signal events to background distribution of NPV."""
    sorted_indices = np.argsort(sig_npvs)
    sig_ht     = sig_ht[sorted_indices]
    sig_scores = sig_scores[sorted_indices]
    sig_npvs   = sig_npvs[sorted_indices]
    sig_njets  = sig_njets[sorted_indices]

    matched_ht = []
    matched_scores = []
    matched_npvs = []
    matched_njets = []

    for npv in data_npvs:
        idx_left = np.searchsorted(sig_npvs, npv, side='left')
        idx_right = np.searchsorted(sig_npvs, npv, side='right')

        if idx_left == len(sig_npvs):
            idx = len(sig_npvs) - 1
        elif idx_left == idx_right:
            idx = idx_left
        else:
            idx = np.random.randint(idx_left, idx_right)

        matched_ht.append(sig_ht[idx])
        matched_scores.append(sig_scores[idx])
        matched_npvs.append(sig_npvs[idx])
        matched_njets.append(sig_njets[idx])

    return (np.array(matched_ht),
            np.array(matched_scores),
            np.array(matched_npvs),
            np.array(matched_njets))

# === CONFIG ===
input_file = "new_Data/Trigger_food.h5"
output_file = "new_Data/Matched_data_2016.h5"

# === READ INPUT ===
(data_ht, data_scores01, data_scores04, data_npvs, data_njets,
 tt_ht, tt_scores01, tt_scores04, tt_npvs, tt_njets,
 aa_ht, aa_scores01, aa_scores04, aa_npvs, aa_njets) = read_h5_data(input_file)

# === PAIRING (both score01 and score04) ===
matched_tt_ht, matched_tt_scores01, matched_tt_npvs, matched_tt_njets = match_to_data(data_npvs, tt_ht, tt_scores01, tt_npvs, tt_njets)
_,              matched_tt_scores04, _,                  _            = match_to_data(data_npvs, tt_ht, tt_scores04, tt_npvs, tt_njets)

matched_aa_ht, matched_aa_scores01, matched_aa_npvs, matched_aa_njets = match_to_data(data_npvs, aa_ht, aa_scores01, aa_npvs, aa_njets)
_,              matched_aa_scores04, _,                  _            = match_to_data(data_npvs, aa_ht, aa_scores04, aa_npvs, aa_njets)

# === SAVE OUTPUT ===
with h5py.File(output_file, 'w') as h5_out:
    # Background
    h5_out.create_dataset("data_ht", data=data_ht)
    h5_out.create_dataset("data_scores01", data=data_scores01)
    h5_out.create_dataset("data_scores04", data=data_scores04)
    h5_out.create_dataset("data_Npv", data=data_npvs)
    h5_out.create_dataset("data_njets", data=data_njets)

    # TTbar
    h5_out.create_dataset("matched_tt_ht", data=matched_tt_ht)
    h5_out.create_dataset("matched_tt_scores01", data=matched_tt_scores01)
    h5_out.create_dataset("matched_tt_scores04", data=matched_tt_scores04)
    h5_out.create_dataset("matched_tt_npvs", data=matched_tt_npvs)
    h5_out.create_dataset("matched_tt_njets", data=matched_tt_njets)

    # H→AA→4b
    h5_out.create_dataset("matched_aa_ht", data=matched_aa_ht)
    h5_out.create_dataset("matched_aa_scores01", data=matched_aa_scores01)
    h5_out.create_dataset("matched_aa_scores04", data=matched_aa_scores04)
    h5_out.create_dataset("matched_aa_npvs", data=matched_aa_npvs)
    h5_out.create_dataset("matched_aa_njets", data=matched_aa_njets)

print(f"Pairing completed and saved to: {output_file}")
