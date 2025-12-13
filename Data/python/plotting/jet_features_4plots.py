
import matplotlib.pyplot as plt
import numpy as np
import h5py
import hdf5plugin
import mplhep as hep

# --- Stile CMS-like ---
hep.style.use("CMS")

# --- CMS header, with configurable positions ---
def add_cms_header(fig, left_x=0.20, right_x=0.94, y=0.99):
    """
    Add 'CMS Open Data' on the left and 'Run 283408' on the right
    in figure coordinates.
    """
    # left label
    fig.text(
        left_x, y, "CMS Open Data",
        ha="left", va="top",
        fontweight="bold", fontsize=24
    )
    # right label
    fig.text(
        right_x, y, "Run 283408",
        ha="right", va="top",
        fontsize=24
    )


def sort_obj(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index,:,:]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index,:,:] = sorted_array

    return data_

def plot_normed_to_max(ax, arrays, labels, colors, bins, xlabel):
    hists = [np.histogram(a, bins=bins, density=True)[0] for a in arrays]
    ymax = max((h.max() for h in hists if h.size), default=1.0)
    for a, lab, col in zip(arrays, labels, colors):
        y, edges = np.histogram(a, bins=bins, density=True)
        ax.stairs(y, edges, label=lab, color=col, linewidth=1.8)
    #ax.set_xlim(bins[0], bins[-1])
    #ax.set_ylim(0, 1.05)
    #ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, loc='center')
    ax.set_ylabel("Density",loc='center')#, fontsize=22)
    ax.legend( frameon=True)

def plot_step_hist(ax, arr, bins, label, color, density=True):
    y, edges = np.histogram(arr, bins=bins, density=density)
    ax.stairs(y, edges, label=label, color=color, linewidth=1.8)

def process_h5_file2(input_filename, thin_factor=1, max_events=None):
    with h5py.File(input_filename, 'r') as h5_file:
        n_events_total = h5_file['j0Eta'].shape[0]

        # --- choose how many events to use ---
        if max_events is not None:
            n_events = min(n_events_total, max_events)
        else:
            n_events = n_events_total

        # we take 1 event every 'thin_factor'
        indices = np.arange(0, n_events, thin_factor, dtype=int)
        n_selected = len(indices)

        n_jets = 8
        n_features = 4
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)

        for i in range(n_jets):
            eta_arr = h5_file[f'j{i}Eta'][indices] + 5.0
            phi_arr = h5_file[f'j{i}Phi'][indices] + np.pi
            pt_arr  = h5_file[f'j{i}Pt'][indices]

            data_array[:, i, 0] = eta_arr
            data_array[:, i, 1] = phi_arr
            data_array[:, i, 2] = pt_arr

        npvsGood_smr1_values = h5_file['PV_npvsGood'][indices]

        sorted_data_array = sort_obj(data_array)

        # --- compute HT in a vectorized way (no Python double loop) ---
        eta = sorted_data_array[:, :, 0] - 5.0  # undo shift
        pt  = sorted_data_array[:, :, 2]

        good_jets = (pt > 20) & (np.abs(eta) < 2.5)

        Ht_values = (pt * good_jets).sum(axis=1)

        # mask jets that fail the selection
        bad_jets = ~good_jets
        sorted_data_array[:, :, 2][bad_jets] = 0.0
        sorted_data_array[:, :, 0][bad_jets] = -1.0
        sorted_data_array[:, :, 1][bad_jets] = -1.0

        # remove npv == 0
        non_zero_mask = npvsGood_smr1_values > 0
        sorted_data_array = sorted_data_array[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]

        # put npvs in last column
        sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]

        return sorted_data_array, Ht_values

def process_h5_file(input_filename):
    
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 4  
        

        #selected_indices = list(range(0, n_events, 1000)) #scommentato
        #n_selected = len(selected_indices) #scommentato
        n_selected = n_events
        
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)
        
        # Fill the array with the datats
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:]+ 5  # Eta #[:] invece di selected incidices anche nelle prossime 2 righe
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:]+ np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt
        

        npvsGood_smr1_values = h5_file['PV_npvsGood'][:]#_smr1
        #Ht_values = h5_file['ht'][:]
        # Calcolo manuale di HT con selezioni su pt ed eta
        sorted_data_array = sort_obj(data_array)

        Ht_values = np.zeros(n_selected)  # <- make sure this is before the for loop

        for i in range(n_selected):
            ht = 0
            for j in range(n_jets):
                pt = sorted_data_array[i, j, 2]
                eta = sorted_data_array[i, j, 0] - 5  # Undo shift
                if pt > 20 and abs(eta) < 2.5:
                    ht += pt
                else:
                    # Mask bad jets
                    sorted_data_array[i, j, 2] = 0.0
                    sorted_data_array[i, j, 0] = -1
                    sorted_data_array[i, j, 1] = -1
            Ht_values[i] = ht
  
        
        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values > 0  
        sorted_data_array = sorted_data_array[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]

        
        # Add npvsGood_smr1 values to the last column (time column)
        sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]

        
        zero_pt_mask = sorted_data_array[:, :, 2] == 0  # Identify where pt == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        


        non_zero_ht_mask = Ht_values > 0

        # normalize the column 
        #sorted_data_array[:, :, 2][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        print(sorted_data_array[0,:,2])

        for i in range(sorted_data_array.shape[0]):
            if(sorted_data_array[i, 0, 3] <= 0):
                print(sorted_data_array[i, 0, 3])

        return sorted_data_array, Ht_values


def compute_missing_ht(jets):
    px = np.sum(jets[:, :, 2] * np.cos(jets[:, :, 1]), axis=1)  # pt * cos(phi)
    py = np.sum(jets[:, :, 2] * np.sin(jets[:, :, 1]), axis=1)  # pt * sin(phi)
    return np.sqrt(px**2 + py**2)

# Load datasets
mc_bkg_jets, mc_bkg_ht = process_h5_file("new_Data/minimumbias_2016_1.h5")
mc_ttbar_jets, mc_ttbar_ht = process_h5_file("new_Data/ttbar_2016_1.h5")
mc_aa_jets, mc_aa_ht = process_h5_file("new_Data/HToAATo4B.h5")
mc_data_jets, mc_data_ht = process_h5_file("new_Data/data_Run_2016_283408_longest.h5")


# FAST MODE per prova label
#mc_bkg_jets, mc_bkg_ht   = process_h5_file("new_Data/minimumbias_2016_1.h5",thin_factor=1000, max_events=20000)
#mc_ttbar_jets, mc_ttbar_ht = process_h5_file("new_Data/ttbar_2016_1.h5",thin_factor=1000, max_events=20000)
#mc_aa_jets, mc_aa_ht     = process_h5_file("new_Data/HToAATo4B.h5",thin_factor=1000, max_events=20000)
#mc_data_jets, mc_data_ht = process_h5_file("new_Data/data_Run_2016_283408_longest.h5",thin_factor=1000, max_events=20000)

print('all data loaded')

# Compute missing HT
mc_bkg_missing_ht = compute_missing_ht(mc_bkg_jets)
mc_ttbar_missing_ht = compute_missing_ht(mc_ttbar_jets)
mc_aa_missing_ht = compute_missing_ht(mc_aa_jets)
mc_data_missing_ht = compute_missing_ht(mc_data_jets)

print('missing Ht calculated')

# Compute number of jets per event (jets with pt > 0)
mc_bkg_njets = np.sum(mc_bkg_jets[:, :, 2] > 0, axis=1)
mc_ttbar_njets = np.sum(mc_ttbar_jets[:, :, 2] > 0, axis=1)
mc_aa_njets = np.sum(mc_aa_jets[:, :, 2] > 0, axis=1)
mc_data_njets = np.sum(mc_data_jets[:, :, 2] > 0, axis=1)

print('njets done')


# Create masks for jets with pT > 0
bkg_mask   = mc_bkg_jets[:, :, 2] > 0
ttbar_mask = mc_ttbar_jets[:, :, 2] > 0
aa_mask    = mc_aa_jets[:, :, 2] > 0
data_mask  = mc_data_jets[:, :, 2] > 0

# Extract and shift pT, eta, phi (only jets with pT > 0)
mc_bkg_pt  = mc_bkg_jets[:, :, 2][bkg_mask]
#mc_bkg_eta = mc_bkg_jets[:, :, 0][bkg_mask] - 5.0
#mc_bkg_phi = mc_bkg_jets[:, :, 1][bkg_mask] - np.pi

print('bkg done')

mc_ttbar_pt  = mc_ttbar_jets[:, :, 2][ttbar_mask]
#mc_ttbar_eta = mc_ttbar_jets[:, :, 0][ttbar_mask] - 5.0
#mc_ttbar_phi = mc_ttbar_jets[:, :, 1][ttbar_mask] - np.pi

print('ttbar done')

mc_aa_pt  = mc_aa_jets[:, :, 2][aa_mask]
#mc_aa_eta = mc_aa_jets[:, :, 0][aa_mask] - 5.0
#mc_aa_phi = mc_aa_jets[:, :, 1][aa_mask] - np.pi

print('aa done')

mc_data_pt  = mc_data_jets[:, :, 2][data_mask]
#mc_data_eta = mc_data_jets[:, :, 0][data_mask] - 5.0
#mc_data_phi = mc_data_jets[:, :, 1][data_mask] - np.pi

print('data done')
# ===================== PLOTTING (separato per subfigure) =====================

# Bins + colori
ht_bins  = np.linspace(0, 700, 16)
mht_bins = np.linspace(0,  175, 8)
pt_bins  = np.linspace(0,  200, 10)
nj_bins  = np.arange(-0.5, 8.5 + 1, 1.0)  # bin centrati su 0..8

COLORS = {
    "data":   "firebrick",   # rosso scuro
    "minbias":"royalblue",   # blu saturo
    "ttbar":  "goldenrod",   # giallo scuro
    "hToAA":  "seagreen"     # verde scuro
}

labels = ["2016 Zerobias Data", "MinBias", "TTbar", "HToAATo4B"]
colors = [COLORS["data"], COLORS["minbias"], COLORS["ttbar"], COLORS["hToAA"]]

# --- 1) HT Distribution ---
fig, ax = plt.subplots(figsize=(7,6))
add_cms_header(fig, left_x=0.22, right_x=0.94, y=0.99)
plot_normed_to_max(
    ax,
    [mc_data_ht, mc_bkg_ht, mc_ttbar_ht, mc_aa_ht],
    labels, colors, ht_bins,
   # "HT Distribution", 
   r"$H_T$ [GeV]"
)
#ax.set_xlim(0, 700)
#ax.set_ylim(0, 0.02)
ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("HT_distribution.pdf")
plt.close()

# --- 2) Number of Jets per Event ---
fig, ax = plt.subplots(figsize=(7,6))
add_cms_header(fig, left_x=0.18, right_x=0.94, y=0.99)
plot_step_hist(ax, mc_data_njets,  nj_bins, "2016 Zerobias Data", COLORS["data"])
plot_step_hist(ax, mc_bkg_njets,   nj_bins, "MinBias",   COLORS["minbias"])
plot_step_hist(ax, mc_ttbar_njets, nj_bins, "TTbar",     COLORS["ttbar"])
plot_step_hist(ax, mc_aa_njets,    nj_bins, "HToAATo4B", COLORS["hToAA"])
#ax.set_title("Number of Jets per Event", fontsize=22)
ax.set_xlabel("Number of Jets",loc='center')#, fontsize=22)
ax.set_ylabel("Density",loc='center')#, fontsize=22)
ax.set_xticks(range(0, 9))
ax.legend(frameon=True)
ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("Njets_distribution.pdf")
plt.close()

# --- 3) Missing HT ---
fig, ax = plt.subplots(figsize=(7,6))
add_cms_header(fig, left_x=0.22, right_x=0.94, y=0.99)
plot_normed_to_max(
    ax,
    [mc_data_missing_ht, mc_bkg_missing_ht, mc_ttbar_missing_ht, mc_aa_missing_ht],
    labels, colors, mht_bins,
    #r"Missing $H_T$ Distribution", 
    r"Missing $H_T$ [GeV]"
)
#ax.set_xlim(0, 175)
#ax.set_ylim(0, 0.04)
ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("MissingHT_distribution.pdf")
plt.close()

# --- 4) Jet pT ---
fig, ax = plt.subplots(figsize=(7,6))
add_cms_header(fig, left_x=0.22, right_x=0.94, y=0.99)
plot_normed_to_max(
    ax,
    [mc_data_pt, mc_bkg_pt, mc_ttbar_pt, mc_aa_pt],
    labels, colors, pt_bins,
  #  r"Jet $p_T$ Distribution",
     r"$p_T$ [GeV]"
)
ax.set_xlim(-10, 220)
#ax.set_ylim(0, 0.035)
#ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("jetPt_distribution.pdf")
plt.close()

# =================== FINE PLOTTING ===================


