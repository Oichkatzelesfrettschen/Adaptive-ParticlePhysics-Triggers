import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin

#import atlas_mpl_style as aplt
#aplt.use_atlas_style()

import mplhep as hep
hep.style.use("CMS")
#hep.style.use(hep.style.ROOT) 
# Or choose one of the experiment styles
#hep.style.use(hep.style.ATLAS)
# or
 # string aliases work too

def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.98):
    """
    Add 'CMS Open Data' on the left and 'Run 283408' on the right
    in figure coordinates.
    """
    fig.text(
        left_x, y, "CMS Open Data",
        ha="left", va="top",
        fontweight="bold", fontsize=24
    )
    fig.text(
        right_x, y, "Run 283408",
        ha="right", va="top",
        fontsize=24
    )

def save_subplot(fig, ax, filename, pad=0.1):
    """Save a single subplot with a little padding (in inches)."""
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    # Expand bbox by pad (fraction of inch)
    bbox = mtransforms.Bbox.from_extents(
        bbox.x0 - pad, bbox.y0 - pad,
        bbox.x1 + pad, bbox.y1 + pad
    )

    fig.savefig(filename, bbox_inches=bbox)

def plot_evolution(Ht_vals, AS_vals, real_Ht_vals, real_AS_vals, title="real_evoultion_plot(r1)"):
    plt.figure(figsize=(8, 6))

    # Plot all points
    #plt.scatter(Ht_vals, AS_vals, c=np.arange(len(Ht_vals)), marker='o', cmap="viridis", label="ideal Points",s=50)
    plt.scatter(real_Ht_vals, real_AS_vals, c=np.arange(len(real_Ht_vals)),marker='+', cmap="viridis", label="real-time Points",s=50)

    # Plot arrows showing evolution
    #for i in range(len(Ht_vals) - 1):
        #plt.quiver(Ht_vals[i], AS_vals[i], Ht_vals[i+1] - Ht_vals[i], AS_vals[i+1] - AS_vals[i], 
        # angles="xy", scale_units="xy", scale=1, color="green", alpha=0.6)

    # Highlight start and end points
    #plt.scatter(Ht_vals[0], AS_vals[0],marker='o', color="blue", s=100, label="ideal Start")
    #plt.scatter(Ht_vals[-1], AS_vals[-1],marker='o', color="red", s=100, label="ideal End")

    #plt.scatter(real_Ht_vals[0], real_AS_vals[0],marker='+', color="blue", s=100, label="real Start")
    #plt.scatter(real_Ht_vals[-1], real_AS_vals[-1],marker='+', color="red", s=100, label="real End")

    plt.xlabel("Ht Cut")#, fontsize=22)
    plt.ylabel("AD Cut")#, fontsize=22)
    plt.xlim(75,350)
    plt.ylim(75,250)
    plt.title(title)#, fontsize=22)
    plt.colorbar(label="Iteration Step")
    plt.legend( loc='best', frameon=True)
    plt.grid(True)
    plt.savefig(f"paper/{title}.pdf")
    plt.savefig(f"paper/{title}.png")



def update_accumulated(acc_list, new_val, new_sample_size, total_samples_list):
    
    if not acc_list:  # If empty, initialize with the first value
        acc_list.append(new_val)
        total_samples_list.append(new_sample_size)
    else:
        prev_total_samples = total_samples_list[-1]
        new_total_samples = prev_total_samples + new_sample_size
        new_avg = (acc_list[-1] * prev_total_samples + new_val * new_sample_size) / new_total_samples
        acc_list.append(new_avg)
        total_samples_list.append(new_total_samples)


def comp_costs(b_accepted, b_accepted_events, b_ht_count, b_both_count, b_as_count, bnjets, ht_weight=1, as_weight=4):

    bnjets_reshaped = bnjets[:, None, None] 

    # Trigger path cost: average nJets over accepted events
    b_Ecomp_cost = ((b_accepted * bnjets_reshaped).sum(axis=0)) / (b_accepted_events + 1e-10)

    # Event-level cost: weighted counts per accepted event
    b_Tcomp_cost = (ht_weight * (b_ht_count - b_both_count) + as_weight * b_as_count) / (b_accepted_events + 1e-10)

    return b_Ecomp_cost, b_Tcomp_cost


def V1_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2):

    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    

    MAX = np.percentile(bas,99.999)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r2_s = 100 * s2_accepted_events / sht2.shape[0]

    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    total_s_rate = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])
    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]
    
    # Overlap 
    b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r1_sht = 100 * s1_ht_count / sht1.shape[0]
    r1_sas = 100 * s1_as_count / sas1.shape[0]

    r2_sht = 100 * s2_ht_count / sht2.shape[0]
    r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    t_b = 0.25
    a = [100, .2]
    
    
    #cost = (a[0]*np.abs(r_b - t_b))**(4) + (a[1] *np.abs(total_s_rate - 100))**1 #+ a[2]*b_overlap**2 + a[2]*s_overlap**2
    cost = (a[0]*np.abs(r_b - t_b)) + (a[1] *np.abs(total_s_rate - 100))
    log_Cost = np.log10(cost.clip(min=1e-10))
    

    return log_Cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V2_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2):
    #ttbar is picked to be 1
    
    

    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)

    MAX = np.percentile(bas,99.99)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    total_s_rate = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    
    
    r_as_ex = 100 * (b_as_count - b_both_count)/(bht.shape[0])

    
    
    # Overlap 
    b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r1_sht = 100 * s1_ht_count / sht1.shape[0]
    r1_sas = 100 * s1_as_count / sas1.shape[0]

    r2_sht = 100 * s2_ht_count / sht2.shape[0]
    r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    a = [100, .2, 25]
    t_b = 0.25
    percentage = .3
    #(a[0] *np.abs(r_b - t_b))**(4) + (a[1]*np.abs(r1_s - 90))**1 + 
    cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) + (a[2] * np.abs(r_as_ex - percentage*t_b))

    log_Cost = np.log10(cost.clip(min=1e-10))
    

    return log_Cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V4_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2):

    max1 = np.percentile(sht1,99.99)
    max2 = np.percentile(sht2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 120)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    
    

    max1 = np.percentile(sas1,99.99)
    max2 = np.percentile(sas2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bas,99.999)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 50)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]
    
    # Overlap 
    b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r1_sht = 100 * s1_ht_count / sht1.shape[0]
    r1_sas = 100 * s1_as_count / sas1.shape[0]

    r2_sht = 100 * s2_ht_count / sht2.shape[0]
    r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    # -----------------------------
    # Compute the cost based on the selected index.
    a = [1, 10, 3]
    t_b = 0.25
    
    
    event_cost_ht = 1
    event_cost_as = 10

    b_comp_cost = (
        event_cost_ht * (b_ht_count - b_both_count) + 
        event_cost_as * (b_as_count - b_both_count) +
        (event_cost_ht + event_cost_as) * b_both_count
    )/bht.shape[0]
    
    s1_comp_cost = (
        event_cost_ht * (s1_ht_count - s1_both_count) + 
        event_cost_as * (s1_as_count - s1_both_count) +
        (event_cost_ht + event_cost_as) * s1_both_count
    )
    
    s2_comp_cost = (
        event_cost_ht * (s2_ht_count - s2_both_count) + 
        event_cost_as * (s2_as_count - s2_both_count) +
        (event_cost_ht + event_cost_as) * s2_both_count
    )

    
    
    #a[0] * (1000*np.abs(r_b - t_b))**(1)
    cost = a[0] * (np.abs(r_b - t_b))**(1) + (a[1] *np.abs(r1_s - 90))**1 + (a[2]*(b_comp_cost +s1_comp_cost+s2_comp_cost))**2
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V3_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2):
    
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 50)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    
    MAX = np.percentile(bas,99.99)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 50)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    total_s_rate = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    b_accepted = (b_accepted_ht | b_accepted_as)
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]
    
    # Overlap 
    b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r1_sht = 100 * s1_ht_count / sht1.shape[0]
    r1_sas = 100 * s1_as_count / sas1.shape[0]

    r2_sht = 100 * s2_ht_count / sht2.shape[0]
    r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    # -----------------------------
    # Compute the cost based on the selected index.
    a = [100, .2, 1/0.5, 1/0.5]
    #a = [100, .2, 1/3.5, 1/2.5]
    t_b = 0.25
    
    #reshape jets per event for broadcasting
    bnjets_reshaped = bnjets[:, None, None]  # shape (N_events, 1, 1)
    
    Ht_cost = 1
    AS_cost = 4
    
    # Trigger path cost
    b_Tcomp_cost = ((b_accepted*bnjets_reshaped).sum(axis=0))/(b_accepted_events)
        
    # Event level Cost 
    b_Ecomp_cost = (Ht_cost * (b_ht_count - b_both_count) + AS_cost * (b_as_count))/(b_accepted_events)
    
    cost = (
    a[0] * np.abs(r_b - t_b) +
    a[1] * np.abs(total_s_rate - 100) +
    a[2] * np.maximum(b_Ecomp_cost - 5.5, 0) +
    a[3] * np.maximum(b_Tcomp_cost - 2.5, 0)
    )
    
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V1_Trigger_localAgent(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    
    # Define the local range around the given ht_value and as_value
    ht_min, ht_max = ht_value - ht_window, ht_value + ht_window
    as_min, as_max = as_value - as_window, as_value + as_window

    # Generate local ht and as values
    MAX = np.percentile(bht,99.99)

    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num_points)
    
    MAX = np.percentile(bas,99.99)

    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num_points)

    # Create a grid in the local window
    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij')
    

    # Signal computations
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])
    
    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    #r_s1 = 100 * s1_accepted_events / sht1.shape[0]
    
    
    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])
    
    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    #r_s2 = 100 * s2_accepted_events / sht2.shape[0]
    
    r_s = 100 * (s1_accepted_events+s2_accepted_events) / (sht1.shape[0]+sht2.shape[0])


    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])

    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)

    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    # Overlap calculations
    #b_overlap = 100 * b_both_count / bht.shape[0]
    #s_overlap = 100 * s_both_count / sht.shape[0]

    # Additional rates
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r_sht = 100 * (s1_ht_count + s2_ht_count)/ (sht1.shape[0]+sht2.shape[0])
    r_sas = 100 * (s1_as_count + s2_as_count)/ (sas1.shape[0]+sas2.shape[0])
    
    
    
    # Cost function
    a = [100, .2]
    t_b = 0.25
    cost = (a[0]*np.abs(r_b - t_b)) + (a[1] *np.abs(r_s - 100))
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS


def V2_Trigger_localAgent(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    

    # Define the local range around the given ht_value and as_value
    ht_min, ht_max = ht_value - ht_window, ht_value + ht_window
    as_min, as_max = as_value - as_window, as_value + as_window

    # Generate local ht and as values
    MAX = np.percentile(bht,99.99)

    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num_points)
    
    MAX = np.percentile(bht,99.99)

    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num_points)

    # Create a grid in the local window
    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij')
    

    # Signal computations
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    #r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    r_s = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    
    
    r_as_ex = 100 * (b_as_count - b_both_count)/(bht.shape[0])

    
    
    # Overlap 
    #b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    #s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    #s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r_sht = 100 * (s1_ht_count + s2_ht_count)/ (sht1.shape[0]+sht2.shape[0])
    r_sas = 100 * (s1_as_count + s2_as_count)/ (sas1.shape[0]+sas2.shape[0])


    #r1_sht = 100 * s1_ht_count / sht1.shape[0]
    #r1_sas = 100 * s1_as_count / sas1.shape[0]

    #r2_sht = 100 * s2_ht_count / sht2.shape[0]
    #r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    a = [100, .2, 25]
    t_b = 0.25
    percentage = .3
    #(a[0] *np.abs(r_b - t_b))**(4) + (a[1]*np.abs(r1_s - 90))**1 + 
    cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) + (a[2] * np.abs(r_as_ex - percentage*t_b))

    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS


def V3_Trigger_localAgent(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    
    # Define the local range around the given ht_value and as_value
    ht_min, ht_max = ht_value - ht_window, ht_value + ht_window
    as_min, as_max = as_value - as_window, as_value + as_window


    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num_points)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    

    MAX = np.percentile(bas,99.99)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num_points)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    #r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    #r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    r_s = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    #b_accepted = (b_accepted_ht | b_accepted_as)
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]
    
    # Overlap 
    #b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    #s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    #s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r_sht = 100 * (s1_ht_count + s2_ht_count)/ (sht1.shape[0]+sht2.shape[0])
    r_sas = 100 * (s1_as_count + s2_as_count)/ (sas1.shape[0]+sas2.shape[0])

    #r1_sht = 100 * s1_ht_count / sht1.shape[0]
    #r1_sas = 100 * s1_as_count / sas1.shape[0]

    #r2_sht = 100 * s2_ht_count / sht2.shape[0]
    #r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    # -----------------------------
    # Compute the cost based on the selected index.
    a = [100, .2, 1/0.5, 1/0.5]
    #a = [100, .2, 1/3.5, 1/2.5]
    t_b = 0.25
    
    b_Ecomp_cost, b_Tcomp_cost = comp_costs(
    (b_accepted_ht | b_accepted_as),
    b_accepted_events,
    b_ht_count,
    b_both_count,
    b_as_count,
    bnjets,
    )

    
    cost = (
    a[0] * np.abs(r_b - t_b) +
    a[1] * np.abs(r_s - 100) +
    a[2] * np.maximum(b_Ecomp_cost - 5.5, 0) +
    a[3] * np.maximum(b_Tcomp_cost - 2.5, 0)
    )
    
    log_Cost = np.log10(cost.clip(min=1e-10))
    

    return log_Cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS


def Trigger(bht_, sht1_, sht2_, bas_, sas1_, sas2_, bnjets ,ht_cut, as_cut):
    #num_signal = sht_.shape[0]
    #num_background = bht_.shape[0]
    
    
    # Apply cuts to both signal and background
    s1_accepted_ht = sht1_ >= ht_cut  # Signal events accepted by Ht cut
    s2_accepted_ht = sht2_ >= ht_cut
    s1_accepted_as = sas1_ >= as_cut  # Signal events accepted by AS cut
    s2_accepted_as = sas2_ >= as_cut
    b_accepted_ht = bht_ >= ht_cut  # Background events accepted by Ht cut
    b_accepted_as = bas_ >= as_cut  # Background events accepted by AS cut
            
    # Calculate the number of accepted signal events
    s1_accepted_events = np.sum(s1_accepted_ht) + np.sum(s1_accepted_as) - np.sum(s1_accepted_ht & s1_accepted_as)
    s2_accepted_events = np.sum(s2_accepted_ht) + np.sum(s2_accepted_as) - np.sum(s2_accepted_ht & s2_accepted_as)
    
    # Calculate the number of accepted background events
    b_accepted_events = np.sum(b_accepted_ht) + np.sum(b_accepted_as) - np.sum(b_accepted_ht & b_accepted_as)
            
    # Calculate rates
    #r1_s = 100 * s1_accepted_events / sht1_.shape[0]
    #r2_s = 100 * s2_accepted_events / sht2_.shape[0]
    r_s = 100 *(s1_accepted_events+s2_accepted_events)/(sht1_.shape[0]+sht2_.shape[0]+1e-10)
    r_b = 100 * b_accepted_events / bht_.shape[0]
    
    r_sht = 100*(np.sum(s1_accepted_ht)+np.sum(s2_accepted_ht))/(sht1_.shape[0]+sht2_.shape[0]+1e-10)
    r_bht = 100*(np.sum(b_accepted_ht)/bht_.shape[0])
    r_sas = 100*(np.sum(s1_accepted_as)+np.sum(s2_accepted_as))/(sas1_.shape[0]+sas2_.shape[0]+1e-10)
    r_bas = 100*(np.sum(b_accepted_as)/bas_.shape[0])
    
    b_accepted = (b_accepted_ht | b_accepted_as)
    
    ht_weight = 1
    as_weight = 4
    b_ht_count = np.sum(b_accepted_ht)
    b_as_count = np.sum(b_accepted_as)
    b_both_count = np.sum(b_accepted_ht & b_accepted_as)
    
    
    # Trigger path cost: average nJets over accepted events
    b_Ecomp_cost = ((b_accepted * bnjets).sum(axis=0)) / (b_accepted_events + 1e-10)

    # Event-level cost: weighted counts per accepted event
    b_Tcomp_cost = (ht_weight * (b_ht_count - b_both_count) + as_weight * b_as_count) / (b_accepted_events + 1e-10)

            
    
                
    return r_b, r_s, r_bht, r_bas, r_sht, r_sas, b_Ecomp_cost, b_Tcomp_cost




def read_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Background
        Bas01_tot = h5_file['data_scores01'][:]
        Bas04_tot = h5_file['data_scores04'][:]
        Bht_tot   = h5_file['data_ht'][:]
        B_npvs    = h5_file['data_Npv'][:]
        B_njets   = h5_file['data_njets'][:]

        # Signal (ttbar)
        Sas01_tot1 = h5_file['matched_tt_scores01'][:]
        Sas04_tot1 = h5_file['matched_tt_scores04'][:]
        Sht_tot1   = h5_file['matched_tt_ht'][:]
        S_npvs1    = h5_file['matched_tt_npvs'][:]
        S_njets1 = h5_file['matched_tt_njets'][:]
        # Signal (AA→4b)
        Sas01_tot2 = h5_file['matched_aa_scores01'][:]
        Sas04_tot2 = h5_file['matched_aa_scores04'][:]
        Sht_tot2   = h5_file['matched_aa_ht'][:]
        S_npvs2    = h5_file['matched_aa_npvs'][:]
        S_njets2 = h5_file['matched_aa_njets'][:]

    return (
        Sas01_tot1,  Sht_tot1, S_npvs1, S_njets1, 
        Sas01_tot2, Sht_tot2, S_npvs2, S_njets2,
        Bas01_tot,   Bht_tot,  B_npvs , B_njets
    )

path = "new_Data/Matched_data_2016.h5"

# Load data from both files
Sas_tot1, Sht_tot1, S_npvs1, S_njets1, Sas_tot2, Sht_tot2, S_npvs2, S_njets2, Bas_tot, Bht_tot, B_npvs, B_njets = read_data(path)
#Sht_tot1 /= 2.5
#Sht_tot2 /= 2.5

Nb = len(B_npvs)


print('hi')

N = Nb
#N = 40*50000


fixed_Ht_cut = np.percentile(Bht_tot[500000:600000],99.8)
fixed_AS_cut = np.percentile(Bas_tot[500000:600000],99.9)

Ht_cut = fixed_Ht_cut
AS_cut = fixed_AS_cut
fivefold_window = []

#bestcosts1 = []
#bestcosts2 = []

R = []
Rht = []
Ras = []

E = []
GE = []
Eht = []
Eas = []

RFixed = []
EFixed = []

contour_i1, contour_j1 = [], []
contour_f, contour_g = [], []
Id1_R, Id1_E, Id1_GE = [], [], []
Id1_r1_s, Id1_r2_s, Id1_r_s = [], [], []
Id1_r_bht, Id1_r_bas = [], []
Id1_r1_sht, Id1_r2_sht, Id1_r1_sas, Id1_r2_sas = [], [], [], []

acc_Id1_r1_s, acc_Id1_r1_sht, acc_Id1_r1_sas = [], [], []
acc_Id1_r2_s, acc_Id1_r2_sht, acc_Id1_r2_sas = [], [], []

# Total sample count lists for correct accumulation
total_samples_r1_s, total_samples_r2_s = [], []
total_samples_r1_sht, total_samples_r1_sas = [], []
total_samples_r2_sht, total_samples_r2_sas = [], []

total_samples_r_s = []
total_samples_ef = []


abs_rb_list = []
rs_list = []
b_TC_list = []
b_EC_list = []
total_cost_list = []
performance_list = []


chunk_size = 50000


for I in range(N):

    if I<10*chunk_size:continue

    if I==10*chunk_size: 
    
        start_idx = I
        end_idx = min(I + chunk_size, N)
        indices = list(range(start_idx, end_idx))

        bht = Bht_tot[indices]
        bas = Bas_tot[indices]
        b_npvs = B_npvs[indices]
        bnjets = B_njets[indices]  
        
        npv_min = np.min(b_npvs)
        npv_max = np.max(b_npvs)
        
        # Select signal events that fall within this Npv range
        signal_mask1 =indices #~(S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
    
        # Extract matching signal events
        sht1 = Sht_tot1[signal_mask1]
        sas1 = Sas_tot1[signal_mask1]
        snjets1 = S_njets1[signal_mask1]

        signal_mask2 = indices #(S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)
    
        # Extract matching signal events
        sht2 = Sht_tot2[signal_mask2]
        sas2 = Sas_tot2[signal_mask2]
        snjets2 = S_njets2[signal_mask2]
        
        cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = V1_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2)
        
        #cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS= V3_Trigger_Agent(
        #bht, sht1,sht2, bas, sas1, sas2, bnjets, snjets1, snjets2)

        bestcost_index = np.argmin(cost)
        i, j = np.unravel_index(bestcost_index, cost.shape)
        
        Ht_cut = HT[i,j]
        AS_cut = AS[i,j]
        
        contour_i1.append(HT[i, j])
        contour_j1.append(AS[i, j])
        
        
    #elif I%chunk_size==0 : 
    if I%chunk_size==0 : 
        
        start_idx = I
        end_idx = min(I + chunk_size, N)
        indices = list(range(start_idx, end_idx))

        bht = Bht_tot[indices]
        bas = Bas_tot[indices]
        b_npvs = B_npvs[indices] 
        bnjets = B_njets[indices]  
        
        npv_min = np.min(b_npvs)
        npv_max = np.max(b_npvs)
        
        # Select signal events that fall within this Npv range
        signal_mask1 = indices#(S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
    
        # Extract matching signal events
        sht1 = Sht_tot1[signal_mask1]
        sas1 = Sas_tot1[signal_mask1]
        snjets1 = S_njets1[signal_mask1]

        signal_mask2 = indices#(S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)
    
        # Extract matching signal events
        sht2 = Sht_tot2[signal_mask2]
        sas2 = Sas_tot2[signal_mask2]
        snjets2 = S_njets2[signal_mask2]

        #sht = np.hstack((sht1,sht2))
        #sas= np.hstack((sas1,sas2))
        
        r, ef, rht, ras, efht, efas, b_EC_val, b_TC_val = Trigger(bht,sht1,sht2,bas,sas1,sas2, bnjets,Ht_cut,AS_cut)
        #r_fixed, ef_fixed, rht_fixed, ras_fixed, efht_fixed, efas_fixed, b_EC_val_fixed, b_TC_val_fixed = Trigger(bht,sht1,sht2,bas,sas1,sas2, bnjets,fixed_Ht_cut,fixed_AS_cut)        
        
        abs_rb = 1/(1 - (np.abs(r - 0.25)))
        #print('abs_rb',abs_rb)
        rs_val = ef
        #b_TC_val = b_TC[i, j]
        #b_EC_val = b_EC[i, j]
        total_cost_val = b_EC_val + b_TC_val
        performance_val = abs_rb * rs_val
        
        abs_rb_list.append(abs_rb)
        rs_list.append(rs_val)
        b_TC_list.append(b_TC_val)
        b_EC_list.append(b_EC_val)
        total_cost_list.append(total_cost_val)
        performance_list.append(performance_val)
        
        R.append(r)
        #RFixed.append(r_fixed)
        Rht.append(rht)
        Ras.append(ras)

        E.append(ef)
        #EFixed.append(ef_fixed)
        Eht.append(efht)
        Eas.append(efas)

        update_accumulated(GE, ef, (sht1.shape[0]+sht2.shape[0]), total_samples_ef)


        #real point evo
        contour_f.append(Ht_cut)
        contour_g.append(AS_cut)
        
        
        #cost1
        
        cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS = V1_Trigger_localAgent(
        bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2,Ht_cut,AS_cut)
        
        #cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS = V3_Trigger_localAgent(
        #bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2, Ht_cut, AS_cut)

        bestcost_index = np.argmin(cost)
        i, j = np.unravel_index(bestcost_index, cost.shape)
        
        Ht_cut = HT[i,j]
        AS_cut = AS[i,j]
        
        fivefold_window.append([HT[i,j],AS[i,j]])
        if len(fivefold_window)>=5: 
            Ht_cut, AS_cut = np.mean(np.array(fivefold_window)[-5:], axis=0)

        


        cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = V1_Trigger_Agent(
        bht, sht1, sht2, bas, sas1, sas2)
        
        #cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS= V3_Trigger_Agent(
        #bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2)

        bestcost_index = np.argmin(cost)
        i, j = np.unravel_index(bestcost_index, cost.shape)


        contour_i1.append(HT[i, j])
        contour_j1.append(AS[i, j])
        
        
        Id1_R.append(r_b[i, j])
        #Id1_r_bht.append(r_bht[i, j])
        #Id1_r_bas.append(r_bas[i, j])

        # Store signal rates
        #Id1_r1_s.append(r1_s[i, j])
        #Id1_r2_s.append(r2_s[i, j])
        r_s = (r1_s[i, j]*sht1.shape[0] + r2_s[i, j]*sht2.shape[0])/(sht1.shape[0]+sht2.shape[0])
        Id1_E.append(r_s)  # Total signal rate

        #a = ((len(Id1_GE) - 1) * Id1_GE[-1] + r1_s[i, j]) / len(Id1_GE)
        #Id1_GE.append(a)

        # Store signal rates per cut
        #Id1_r1_sht.append(r1_sht[i, j])
        #Id1_r2_sht.append(r2_sht[i, j])
        #Id1_r1_sas.append(r1_sas[i, j])
        #Id1_r2_sas.append(r2_sas[i, j])

        update_accumulated(Id1_GE, r_s, (sht1.shape[0]+sht2.shape[0]), total_samples_r_s)

        # Update accumulated values with correct sample sizes
        #update_accumulated(acc_Id1_r1_s, r1_s[i, j], sht1.shape[0], total_samples_r1_s)
        #update_accumulated(acc_Id1_r2_s, r2_s[i, j], sht2.shape[0], total_samples_r2_s)
        #update_accumulated(acc_Id1_r1_sht, r1_sht[i, j], sht1.shape[0], total_samples_r1_sht)
        #update_accumulated(acc_Id1_r1_sas, r1_sas[i, j], sht1.shape[0], total_samples_r1_sas)
        #update_accumulated(acc_Id1_r2_sht, r2_sht[i, j], sht2.shape[0], total_samples_r2_sht)
        #update_accumulated(acc_Id1_r2_sas, r2_sas[i, j], sht2.shape[0], total_samples_r2_sas)
        
        
        print('time index:',I)


def average_perf_bins(performance_list, n_bins=25):
    time_indices = np.arange(len(performance_list))
    bins = np.array_split(time_indices, n_bins)
    avg_performance = []
    for bin_indices in bins:
        avg_perf = np.mean([performance_list[i] for i in bin_indices])
        avg_performance.append(avg_perf)

    return np.array(avg_performance)


time = np.linspace(0, 1, len(R))
R    = np.array(R)    * 400
Rht  = np.array(Rht)  * 400
Ras  = np.array(Ras)  * 400
Id1_R = np.array(Id1_R) * 400

# --- FIGURA (a): BACKGROUND ---

fig_a, ax = plt.subplots(figsize=(9, 6))

ax.plot(time, R,   label="Total Rate",          color='navy',        linewidth=2, marker='o')
ax.plot(time, Rht, label="HT Rate",             color='cyan',        linewidth=1, marker='o')
ax.plot(time, Ras, label="AD Rate",             color='deepskyblue', linewidth=1, marker='o')
ax.plot(time, Id1_R, linestyle="dashed", color='dodgerblue', linewidth=2,
        label="Total Ideal Rate")

ax.axhline(y=110, color='grey', linestyle='--', linewidth=1.5,
           label='Upper Tolerance (110kHz)')
ax.axhline(y=90,  color='grey', linestyle='--', linewidth=1.5,
           label='Lower Tolerance (90kHz)')

ax.set_xlabel('Time (Fraction of Run)', labelpad=10, loc='center')
ax.set_ylabel('Background Rate (kHz)',  labelpad=10, loc='center')

ax.set_ylim(40, 290)

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
ax.set_yticks(np.arange(0, 251, 45))
ax.tick_params(axis='both')

yticks0 = ax.get_yticks()
ax.set_yticks(yticks0[1:])

# *** LEGENDA IDENTICA ***
ax.legend(title="Case 1", fontsize=15,
          loc='best', frameon=True, handlelength=2.5,
          handletextpad=0.5, borderpad=0.8)
ax.grid(True)

add_cms_header(fig_a, left_x=0.16, right_x=0.96, y=0.99)  # CMS Open Data + Run 283408
fig_a.tight_layout()
fig_a.savefig("paper/simple_controller_case1_a.pdf", bbox_inches="tight")
fig_a.savefig("paper/simple_controller_case1_a.png", bbox_inches="tight")
plt.close(fig_a)

# --- FIGURA (b): EFFICIENCY ---

# Se non lo sono già, rendi array anche queste liste
GE     = np.array(GE)
Eht    = np.array(Eht)
Eas    = np.array(Eas)
Id1_GE = np.array(Id1_GE)

fig_b, ax = plt.subplots(figsize=(9, 6))

ax.plot(time, GE,  label="Total Cumulative Signal Efficiency",
        color='mediumvioletred', linewidth=2, marker='o')
ax.plot(time, Eht, label="HT Signal Efficiency",
        color='mediumpurple',    linewidth=1, marker='o')
ax.plot(time, Eas, label="AD Signal Efficiency",
        color='orchid',          linewidth=1, marker='o')
ax.plot(time, Id1_GE, linestyle="dashed", color='black', linewidth=2,
        label="Total Ideal Cumulative Signal Efficiency")

ax.set_xlabel('Time (Fraction of Run)', labelpad=10, loc='center')
ax.set_ylabel('Efficiency (%)',         labelpad=10, loc='center')

ax.set_ylim(40, 70)

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
ax.set_yticks(np.arange(35, 70, 5))
ax.tick_params(axis='both')

yticks1 = ax.get_yticks()
ax.set_yticks(yticks1[1:])

# *** LEGENDA IDENTICA ***
ax.legend(title="Case 1", fontsize=15,
          loc='best', frameon=True, handlelength=2.5,
          handletextpad=0.5, borderpad=0.8)
ax.grid(True)

add_cms_header(fig_b, left_x=0.14, right_x=0.96, y=0.99)  # CMS Open Data + Run 283408
fig_b.tight_layout()
fig_b.savefig("paper/simple_controller_case1_b.pdf", bbox_inches="tight")
fig_b.savefig("paper/simple_controller_case1_b.png", bbox_inches="tight")
plt.close(fig_b)
