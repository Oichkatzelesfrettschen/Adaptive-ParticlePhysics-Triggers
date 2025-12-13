import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import time

import mplhep as hep
hep.style.use("CMS")



#plot che fa l'evoluzione dei cut su Ht e AS   
def plot_evolution(Ht_vals, AS_vals, real_Ht_vals, real_AS_vals, title="real_evoultion_plot(r1)"):
    plt.figure(figsize=(8, 6))
    # Plot all points
    plt.scatter(Ht_vals, AS_vals, c=np.arange(len(Ht_vals)), marker='o', cmap="viridis", label="ideal Points",s=50)
    plt.xlabel("Ht Cut", fontsize=22)
    plt.ylabel("AS Cut", fontsize=22)
    plt.xlim(np.min(Ht_vals) * 0.9, np.max(Ht_vals) * 1.1)
    plt.ylim(np.min(AS_vals) * 0.9, np.max(AS_vals)* 1.1)

    plt.title(title, fontsize=22)
    plt.colorbar(label="Iteration Step")
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.grid(True)
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


def V0_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2):

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
    #a = [5, 5, 3]
    t_b = 0.25
    
    
    cost = np.abs(r_b - t_b)
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V1_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2):
   
    #max1 = np.percentile(sht1,99.99)
    #max2 = np.percentile(sht2,99.99)
    #MAX = max(max1,max2)
    MAX = np.percentile(bht,99.99)
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 50)

    #max1 = np.percentile(sas1,99.99)
    #max2 = np.percentile(sas2,99.99)
    #MAX = max(max1,max2)
    MAX = np.percentile(bas,99.99)
        
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
    
    
    max1 = np.percentile(sht1,99.99)
    max2 = np.percentile(sht2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.001), MAX, 50)

    max1 = np.percentile(sas1,99.99)
    max2 = np.percentile(sas2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bas,99.99)
    
    as_vals = np.linspace(np.percentile(bas,0.001), MAX, 50)

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
    r2_s = 100 * s2_accepted_events / sht2.s10hape[0]
    
    
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
    #cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) - (a[2] * r_as_ex)

    log_Cost = np.log10(cost.clip(min=1e-10))

    return cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS



def V3_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2):
    
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 50)#previous 100
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    
    MAX = np.percentile(bas,99.99)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 50)#previous 100

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
    
    #reshape jets per event for broadcasting
    bnjets_reshaped = bnjets[:, None, None]  # shape (N_events, 1, 1)
    #snjets1_reshaped = snjets1[:, None, None] 
    #snjets2_reshaped = snjets2[:, None, None] 

    Ht_cost = 1
    AS_cost = 4

    a = [100, .2, 1/0.5, 1/0.5]
    t_b = 0.25
    
    
    
    # Trigger path cost
    b_Ecomp_cost = ((b_accepted*bnjets_reshaped).sum(axis=0))/(b_accepted_events)  
        
    # Event level Cost 
    b_Tcomp_cost = (Ht_cost*(b_ht_count - b_both_count) + AS_cost * (b_as_count))/(b_accepted_events)
    
    
    
    cost = (
    a[0] * np.abs(r_b - t_b) +
    a[1] * np.abs(total_s_rate - 100) +
    a[2] * np.maximum(b_Ecomp_cost - 4.2, 0) +
    a[3] * np.maximum(b_Tcomp_cost - 3.1, 0)
    )
    
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS
    
 


def comp_cost_test(bht, bas, bnjets, sht1, sas1, snjets1, sht2, sas2, snjets2,
use_path1=True, use_path2=True):

    # Build HT and AS thresholds
    MAX_ht = np.percentile(bht, 99.99)
    ht_vals = np.linspace(np.percentile(bht, 0.01), MAX_ht, 50)

    MAX_as = np.percentile(bas, 99.99)
    as_vals = np.linspace(np.percentile(bas, 0.01), MAX_as, 50)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij')

    # ---------- SIGNAL PATH 1 ----------
    if use_path1:
        s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
        s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
        b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
        
    else:
        s1_accepted_ht = np.zeros_like(sht1, dtype=bool)[:, None, None]
        s2_accepted_ht = np.zeros_like(sht2, dtype=bool)[:, None, None]       
        b_accepted_ht = np.zeros_like(bht, dtype=bool)[:, None, None]

        
        

    # ---------- SIGNAL PATH 2 ----------
    if use_path2:
        s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
        s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]
        b_accepted_as = bas[:, None, None] >= AS[None, :, :]
        
    else:
        s1_accepted_as = np.zeros_like(sas1, dtype=bool)[:, None, None]
        s2_accepted_as = np.zeros_like(sas2, dtype=bool)[:, None, None]
        b_accepted_as = np.zeros_like(bas, dtype=bool)[:, None, None]



    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s2_ht_count = s2_accepted_ht.sum(axis=0)
    
    s1_as_count = s1_accepted_as.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)


    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count

    # ---------- Total Signal Rate ----------
    total_s_accepted_events = s1_accepted_events + s2_accepted_events
    
    total_s_rate = 100 * total_s_accepted_events / (sht1.shape[0] + sht2.shape[0] + 1e-10)

    Ht_cost = 1
    AS_cost =4
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    
    b_accepted = b_accepted_ht | b_accepted_as
    
    
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    # ---------- COMPUTATIONAL COST ----------
    bnjets_reshaped = bnjets[:, None, None]
    b_Ecomp_cost = ((b_accepted * bnjets_reshaped).sum(axis=0))/(b_accepted_events)


    b_Tcomp_cost = (Ht_cost*(b_ht_count - b_both_count) + AS_cost * b_as_count)/(b_accepted_events)
    
    # ---------- COST FUNCTION ----------
    a = [100, 0.2]
    t_b = 0.25
    cost = a[0] * np.abs(r_b - t_b) + a[1] * np.abs(total_s_rate - 100)

    return cost, b_Ecomp_cost, b_Tcomp_cost


def read_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        #Read datasets for background

        Bas01_tot = h5_file['data_scores01'][:] #dim = 1
        Bht_tot = h5_file['data_ht'][:]
        B_npvs = h5_file['data_Npv'][:]
        B_njets = h5_file['data_njets'][:]


        # Read datasets for signal
        Sas01_tot1 = h5_file['matched_tt_scores01'][:]
        Sht_tot1 = h5_file['matched_tt_ht'][:]
        S_npvs1 = h5_file['matched_tt_npvs'][:]
        S_njets1 = h5_file['matched_tt_njets'][:]


        Sas01_tot2 = h5_file['matched_aa_scores01'][:]
        Sht_tot2 = h5_file['matched_aa_ht'][:]
        S_npvs2 = h5_file['matched_aa_npvs'][:]
        S_njets2 = h5_file['matched_aa_njets'][:]

        
    return Sas01_tot1, Sht_tot1, S_npvs1, S_njets1, Sas01_tot2, Sht_tot2, S_npvs2, S_njets2, Bas01_tot, Bht_tot, B_npvs, B_njets #,data_ht, data_score, data_npv

# path = "new_Data/Matched_data_2016.h5"
path = "Data/Matched_data_2016_with04_paper.h5"

# Load data from both files
Sas_tot1, Sht_tot1, S_npvs1, S_njets1, Sas_tot2, Sht_tot2, S_npvs2, S_njets2, Bas_tot, Bht_tot, B_npvs, B_njets = read_data(path)
#Sht_tot1 /= 2.5
#Sht_tot2 /= 2.5

Nb = len(B_npvs)


print('hi')

N = Nb
print(Nb)
#N = 10*50000

bestcosts1 = []
bestcosts2 = []

contour_i1, contour_j1 = [], []
Id1_R, Id1_E, Id1_GE = [], [], []
Id1_r1_s, Id1_r2_s = [], []
Id1_r_bht, Id1_r_bas = [], []
Id1_r1_sht, Id1_r2_sht, Id1_r1_sas, Id1_r2_sas = [], [], [], []

acc_Id1_r1_s, acc_Id1_r1_sht, acc_Id1_r1_sas = [], [], []
acc_Id1_r2_s, acc_Id1_r2_sht, acc_Id1_r2_sas = [], [], []

# Total sample count lists for correct accumulation
total_samples_r1_s, total_samples_r2_s = [], []
total_samples_r1_sht, total_samples_r1_sas = [], []
total_samples_r2_sht, total_samples_r2_sas = [], []

total_samples_r_s = []

test_Ecost_both = []
test_Tcost_both = []

test_Ecost_ht = []
test_Tcost_ht = []

test_Ecost_as = []
test_Tcost_as = []


chunk_size = 20000


for I in range(N):
    if I < 200000: continue
    if I%chunk_size==0 : 
        start=time.time()

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
        #signal_mask1 = (S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
    
        # Extract matching signal events
        sht1 = Sht_tot1[indices]
        sas1 = Sas_tot1[indices]
        snjets1 = S_njets1[indices]

        #signal_mask2 = (S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)
    
        # Extract matching signal events
        sht2 = Sht_tot2[indices]
        sas2 = Sas_tot2[indices]
        snjets2 = S_njets2[indices]

        #sht = np.hstack((sht1,sht2))
        #sas= np.hstack((sas1,sas2))

        #cost1
        #cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = V3_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2) 
                
        cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = V1_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2)
        bestcost_index = np.argmin(cost)
        i, j = np.unravel_index(bestcost_index, cost.shape)
       


    ############################################################da qui
        #test case for cost
        cost, E_cost, T_cost  = comp_cost_test(bht, bas, bnjets, sht1, sas1, snjets1, sht2, sas2, snjets2,
        use_path1=True, use_path2=True)

        bestcost_index = np.argmin(cost)
        i, j = np.unravel_index(bestcost_index, cost.shape)
        bestcosts1.append(np.log(cost[i,j]))

        
        e_cost = E_cost[i,j]
        t_cost = T_cost[i,j]
        test_Ecost_both.append(e_cost)
        test_Tcost_both.append(t_cost)
        
        print('both: ', e_cost, t_cost)
        
        
        cost, E_cost, T_cost  = comp_cost_test(bht, bas, bnjets, sht1, sas1, snjets1, sht2, sas2, snjets2,
        use_path1=True, use_path2=False)

        bestcost_index = np.argmin(cost)
        i, j = np.unravel_index(bestcost_index, cost.shape)
        
        bestcosts2.append(np.log(cost[i,j]))

        e_cost = E_cost[i,j]
        t_cost = T_cost[i,j]
        test_Ecost_ht.append(e_cost)
        test_Tcost_ht.append(t_cost)
        
        print('ht: ', e_cost, t_cost)
        
        
        cost, E_cost, T_cost  = comp_cost_test(bht, bas, bnjets, sht1, sas1, snjets1, sht2, sas2, snjets2,
        use_path1=False, use_path2=True)

        bestcost_index = np.argmin(cost)
        i, j = np.unravel_index(bestcost_index, cost.shape)
        
        
        e_cost = E_cost[i,j]
        t_cost = T_cost[i,j]
        test_Ecost_as.append(e_cost)
        test_Tcost_as.append(t_cost)
        
        print('as: ', e_cost, t_cost)
        


              
        #cost2
        ###cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = V2_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2)

        ###bestcost_index = np.argmin(cost)
        ###i, j = np.unravel_index(bestcost_index, cost.shape)
        
        #bestcosts2.append(np.log(cost[i,j]))

    

        
        #cost3
        ###cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = V3_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2) 

        #cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = V3_Trigger_Agent(
        #bht, sht1, sht2, bas, sas1, sas2)
        
        ###bestcost_index = np.argmin(cost)
        ###i, j = np.unravel_index(bestcost_index, cost.shape)
    ###############################finisce qui 

        contour_i1.append(HT[i, j])
        contour_j1.append(AS[i, j])
        
        
        Id1_R.append(r_b[i, j])
        Id1_r_bht.append(r_bht[i, j])
        Id1_r_bas.append(r_bas[i, j])

        # Store signal rates
        Id1_r1_s.append(r1_s[i, j])
        Id1_r2_s.append(r2_s[i, j])
        r_s = (r1_s[i, j]*sht1.shape[0] + r2_s[i, j]*sht2.shape[0])/(sht1.shape[0]+sht2.shape[0])
        Id1_E.append(r_s)  # Total signal rate

        #a = ((len(Id1_GE) - 1) * Id1_GE[-1] + r1_s[i, j]) / len(Id1_GE)
        #Id1_GE.append(a)

        # Store signal rates per cut
        Id1_r1_sht.append(r1_sht[i, j])
        Id1_r2_sht.append(r2_sht[i, j])
        Id1_r1_sas.append(r1_sas[i, j])
        Id1_r2_sas.append(r2_sas[i, j])

        update_accumulated(Id1_GE, r_s, (sht1.shape[0]+sht2.shape[0]), total_samples_r_s)
        
        print('time index:',I)




print(bestcosts2[:5]/np.mean(bestcosts2))
print(np.mean(bestcosts1))
bestcosts1 = np.array(bestcosts1)
bestcosts2 = np.array(bestcosts2)
time = range(len(bestcosts1))













#fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ----------------------------
# ----------------------------
# 1) Event cost (sinistra) - normalizzato a max = 1, bin COMUNI + stairs
# ----------------------------
datasets = [
    (test_Ecost_both, "both paths"),
    (test_Ecost_ht,   "HT path only"),
    (test_Ecost_as,   "AD path only")
]
colors = ["tab:blue", "tab:orange", "tab:green"]

all_ecost = np.concatenate([d for d, _ in datasets])
edges = np.linspace(all_ecost.min(), all_ecost.max(), 21)

hists = [np.histogram(d, bins=edges, density=True)[0] for d, _ in datasets]
ymax = max(h.max() for h in hists) or 1.0



plt.figure(figsize=(6, 5))
for (y, (data, label), col) in zip(hists, datasets, colors):
    plt.stairs(y / ymax, edges,
               label=f"{label}, mean = {np.mean(data):.1f}",
               color=col, linewidth=1.5)


plt.xlabel('Event Cost in Case 1',   loc='center')
plt.ylabel('Density', loc='center')
#plt.yscale('log')
plt.ylim(0,1.5)
#plt.ylim(1e-3, 2.05)  # range adatto per scala log

plt.legend(fontsize=14, frameon=True, facecolor='white')
plt.gcf().text(0.26, 0.92, "CMS Open Data", fontsize=18, fontweight="bold")
plt.gcf().text(0.92, 0.92, "Run 283408", ha='right', fontsize=18)


plt.tick_params(axis='both')
plt.tight_layout()

plt.savefig("paper/event_cost_case1.pdf", dpi=300, bbox_inches='tight')
plt.savefig("paper/event_cost_case1.png", dpi=300, bbox_inches='tight')
plt.close()



# =========================
# 2 TRIGGER COST PLOT
# =========================
plt.figure(figsize=(6, 5))

test_Tcost_both = np.array(test_Tcost_both)
test_Tcost_ht   = np.array(test_Tcost_ht)
test_Tcost_as   = np.array(test_Tcost_as)

plt.hist(test_Tcost_both, bins=20, alpha=1, density=True,
         label=f'both paths, mean = {np.mean(test_Tcost_both):.1f}',
         histtype='step', color='tab:blue', linewidth=2)

plt.hist(test_Tcost_ht, bins=20, alpha=1, density=True,
         label=f'HT path only, mean = {np.mean(test_Tcost_ht):.1f}',
         histtype='step', color='tab:orange', linewidth=2)

plt.hist(test_Tcost_as, bins=20, alpha=1, density=True,
         label=f'AD path only, mean = {np.mean(test_Tcost_as):.1f}',
         histtype='step', color='tab:green', linewidth=2)
plt.gcf().text(0.26, 0.92, "CMS Open Data", fontsize=18, fontweight="bold")
plt.gcf().text(0.92, 0.92, "Run 283408", ha='right', fontsize=18)


plt.xlabel('Trigger Cost in Case 1', loc='center')
plt.ylabel('Density',  loc='center')
plt.yscale('log')
plt.ylim(1e-3, 1e2)  # range adatto per scala log
plt.legend(fontsize=13, loc='upper center', frameon=True, facecolor='white')
plt.tight_layout()
 
plt.savefig("paper/trigger_cost_case1.pdf", dpi=300, bbox_inches='tight')
plt.savefig("paper/trigger_cost_case1.png", dpi=300, bbox_inches='tight')

plt.close()