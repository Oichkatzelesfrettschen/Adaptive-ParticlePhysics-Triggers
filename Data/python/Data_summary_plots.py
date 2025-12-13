import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from matplotlib.lines import Line2D
import mplhep as hep

hep.style.use(hep.style.ROOT) 
# Or choose one of the experiment styles
hep.style.use(hep.style.ATLAS)
# or
hep.style.use("CMS") # string aliases work too


def comp_costs(b_accepted, b_accepted_events, b_ht_count, b_both_count, b_as_count, bnjets, ht_weight=1, as_weight=4):

    bnjets_reshaped = bnjets[:, None, None] 

    # Trigger path cost: average nJets over accepted events
    b_Ecomp_cost = ((b_accepted * bnjets_reshaped).sum(axis=0)) / (b_accepted_events + 1e-10)

    # Event-level cost: weighted counts per accepted event
    b_Tcomp_cost = (ht_weight * (b_ht_count - b_both_count) + as_weight * b_as_count) / (b_accepted_events + 1e-10)

    return b_Ecomp_cost, b_Tcomp_cost


def V1_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2, bnjets):

    max1 = np.percentile(sht1,99.99)
    max2 = np.percentile(sht2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    
    

    max1 = np.percentile(sas1,99.99)
    max2 = np.percentile(sas2,99.99)
    MAX = max(max1,max2)
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


def V2_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2, bnjets):
    #ttbar is picked to be 1
    
    
    max1 = np.percentile(sht1,99.99)
    max2 = np.percentile(sht2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)

    max1 = np.percentile(sas1,99.99)
    max2 = np.percentile(sas2,99.99)
    MAX = max(max1,max2)
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


def V3_Trigger_Agent(bht, sht1, sht2, bas, sas1, sas2, bnjets):
    
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    
    MAX = np.percentile(bas,99.99)
    
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
    #a = [100, .2, 1/0.5, 1/0.5]
    a = [100, 0.2, 1, 1]

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
    a[2] * np.maximum(b_Ecomp_cost - 3.2, 0) +
    #a[3] * np.maximum(b_Tcomp_cost - 2.5, 0)
    a[3] * np.maximum(b_Tcomp_cost - 2.2, 0)
    )
    
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r1_s, r2_s, b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V1_Trigger_localAgent(bht, sht1, sht2, bas, sas1, sas2, bnjets, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    
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


def V2_Trigger_localAgent(bht, sht1, sht2, bas, sas1, sas2, bnjets, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    

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


def V3_Trigger_localAgent(bht, sht1, sht2, bas, sas1, sas2, bnjets, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    
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
    #a = [100, .2, 1/0.5, 1/0.5]
    a = [100, 0.05, 1, 1]
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
    a[2] * np.maximum(b_Ecomp_cost - 4.3, 0) +
    #a[3] * np.maximum(b_Tcomp_cost - 2.5, 0)
    a[3] * np.maximum(b_Tcomp_cost - 3.3, 0)
    )
    
    log_Cost = np.log10(cost.clip(min=1e-10))
    #i_, j_ = np.unravel_index(np.argmin(log_Cost), log_Cost.shape)
    #print('case3',b_Ecomp_cost[i_,j_],b_Tcomp_cost[i_,j_])

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
        #Read datasets for background

        # Bas01_tot = h5_file['data_scores'][:]
        Bas01_tot = h5_file['data_scores01'][:] #dim = 1
        Bht_tot = h5_file['data_ht'][:]
        B_npvs = h5_file['data_Npv'][:]
        B_njets = h5_file['data_njets'][:]


        # Read datasets for signal
        # Sas01_tot1 = h5_file['matched_tt_scores'][:]
        Sas01_tot1 = h5_file['matched_tt_scores01'][:]
        Sht_tot1 = h5_file['matched_tt_ht'][:]
        S_npvs1 = h5_file['matched_tt_npvs'][:]
        S_njets1 = h5_file['matched_tt_njets'][:]


        # Sas01_tot2 = h5_file['matched_aa_scores'][:]
        Sas01_tot2 = h5_file['matched_aa_scores01'][:]
        Sht_tot2 = h5_file['matched_aa_ht'][:]
        S_npvs2 = h5_file['matched_aa_npvs'][:]
        S_njets2 = h5_file['matched_aa_njets'][:]

        
    return Sas01_tot1, Sht_tot1, S_npvs1, S_njets1, Sas01_tot2, Sht_tot2, S_npvs2, S_njets2, Bas01_tot, Bht_tot, B_npvs, B_njets #,data_ht, data_score, data_npv



# path =  "new_Data/Matched_data_2016.h5"
path = "Data/Matched_data_2016_with04_paper.h5" #Zixin

def average_perf_bins(performance_list, cost_list, n_bins=10):
    time_indices = np.arange(len(performance_list))
    bins = np.array_split(time_indices, n_bins)
    avg_performance, avg_cost = [], []
    for bin_indices in bins:
        avg_perf = np.mean([performance_list[i] for i in bin_indices])
        avg_cst = np.mean([cost_list[i] for i in bin_indices])
        avg_performance.append(avg_perf)
        avg_cost.append(avg_cst)
    return np.array(avg_performance), np.array(avg_cost)


def plot_performance(cases_data, n_bins=5, save_path=None):
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for idx, (case_name, data) in enumerate(cases_data.items()):
        avg_perf, avg_cost = average_over_bins(data['performance'], data['cost'], n_bins=n_bins)
        plt.plot(avg_perf, avg_cost, marker=markers[idx % len(markers)], color=colors[idx % len(colors)], label=case_name, linewidth=2)
    plt.xlabel('Average Performance (per time bin)')
    plt.ylabel('Average Total Cost (per time bin)')
    plt.xlim(65,90)
    plt.ylim(5.5,10.5)
    plt.title('Comparison of Performance vs Computational Cost Across Cases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        print('no savefig path')


def average_pair_over_bins(x_list, y_list, n_bins=10):
    
    idx = np.arange(len(x_list))
    bins = np.array_split(idx, n_bins)
    x_avg, y_avg = [], []
    #x_avg.append(x_list[0])
    #y_avg.append(y_list[0])
    
    for b in bins:
        x_avg.append(np.mean([x_list[i] for i in b]))
        y_avg.append(np.mean([y_list[i] for i in b]))
    return np.array(x_avg), np.array(y_avg)



def plot_case_comparison_split(cases_data, n_bins=10, save_prefix="paper/Summary_dim1"):
    x_keys  = ['w0absrb', 'absrb', 'rs', 'Tcost']
    y_keys  = ['w1rs', 'cost', 'cost', 'Ecost']
    xlabels = [
        r'$|r_b-r_t|/\sigma_b$',
        r'$|r_b-r_t|$',
        r'$\epsilon$: Total Signal Efficiency $(\%)$',
        'Average Trigger Cost'
    ]
    ylabels = [
        r'$(1-\epsilon)/\sigma_s$',
        'Total Computational Cost',
        'Total Computational Cost',
        'Average Event Cost'
    ]

    markers = ['o', 's', 'P', 'D']
    base_colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

    for p in range(4):
        fig, ax = plt.subplots(figsize=(7, 6))

        for c_idx, (case_name, data) in enumerate(cases_data.items()):
            base_color = mpl.colors.to_rgb(base_colors[c_idx % len(base_colors)])
            x_vals, y_vals = average_pair_over_bins(data[x_keys[p]], data[y_keys[p]], n_bins=n_bins)
            if len(x_vals) < 2:
                continue

            # --- segmenti con gradiente
            points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t_vals = np.linspace(0, 1, len(segments))
            colors = [(1 - t) * np.array(base_color) + t * np.ones(3) for t in t_vals]
            lc = LineCollection(segments, colors=colors, linewidth=min(2*(4-c_idx),4))
            ax.add_collection(lc)

            # --- stella sul primo punto
            ax.plot(x_vals[0], y_vals[0], marker='*', color=base_color, markersize=15)

            # --- marker sui punti successivi
            ax.plot(x_vals[1:], y_vals[1:], marker=markers[c_idx % len(markers)],
                    linestyle='None', color=base_color, markersize=5)

        # === legenda in ogni subplot ===
        legend_elements = [
            Line2D([], [], color='tab:blue', marker='o', linestyle='-', label='Fixed Menu'),
            Line2D([], [], color='tab:orange', marker='s', linestyle='-', label='Standard'),
            Line2D([], [], color='tab:green', marker='P', linestyle='-', label='Anomaly Focused'),
            Line2D([], [], color='tab:red', marker='D', linestyle='-', label='Low-Comp Focused'),
            Line2D([], [], marker='*', color='black', linestyle='None',
                   markersize=10, label='Start of the Run')
        ]
        ax.legend(handles=legend_elements,  frameon=True, fontsize=16,loc='best')
        # === colorbar invertita (Light → Dark) in ogni subplot ===
        sm = ScalarMappable(cmap='Greys_r', norm=plt.Normalize(0, 1))  # _r = reversed
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
        cbar.set_label('Relative Time (Dark → Light)', fontsize=22)
        cbar.set_ticks([])

        # === formattazione assi ===
        #ax.set_xlabel(xlabels[p])
        #ax.set_ylabel(ylabels[p])
        ax.set_xlabel(xlabels[p],  loc='center')
        ax.set_ylabel(ylabels[p],   loc='center')
        #if p==0 :
        #    ax.set_xlim(0,20)
        #if p==1 :
        #    ax.set_xlim(0,80)
        #    ax.set_ylim(7,12)
#
        if p==2 :
            ax.set_ylim(6.5,10)        
        if p==3 :
            ax.legend(handles=legend_elements,  frameon=True, fontsize=16,loc='lower left')
            ax.set_ylim(2.5,5)        
            ax.set_xlim(1.5,5)


        ax.grid(True)
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=8)
        ax.margins(x=0.05, y=0.05)   # margine 5%

        fig.tight_layout()
        fig.savefig(f"{save_prefix}_subplot{p+1}.pdf")
        plt.close(fig)





if __name__ == "__main__":
    # path = "new_Data/Matched_data_2016.h5"
    path = "Data/Matched_data_2016_with04_paper.h5"
    Sas_tot1, Sht_tot1, S_npvs1, S_njets1, Sas_tot2, Sht_tot2, S_npvs2, S_njets2, Bas_tot, Bht_tot, B_npvs, B_njets = read_data(path)
    #Bas_tot, Bht_tot, B_npvs, B_njets = Bas_tot[500000:], Bht_tot[500000:], B_npvs[500000:], B_njets[500000:]
    N = len(B_npvs)
    #N = 20*50000
    #chunk_size = 50000
    chunk_size = 20000

    
    

    cases_lists = {
        'Fixed Menu': {'cost': [], 'absrb':[], 'rs':[], 'w0absrb':[],'w1rs':[],  'Tcost':[], 'Ecost':[]},
        'Standard': {'cost': [], 'absrb':[], 'rs':[],'w0absrb':[],'w1rs':[], 'Tcost':[], 'Ecost':[]},
        'Anomaly Focused': {'cost': [], 'absrb':[], 'rs':[],'w0absrb':[],'w1rs':[], 'Tcost':[], 'Ecost':[]},
        'Low-Comp Focused': {'cost': [], 'absrb':[], 'rs':[],'w0absrb':[],'w1rs':[], 'Tcost':[], 'Ecost':[]}
    }
    

    # Initialize HT and AS cuts from the first value of each Agent
    initial_indices = list(range(0, chunk_size))
    bht_init, bas_init, bnjets_init = Bht_tot[initial_indices], Bas_tot[initial_indices], B_njets[initial_indices]
    npv_init = B_npvs[initial_indices]
    mask1_init = initial_indices#(S_npvs1 >= np.min(npv_init)) & (S_npvs1 <= np.max(npv_init))
    mask2_init = initial_indices#(S_npvs2 >= np.min(npv_init)) & (S_npvs2 <= np.max(npv_init))
    sht1_init, sas1_init, snjets1_init = Sht_tot1[mask1_init], Sas_tot1[mask1_init], S_njets1[mask1_init]
    sht2_init, sas2_init, snjets2_init = Sht_tot2[mask2_init], Sas_tot2[mask2_init], S_njets2[mask2_init]

    init_agents = [V1_Trigger_Agent, V2_Trigger_Agent, V3_Trigger_Agent]
    initial_cuts = {}
    for case, init_agent in zip(['Standard', 'Anomaly Focused', 'Low-Comp Focused'], init_agents):
        cost_init, _, _, _, _, _, _, _, _, _, _, _, _, HT_init, AS_init = init_agent(bht_init, sht1_init, sht2_init, bas_init, sas1_init, sas2_init, bnjets_init)
        i_init, j_init = np.unravel_index(np.argmin(cost_init), cost_init.shape)
        initial_cuts[case] = (HT_init[i_init, j_init], AS_init[i_init, j_init])
        
    
    cuts = {
    'Standard': {'Ht': initial_cuts['Standard'][0], 'AS': initial_cuts['Standard'][1]},
    'Anomaly Focused': {'Ht': initial_cuts['Anomaly Focused'][0], 'AS': initial_cuts['Anomaly Focused'][1]},
    'Low-Comp Focused': {'Ht': initial_cuts['Low-Comp Focused'][0], 'AS': initial_cuts['Low-Comp Focused'][1]},
    }
    
    #print(cuts)
    
    
    
   # Ht_cut_fixed, AS_cut_fixed = np.percentile(Bht_tot[0:100000],99.8), np.percentile(Bas_tot[0:100000],99.9)
    Ht_cut_fixed, AS_cut_fixed = np.percentile(Bht_tot[500000:600000],99.8), np.percentile(Bas_tot[500000:600000],99.9)

    #print('Ht_cut_fixed: ', Ht_cut_fixed)
    #print('AS_cut_fixed: ', AS_cut_fixed)
    #Ht_cut_fixed, AS_cut_fixed =  cuts['Standard']['Ht'], cuts['Standard']['AS']
    #print(Ht_cut_fixed, AS_cut_fixed)

    for I in range(0, N, chunk_size):
        end_idx = min(I + chunk_size, N)
        indices = list(range(I, end_idx))
        bht, bas, bnpv, bnjets = Bht_tot[indices], Bas_tot[indices], B_npvs[indices], B_njets[indices]
        npv_min, npv_max = np.min(bnpv), np.max(bnpv)
        mask1 = indices# (S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
        mask2 = indices#(S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)
        sht1, sas1, snjets1 = Sht_tot1[mask1], Sas_tot1[mask1], S_njets1[mask1]
        sht2, sas2, snjets2 = Sht_tot2[mask2], Sas_tot2[mask2], S_njets2[mask2]
        
        case = 'Fixed Menu'
        r_b_val, r_s_val, _, _, _, _, b_EC_val, b_TC_val = Trigger(bht, sht1, sht2, bas, sas1, sas2, bnjets, Ht_cut_fixed, AS_cut_fixed)
        perf_val = (1 - np.abs(r_b_val - 0.25)) * r_s_val
        
        #print('r_b_val',400*r_b_val)
        #'perf', 'cost', 'absrb', 'rs', 'Tcost', 'Ecost'
        
        cases_lists[case]['absrb'].append(np.abs(400*r_b_val - 100))
        #cases_lists[case]['absrb'].append(np.abs(r_b_val - 0.25))
        cases_lists[case]['w0absrb'].append(100*np.abs(r_b_val - 0.25))
        cases_lists[case]['rs'].append(r_s_val)
        cases_lists[case]['w1rs'].append(0.2*(100 - r_s_val))
        cases_lists[case]['Tcost'].append(b_TC_val)
        cases_lists[case]['Ecost'].append(b_EC_val)
        #cases_lists[case]['perf'].append(perf_val)
        cases_lists[case]['cost'].append(b_EC_val + b_TC_val)

        for case, agent in zip(['Standard', 'Anomaly Focused', 'Low-Comp Focused']
        ,[V1_Trigger_localAgent, V2_Trigger_localAgent, V3_Trigger_localAgent]):
        #for case, agent in zip(['Low-Comp Focused'],
        #[V3_Trigger_localAgent]):
            
            Ht_cut = cuts[case]['Ht']
            AS_cut = cuts[case]['AS']
            
            r_b_val, r_s_val, _, _, _, _, b_EC_val, b_TC_val = Trigger(bht, sht1, sht2, bas, sas1, sas2, bnjets, Ht_cut, AS_cut)
            perf_val = (1 - np.abs(r_b_val - 0.25)) * r_s_val

            #cases_lists[case]['absrb'].append(np.abs(r_b_val - 0.25))
            cases_lists[case]['absrb'].append(np.abs(400*r_b_val - 100))
            cases_lists[case]['w0absrb'].append(100*np.abs(r_b_val - 0.25))
            cases_lists[case]['rs'].append(r_s_val)
            cases_lists[case]['w1rs'].append(0.2*(100 - r_s_val))
            cases_lists[case]['Tcost'].append(b_TC_val)
            cases_lists[case]['Ecost'].append(b_EC_val)
            cases_lists[case]['cost'].append(b_EC_val + b_TC_val)
            
            #print(case, ', b_EC_val', b_EC_val)
            #print(case, ', b_TC_val', b_TC_val)
            
            
            
            cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS = agent(bht, sht1, sht2, bas, sas1, sas2, bnjets, Ht_cut, AS_cut)
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            
            #fivefold_window.append([HT[i,j],AS[i,j]])
            #if len(fivefold_window)>=5: 
                #Ht_cut, AS_cut = np.mean(np.array(fivefold_window)[-5:], axis=0)

            cuts[case]['Ht'], cuts[case]['AS'] = HT[i, j], AS[i, j]  # update cuts for next batch
    

        print(f"Processed chunk starting at index {I}")



    #cases_data = {name: {'performance': data['perf'][10:], 'cost': data['cost'][10:]} for name, data in cases_lists.items()}
    #plot_case_comparison(cases_data, n_bins=10, save_path='paper/Summary(dim1).pdf')

    cases_data = {
    name: {
        'w0absrb':  d['w0absrb'][10:],   # or whatever slicing you prefer
        'w1rs':  d['w1rs'][10:],
        'cost':  d['cost'][10:],
        'absrb': d['absrb'][10:],

        'rs':    d['rs'][10:],
        'Tcost': d['Tcost'][10:],
        'Ecost': d['Ecost'][10:]
    }
    for name, d in cases_lists.items()
    }


    #plot_case_comparison1(cases_data, n_bins=5, save_path='paper/Summary4Panel_dim1.pdf')

    plot_case_comparison_split(cases_data, n_bins=5, save_prefix="paper/Summary_dim1_data")