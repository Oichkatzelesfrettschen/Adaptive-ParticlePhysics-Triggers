
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import hdf5plugin
#from AE import select_first_25, repeat_last_element, masked_mse_loss, calculate_score_tensor

def calculate_batch_loss(autoencoder_, images_):
    reconstructed_images_ = autoencoder_.predict(images_, verbose=0)
    assert reconstructed_images_.shape == images_.shape, (
        f"Shape mismatch: reconstructed_images_ has shape {reconstructed_images_.shape}, "
        f"but images_ has shape {images_.shape}"
    )
    mse_losses = np.mean(np.square(images_ - reconstructed_images_), axis=(1, 2))  
    
    return mse_losses

def calculate_H_met(data_array, Ht_values):

    # Extract  Pt and Phi
    pt_norm = data_array[:, :, 2]  
    phi = data_array[:, :, 1]  

    # Denormalize 
    pt_actual = pt_norm * Ht_values[:, np.newaxis]  # Restore actual Pt (shape: n_events, n_jets)

    # Compute px and py components
    px = pt_actual * np.cos(phi)
    py = pt_actual * np.sin(phi)

    
    sum_px = np.sum(px, axis=1)  
    sum_py = np.sum(py, axis=1)  

    
    H_met_values = np.sqrt(sum_px**2 + sum_py**2)  # (n_events,)

    return H_met_values

def sort_obj(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index,:,:]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index,:,:] = sorted_array

    return data_
    
def process_h5_file(input_filename):
    
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 4  
        

        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events
        
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)
        
        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
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



# Load the saved model
# Giovanna's paths
# autoencoder01 = load_model('autoencoder_model0_1.keras')
# autoencoder04 = load_model('autoencoder_model0_4.keras')

#Giovanna's paths
# mc_bkg_jets, mc_bkg_ht = process_h5_file("new_Data/data_Run_2016_283408_longest.h5")
# mc_aa_jets, mc_aa_ht = process_h5_file("new_Data/HToAATo4B.h5")
# #data_jets, data_ht = process_h5_file("Data/data_sortedByEvtNo.h5")
# mc_tt_jets, mc_tt_ht = process_h5_file("new_Data/ttbar_2016_2.h5")

#Zixin's paths
autoencoder01 = load_model('Data/python/autoencoder_model0_1_realdata.keras')
autoencoder04 = load_model('Data/python/autoencoder_model0_4_realdata.keras')

mc_bkg_jets, mc_bkg_ht = process_h5_file("Data/data_Run_2016_283408_longest.h5")
mc_aa_jets, mc_aa_ht = process_h5_file("Data/HToAATo4B.h5")
#data_jets, data_ht = process_h5_file("Data/data_sortedByEvtNo.h5")
mc_tt_jets, mc_tt_ht = process_h5_file("Data/TT_1.h5")


#print(data.shape, ht_values.shape)

mc_bkg_npvs = mc_bkg_jets[:,0,3]
mc_aa_npvs = mc_aa_jets[:,0,3]
#data_npvs = data_jets[:,0,3]
mc_tt_npvs = mc_tt_jets[:,0,3]

aa_key = [1 for i in range(len(mc_aa_npvs))]
tt_key = [2 for i in range(len(mc_tt_npvs))]


#mc_bkg_jets = np.transpose(mc_bkg_jets, (0, 2, 1)) 
#mc_tt_jets = np.transpose(mc_tt_jets, (0, 2, 1)) 
#data_jets = np.transpose(data_jets, (0, 2, 1)) 
#mc_dijet_jets = np.transpose(mc_dijet_jets, (0, 2, 1)) 

mc_bkg_scores01 = calculate_batch_loss(autoencoder01,mc_bkg_jets)
mc_aa_scores01 = calculate_batch_loss(autoencoder01,mc_aa_jets)
#data_scores01 = calculate_batch_loss(autoencoder01,data_jets)
mc_tt_scores01 = calculate_batch_loss(autoencoder01,mc_tt_jets)

mc_bkg_scores04 = calculate_batch_loss(autoencoder04,mc_bkg_jets)
mc_aa_scores04 = calculate_batch_loss(autoencoder04,mc_aa_jets)
#data_scores04 = calculate_batch_loss(autoencoder04,data_jets)
mc_tt_scores04 = calculate_batch_loss(autoencoder04,mc_tt_jets)

#computational cost
#n_jets
mc_bkg_njets = np.sum(mc_bkg_jets[:, :, 2] > 0, axis=1)
mc_aa_njets = np.sum(mc_aa_jets[:, :, 2] > 0, axis=1)
mc_tt_njets = np.sum(mc_tt_jets[:, :, 2] > 0, axis=1)
#data_njets = np.sum(data_jets[:, :, 2] > 0, axis=1)

print('mc_bkg_njets: ', mc_bkg_njets.shape)
print('mc_aa_njets: ',mc_aa_njets.shape)
print('mc_tt_njets: ', mc_tt_njets.shape)
#print('data_njets: ', data_njets.shape)




# Giovanna's output path
# output_filename = 'new_Data/Trigger_food.h5'

# Zixin's output path
output_filename = 'Data/Trigger_food_Data.h5'
with h5py.File(output_filename, 'w') as output_file:
        output_file.create_dataset("mc_bkg_ht", data=np.array(mc_bkg_ht, dtype=np.float32))
        #output_file.create_dataset("mc_bkg_Hmets", data=np.array(mc_bkg_Hmets, dtype=np.float32))
        output_file.create_dataset("mc_bkg_score01", data=np.array(mc_bkg_scores01, dtype=np.float32))
        output_file.create_dataset("mc_bkg_score04", data=np.array(mc_bkg_scores04, dtype=np.float32))
        output_file.create_dataset("mc_bkg_Npv", data=np.array(mc_bkg_npvs, dtype=np.float32))
        output_file.create_dataset("mc_bkg_njets", data=np.array(mc_bkg_njets, dtype=np.float32))

        output_file.create_dataset("mc_aa_ht", data=np.array(mc_aa_ht, dtype=np.float32))
        #output_file.create_dataset("mc_aa_Hmets", data=np.array(mc_aa_Hmets, dtype=np.float32))
        output_file.create_dataset("mc_aa_score01", data=np.array(mc_aa_scores01, dtype=np.float32))
        output_file.create_dataset("mc_aa_score04", data=np.array(mc_aa_scores04, dtype=np.float32))
        output_file.create_dataset("aa_Npv", data=np.array(mc_aa_npvs, dtype=np.float32))
        output_file.create_dataset("aa_key", data=np.array(aa_key, dtype=np.int32))
        output_file.create_dataset("mc_aa_njets", data=np.array(mc_aa_njets, dtype=np.float32))

        output_file.create_dataset("mc_tt_ht", data=np.array(mc_tt_ht, dtype=np.float32))
        #output_file.create_dataset("mc_tt_Hmets", data=np.array(mc_tt_Hmets, dtype=np.float32))
        output_file.create_dataset("mc_tt_score01", data=np.array(mc_tt_scores01, dtype=np.float32))
        output_file.create_dataset("mc_tt_score04", data=np.array(mc_tt_scores04, dtype=np.float32))
        output_file.create_dataset("tt_Npv", data=np.array(mc_tt_npvs, dtype=np.float32))
        output_file.create_dataset("tt_key", data=np.array(tt_key, dtype=np.int32))
        output_file.create_dataset("mc_tt_njets", data=np.array(mc_tt_njets, dtype=np.float32))

        '''
        output_file.create_dataset("data_ht", data=np.array(data_ht, dtype=np.float32))
        output_file.create_dataset("data_Hmets", data=np.array(data_Hmets, dtype=np.float32))
        output_file.create_dataset("data_score01", data=np.array(data_scores01, dtype=np.float32))
        output_file.create_dataset("data_score04", data=np.array(data_scores04, dtype=np.float32))
        output_file.create_dataset("data_Npv", data=np.array(data_npvs, dtype=np.float32))
        output_file.create_dataset("data_njets", data=np.array(data_njets, dtype=np.float32))
        '''

        print(f"Results saved to {output_filename}")
        
# Plot the Npv distributions
plt.figure(figsize=(10, 5))
plt.hist(mc_bkg_npvs, bins=50, alpha=0.5, label="Background", histtype="step", linewidth=2)
plt.hist(mc_aa_npvs, bins=50, alpha=0.5, label="HToAATo4B", histtype="step", linewidth=2)
plt.hist(mc_tt_npvs, bins=50, alpha=0.5, label="ttbar", histtype="step", linewidth=2)

plt.title("Distribution of Npv (PV_npvsGood_smr1)")
plt.xlabel("Npv")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("paper/Npv_distributions.png")  # facoltativo: salva su file
plt.show()
