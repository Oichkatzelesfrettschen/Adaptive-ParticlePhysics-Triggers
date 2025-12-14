### Zixin: Final version in training on full data for paper plots
import numpy as np
import random
import os
import matplotlib.pyplot as plt

import atlas_mpl_style as aplt

aplt.use_atlas_style()

from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Lambda, MultiHeadAttention, Add
from keras.models import Sequential, Model
import h5py
import hdf5plugin
import tensorflow as tf
from tensorflow import keras

import mplhep as hep
hep.style.use("CMS")

SEED = 20251208
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def add_cms_header(fig, left_x=0.17, right_x=0.91, y=0.91):
    """Add 'CMS Open Data' on the left and 'Run 283876' on the right."""
    fig.text(left_x, y, "CMS Open Data",
             ha="left", va="top", fontweight="bold", fontsize=24)
    fig.text(right_x, y, "Run 283876",
             ha="right", va="top", fontsize=24)

def sort_obj0(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index,:,:]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index,:,:] = sorted_array

    return data_


def process_h5_file0(input_filename):
    
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
        sorted_data_array = sort_obj0(data_array)

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



def process_h5_file0_newData(input_filename):
    
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 3  
        

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
        sorted_data_array = sort_obj0(data_array)

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
        #sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]

        
        zero_pt_mask = sorted_data_array[:, :, 2] == 0  # Identify where pt == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        npvsGood_smr1_values = npvsGood_smr1_values[1:]
        


        non_zero_ht_mask = Ht_values > 0

        # normalize the column 
        #sorted_data_array[:, :, 2][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        print(sorted_data_array[0,:,2])

        #for i in range(sorted_data_array.shape[0]):
            #if(sorted_data_array[i, 0, 3] <= 0):
                #print(sorted_data_array[i, 0, 3])

        return sorted_data_array, Ht_values, npvsGood_smr1_values




def sort_obj(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index,:,:]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index,:,:] = sorted_array


    data_ = np.transpose(data_, (0,2,1))

    return data_

def process_h5_file(input_filename):

    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events)
        n_jets = 8
        n_features = 3


        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events

        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)

        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt

        
        npvsGood_smr1_values = h5_file['PV_npvsGood_smr1'][:]
        Ht_values = h5_file['ht'][:]

        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values != 0
        data_array = data_array[non_zero_mask]

        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]

        # Add npvsGood_smr1 values to the last column (time column)
        sorted_data_array = sort_obj(data_array)

        #sorted_data_array[:, 3, :] = npvsGood_smr1_values[:, np.newaxis]


        zero_pt_mask = sorted_data_array[:, 2, :] == 0  # Identify where pt == 0
        sorted_data_array[:, 0, :][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, 1, :][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        npvsGood_smr1_values = npvsGood_smr1_values[1:]


        non_zero_ht_mask = Ht_values > 0
        # normalize the column
        #sorted_data_array[:, 2, :][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        sorted_data_array = sorted_data_array.reshape(len(sorted_data_array),-1)
        npvsGood_smr1_values = npvsGood_smr1_values.reshape(len(npvsGood_smr1_values),-1)
        sorted_data_array = np.hstack((sorted_data_array, npvsGood_smr1_values))



        return sorted_data_array, Ht_values




def sort_obj_mc(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index,:,:]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index,:,:] = sorted_array

    return data_

def process_h5_file_mc(input_filename):
    
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
        

        npvsGood_smr1_values = h5_file['PV_npvsGood_smr1'][:]
        Ht_values = h5_file['ht'][:]
        
        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values != 0  
        data_array = data_array[non_zero_mask]  
        
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        
        # Add npvsGood_smr1 values to the last column (time column)
        sorted_data_array = sort_obj0(data_array)
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

        return sorted_data_array, Ht_values


def process_h5_file_newMC(input_filename):
    
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 3  
        

        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events
        
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)
        
        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt
        

        npvsGood_smr1_values = h5_file['PV_npvsGood_smr1'][:]
        Ht_values = h5_file['ht'][:]
        
        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values != 0  
        data_array = data_array[non_zero_mask]  
        
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        
        
        # Add npvsGood_smr1 values to the last column (time column)
        sorted_data_array = sort_obj0(data_array)
        #sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]
  
        
        zero_pt_mask = sorted_data_array[:, :, 2] == 0  # Identify where pt == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        npvsGood_smr1_values = npvsGood_smr1_values[1:]

        non_zero_ht_mask = Ht_values > 0

        # normalize the column 
        #sorted_data_array[:, :, 2][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        print(sorted_data_array[0,:,2])

        return sorted_data_array, Ht_values, npvsGood_smr1_values

@keras.utils.register_keras_serializable()

def select_first_25(x):
    return x[:, :25]  # Takes only the first 25 elements (batch-wise)

@keras.utils.register_keras_serializable()

def repeat_last_element(x):

    last_value = tf.expand_dims(x[:, -1], axis=-1)  # Shape: (batch, 1)
    repeated_last = tf.repeat(last_value, 8, axis=-1)  # Shape: (batch, 8)
    first_24 = x[:, :-1]  # Shape: (batch, 24)
    return tf.concat([first_24, repeated_last], axis=-1)  # Shape: (batch, 32)


def build_autoencoder0(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) 
    
    # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder

def build_autoencoder_data(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size, activation='relu'))   

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape), activation='relu')) 
    decoder.add(Reshape(img_shape))

    return encoder, decoder

def build_attention_autoencoder_jet(img_shape, code_size, num_heads=1):
    inp = Input(shape=img_shape)

    # Self-attention across jets — directly on raw features
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=img_shape[-1])(inp, inp)
    x = Add()([inp, attn_out])  # Residual connection

    # Flatten and encode
    x_flat = Flatten()(x)
    encoded = Dense(code_size)(x_flat)

    # Decode
    x_decoded = Dense(np.prod(img_shape))(encoded)
    x_out = Reshape(img_shape)(x_decoded)

    model = Model(inputs=inp, outputs=x_out)
    return model


def build_autoencoder2(img_shape, code_size):


    # The Encoder
    encoder = Sequential([
        InputLayer(img_shape),  # Input: (8,4)
        #Reshape((4, 8)),  # Step 1: Change (8,4) → (4,8)
        Flatten(),  # Step 2: Flatten (4,8) → (32,)
        Lambda(select_first_25),  # Step 3: Remove last 7 items (keep first 25)
        Dense(code_size)  # Latent space
    ])

    # The Decoder
    decoder = Sequential([
        InputLayer((code_size,)),  # Latent space input
        Dense(25),  # Expand back to 25 elements
        Lambda(repeat_last_element),  # Explicitly duplicate last element to reach 32
        Reshape((4,8))  # Reshape back to (4,8)
        #Lambda(lambda x: tf.transpose(x, perm=[1, 0]))  # Swap (4,8) to (8,4)

    ])

    return encoder, decoder


def build_autoencoder1(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    #encoder.add(Dense(hidden_size))
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    #decoder.add(Dense(hidden_size))
    decoder.add(Dense(np.prod(img_shape)))

    # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    #decoder.add(Reshape(img_shape))

    return encoder, decoder

def calculate_score(autoencoder_, image_):

    reconstructed_image_ = autoencoder_.predict(image_[None],verbose=0)[0]
    mse_loss = np.mean(np.square(image_ - reconstructed_image_))
    #mse_loss = masked_mse_loss(image_,reconstructed_image_)

    return mse_loss

@keras.utils.register_keras_serializable()

def masked_mse_loss(y_true, y_pred):


    # Create a boolean mask where valid jets have eta >= 0, phi >= 0, pt > 0
    valid_mask = tf.logical_and(
        tf.logical_and(y_true[:, :, 0] >= 0, y_true[:, :, 1] >= 0),
        y_true[:, :, 2] > 0
    )  # Shape: (batch, 8)

    # Expand mask to match (batch, 8, 4)
    valid_mask_expanded = tf.expand_dims(tf.cast(valid_mask, tf.float32), -1)  # Shape: (batch, 8, 1)

    # Compute squared error only for valid jets
    squared_error = tf.square(y_true - y_pred) * valid_mask_expanded

    # Normalize by the number of valid elements (avoid divide-by-zero)
    num_valid = tf.reduce_sum(valid_mask_expanded,axis=[1, 2]) + 1e-8
    loss = tf.reduce_sum(squared_error, axis=[1, 2]) / num_valid

    return tf.math.log1p(loss)



def mse_loss(y_true, y_pred):

    squared_error = tf.abs(y_true - y_pred)

    num = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

    loss = tf.reduce_sum(squared_error, axis=[1, 2]) / num

    return tf.math.log1p(loss)

def calculate_batch_score(autoencoder_, images_):
    reconstructed_images_ = autoencoder_.predict(images_, verbose=0)
    assert reconstructed_images_.shape == images_.shape, (
        f"Shape mismatch: reconstructed_images_ has shape {reconstructed_images_.shape}, "
        f"but images_ has shape {images_.shape}"
    )
    #mse_losses = np.mean(np.square(images_ - reconstructed_images_), axis=(1,2))
    mse_losses = np.mean(np.square(images_ - reconstructed_images_), axis=1)
    #mse_losses = masked_mse_loss(images_ , reconstructed_images_)

    return mse_losses

@keras.utils.register_keras_serializable()

def calculate_score_tensor(autoencoder_, images_):

    if len(images_.shape) == 2:  # If input is (8,4) instead of (batch, 8,4)
        images_ = images_[None]

    print(images_.shape)
    # Predict the reconstructed image
    reconstructed_images_ = autoencoder_.predict(images_, verbose=0)

    print(reconstructed_images_.shape)

    # Convert to TensorFlow tensors
    image_tensor = tf.convert_to_tensor(images_, dtype=tf.float32)
    reconstructed_tensor = tf.convert_to_tensor(reconstructed_images_, dtype=tf.float32)

    # Compute masked MSE loss (returns a Tensor)
    mse_losses = masked_mse_loss(image_tensor, reconstructed_tensor)
    #mse_losses = mse_loss(image_tensor, reconstructed_tensor)
    #mse_loss = tf.math.log1p(mse_loss)
    print(mse_losses.numpy().shape)


    return mse_losses.numpy()  # Convert Tensor to a NumPy scalar


#mc_bkg_jets, mc_bkg_ht = process_h5_file0("Data/data_Run_2016_283876.h5")
mc_bkg_jets, mc_bkg_ht, mc_bkg_npv= process_h5_file0_newData("Data/data_Run_2016_283876.h5") #Data/data_Run_2016_283876.h5
# mc_bkg_jets, mc_bkg_ht, mc_bkg_npv= process_h5_file_newMC("../MinBias_1.h5") #Data/MinBias_1.h5
#mc_bkg_jets, mc_bkg_ht = process_h5_file_mc("Data/MinBias_1.h5")
#X1 = mc_bkg_jets[::100]
#Npv1 = mc_bkg_jets[::100,0,3]
#Jets1 = mc_bkg_jets[::100, :, :-1]
#HT1 = mc_bkg_ht[::100]
#mc_bkg_jets = mc_bkg_jets[::10000]
#mc_bkg_ht = mc_bkg_ht[::10000]
#mc_bkg_npv = mc_bkg_npv[::10000]

Jets1 = mc_bkg_jets.reshape(mc_bkg_jets.shape[0],-1)
#Npv1 = mc_bkg_jets[:,0,3]
Npv1  = mc_bkg_npv.reshape(-1, 1)

X1 = np.concatenate([Jets1, Npv1], axis=1)
#Jets1 = mc_bkg_jets[:, :, :-1]
#X1 = np.concatenate((Jets1, Npv1), axis=1)
HT1 = mc_bkg_ht

K = 10  # number of subsamples
N_total = len(X1)

indices = np.arange(N_total)
np.random.shuffle(indices)

# Split into K folds (approx equal size)
folds = np.array_split(indices, K)


print('X1.shape: ', X1.shape)
print('Jets1.shape',Npv1.shape)

#Ht1_all = mc_bkg_ht[::100]
N1 = len(X1)

#X2 = np.load('Data/Jet2_bkg_h5.npy',allow_pickle=True)
#N2 = len(X2)


#mc_AA_jets, mc_AA_ht = process_h5_file0("Data/HToAATo4B.h5")
#mc_AA_jets, mc_AA_ht = process_h5_file_mc("Data/HToAATo4B.h5")
mc_AA_jets, mc_AA_ht, mc_aa_npv = process_h5_file0_newData("Data/HToAATo4B.h5") #Data/HToAATo4B.h5
# mc_AA_jets, mc_AA_ht, mc_aa_npv = process_h5_file_newMC("../HToAATo4B.h5") #Data/HToAATo4B.h5
#JetsAA = mc_AA_jets
JetsAA = mc_AA_jets.reshape(mc_AA_jets.shape[0],-1)
#NpvAA = mc_AA_jets[:,0,3]
NpvAA = mc_aa_npv.reshape(-1, 1)
#JetsAA = mc_AA_jets[:,:, :-1]
#HTAA = mc_AA_ht[::100]
X_AA = np.concatenate([JetsAA, NpvAA], axis=1)
HTAA = mc_AA_ht

print('X_AA.shape',X_AA.shape)
N_tt = len(X_AA)
print(JetsAA.shape, NpvAA.shape)


#mc_tt_jets, mc_tt_ht = process_h5_file0("Data/TT_1.h5")
#mc_tt_jets, mc_tt_ht, mc_tt_npv = process_h5_file_mc("Data/TT_1.h5")
mc_tt_jets, mc_tt_ht, mc_tt_npv = process_h5_file0_newData("Data/TT_1.h5") #Data/TT_1.h5
# mc_tt_jets, mc_tt_ht, mc_tt_npv = process_h5_file_newMC("../TT_1.h5") #Data/TT_1.h5
#Jetstt = mc_tt_jets
Jetstt = mc_tt_jets.reshape(mc_tt_jets.shape[0],-1)
#Npvtt = mc_tt_jets[:,0,3]
Npvtt = mc_tt_npv.reshape(-1, 1)
#Jetstt = mc_tt_jets[:,:, :-1]
X_tt = np.concatenate([Jetstt, Npvtt], axis=1)
#HTtt = mc_tt_ht[::100]
HTtt = mc_tt_ht

print('X_tt.shape',X_tt.shape)
N_tt = len(X_tt)
#print(Jetstt.shape, Npvtt.shape)




#X_train, X_test, Jets1_train, Jets1_test, Npv1_train, Npv1_test, HT_train, HT_test = train_test_split(X1, Jets1, Npv1, HT1, test_size=0.2, random_state=42)

#X_train, X_test, Ht_train, Ht_test = train_test_split(X1, Ht1_all, test_size=0.6, random_state=42)


# Same as (8,4), we neglect the number of instances from shape
IMG_SHAPE_X = X1.shape[1:]
IMG_SHAPE_Jets = Jets1.shape[1:]
IMG_SHAPE = [IMG_SHAPE_X, IMG_SHAPE_Jets]
#Train = [X_train, Jets1_train]
#Test = [X_test, Jets1_test]
AA = [X_AA, JetsAA]
TT = [X_tt, Jetstt]

#Dim = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#Dim = [2,4,6,8,10,12]
#Dim = [1,2,4,8,9,10,12]
Dim = [1, 2, 4, 8, 16]
AA_pass_matrix = np.zeros((K, len(Dim)))
TT_pass_matrix = np.zeros((K, len(Dim)))


AA_ht_pass_matrix = np.zeros((K, len(Dim)))
TT_ht_pass_matrix = np.zeros((K, len(Dim)))

results_npv = {}




'''
results_dim = {}
for index, dim in enumerate(Dim):
    encoder, decoder = build_autoencoder0(IMG_SHAPE_X, dim)
    #autoencoder = build_autoencoder0(IMG_SHAPE_X, dim)

    inp = Input(IMG_SHAPE_X)
    code = encoder(inp)
    reconstruction = decoder(code)



    autoencoder = Model(inp,reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')
    #autoencoder.compile(optimizer='adamax', loss=masked_mse_loss)

    print(autoencoder.summary())

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    history = autoencoder.fit(x=Train[0], y=Train[0], epochs=100, validation_data=[Test[0], Test[0]],callbacks=[es])

    bkg_test_scores = calculate_batch_score(autoencoder, Test[0])
    AA_test_scores = calculate_batch_score(autoencoder, AA[0])
    TT_test_scores = calculate_batch_score(autoencoder, TT[0])
    percen_9975 = np.percentile(bkg_test_scores, 99.75)

    # Compute percentage of ttbar passing this threshold
    AA_passed = 100 * np.sum(AA_test_scores > percen_9975) / len(AA_test_scores)
    TT_passed = 100 * np.sum(TT_test_scores > percen_9975) / len(TT_test_scores)



    results_dim[index] = {
        "bkg_scores": bkg_test_scores,
        "AA_scores": AA_test_scores,
        "TT_scores": TT_test_scores,
        "history": history.history,
        "AA_pass":AA_passed,
        "TT_pass":TT_passed
    }
    if dim==10 or dim==7:
        autoencoder.save(f'autoencoder_model0_{dim}.keras')

#print('Npv:',Npv1_test.shape,results_npv[0]["bkg_scores"].shape)

'''
results_dim = {}
for k in range(K):
    print(f"\n========== Processing subsample {k+1}/{K} ==========\n")
    idx = folds[k]

    # build the subsample
    X_sub   = X1[idx]
    Jets_sub = Jets1[idx]
    Npv_sub = Npv1[idx]
    HT_sub  = HT1[idx]

    # split train/test inside this subsample
    X_train, X_test, HT_train, HT_test = train_test_split(X_sub, HT_sub, test_size=0.2, random_state=40)

    for j, dim in enumerate(Dim):
        
        #encoder, decoder = build_autoencoder0(IMG_SHAPE_X, dim)
        encoder, decoder = build_autoencoder_data(IMG_SHAPE_X, dim)
        
        #N_train = 2500*dim**2
        
        
        #final_N = min(N_train,len(X_train))
        #print('final_N',final_N)
        
        #X_train_dim = X_train[:final_N]
        #X_test_dim = X_test[:final_N]
        
        inp = Input(IMG_SHAPE_X)
        code = encoder(inp)
        reconstruction = decoder(code)
        autoencoder = Model(inp, reconstruction)
        autoencoder.compile(optimizer='adamax', loss='mse')

        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = autoencoder.fit(
            X_train, X_train,
            epochs=100,
            validation_data=(X_test, X_test),
            callbacks=[es],
            #verbose=0
        )
        

        # compute anomaly scores
        bkg_test_scores = calculate_batch_score(autoencoder, X_test)
        AA_test_scores = calculate_batch_score(autoencoder, X_AA)
        TT_test_scores = calculate_batch_score(autoencoder, X_tt)

        thr = np.percentile(bkg_test_scores, 99.75)

        AA_pass = 100 * np.mean(AA_test_scores > thr)
        TT_pass = 100 * np.mean(TT_test_scores > thr)

        # store result
        AA_pass_matrix[k, j] = AA_pass
        TT_pass_matrix[k, j] = TT_pass
        
        
        percen_9975_ht = np.percentile(HT_test, 99.75)


        AA_Ex_pass = 100 * np.mean((AA_test_scores > thr)& ~(HTAA > percen_9975_ht))
        TT_Ex_pass = 100 * np.mean((TT_test_scores > thr)& ~(HTtt > percen_9975_ht))
        
        AA_ht_pass_matrix[k, j] = AA_Ex_pass
        TT_ht_pass_matrix[k, j] = TT_Ex_pass
        
        #autoencoder.save(f'autoencoder_modelNewData_{dim}.keras')
        '''
        results_dim[dim] = {
        "bkg_scores": bkg_test_scores,
        "AA_scores": AA_test_scores,
        "TT_scores": TT_test_scores,
        "history": history.history,
        "AA_pass":AA_pass,
        "TT_pass":TT_pass,
        "AA_Ex_pass":AA_Ex_pass,
        "TT_Ex_pass":TT_Ex_pass
        }
        '''
        print(f"Subsample {k+1}, dim {dim}: AA={AA_pass:.2f}%  TT={TT_pass:.2f}%")


#print('AA_ht_passed:', AA_mean_ht, AA_std_ht)
#print('TT_ht_passed:', TT_mean_ht, TT_std_ht)
'''
plt.figure(figsize=(10,6))

plt.errorbar(Dim, AA_mean, yerr=AA_std, fmt='o-', capsize=4,
             label='HToAATo4B', color='green',markersize=1)

plt.errorbar(Dim, TT_mean, yerr=TT_std, fmt='o-', capsize=4,
             label='TTbar', color='goldenrod',markersize=1)

plt.xlabel("Latent Dimension", fontsize=13)
plt.ylabel("Efficiency [%]", fontsize=13)
plt.title("10 Random Subsamples of size 50k, with 50% test size", fontsize=14)
plt.grid(alpha=0.3)
plt.xticks(Dim)
plt.legend()
plt.tight_layout()
plt.savefig("RealData/paper_data/random_newData.pdf")


fig, ax = plt.subplots(figsize=(8,6))

ax.errorbar(Dim, AA_mean, yerr=AA_std, fmt='o-', color='green',
            label="HToAATo4B")
ax.errorbar(Dim, TT_mean, yerr=TT_std, fmt='o-', color='orange',
            label="TTbar")




ax.set_xlabel("Latent Dimension")
ax.set_ylabel("Efficiency [%]")
#ax.set_title("Signal / TT Efficiency vs Latent Dimension\nMean over 10 random subsamples")
ax.legend()
ax.grid(True)


ax.legend(fontsize=10, frameon=True, loc='best', bbox_transform=ax.transData)


#fig.tight_layout(rect=[0, 0, 1, 0.93])
add_cms_header(fig)
fig.savefig("RealData/paper_data/signal_pass_vs_dimension_data_split.pdf")
fig.savefig("RealData/paper_data/signal_pass_vs_dimension_data_split.png")

plt.close(fig)

'''

'''


# Extracting data for d=1 and d=4 from results_dim
dim_1 = results_dim[1] 
dim_2 = results_dim[2]  
dim_4 = results_dim[4]  

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for dim = 1
threshold1 = np.percentile(dim_1["bkg_scores"],99.995)
#range=(0, threshold)

# overflow bin
n_bins = 50
bins1 = np.linspace(0, threshold1, n_bins)

# Clip scores: anything above threshold1 goes into threshold1
bkg_scores = np.clip(dim_1["bkg_scores"], None, threshold1)
AA_scores  = np.clip(dim_1["AA_scores"],  None, threshold1)
TT_scores  = np.clip(dim_1["TT_scores"],  None, threshold1)

axes[0].hist(bkg_scores, bins=bins1, alpha=1, density=True, label='test Real Data Background', histtype='step', color='tab:blue', linewidth=2)
axes[0].hist(AA_scores, bins=bins1, alpha=1, density=True, label='HToAATo4B Signal', histtype='step', color='tab:green', linewidth=2)
axes[0].hist(TT_scores, bins=bins1, alpha=1, density=True, label='TTbar Signal', histtype='step', color='goldenrod', linewidth=2)

axes[0].set_xlabel('Anomaly Score') #, fontsize=18
axes[0].set_ylabel('Density') #, fontsize=18
axes[0].set_yscale("log")
#axes[0].set_xlim(-0.1, 600)
axes[0].set_xlim(-0.1, threshold1+1)
#axes[0].set_ylim(10E-6,10)

# Add note about overflows
axes[0].legend(fontsize=14, loc='best', frameon=True, title='Latent Dimension = 1\n(Overflows in the Last Bin', title_fontsize=14) #,

# Plot for dim = 4
threshold4 = np.percentile(dim_4["bkg_scores"],99.995)

bins4 = np.linspace(0, threshold4, n_bins)

bkg_scores4 = np.clip(dim_4["bkg_scores"], None, threshold4)
AA_scores4  = np.clip(dim_4["AA_scores"],  None, threshold4)
TT_scores4  = np.clip(dim_4["TT_scores"],  None, threshold4)

axes[1].hist(bkg_scores4, bins=bins4, alpha=1, density=True, label='test MinBias Background', histtype='step', color='tab:blue',linewidth=2)
axes[1].hist(AA_scores4, bins=bins4, alpha=1, density=True, label='HToAATo4B Signal', histtype='step', color='tab:green',linewidth=2)
axes[1].hist(TT_scores4, bins=bins4, alpha=1, density=True, label='TTbar Signal', histtype='step', color='goldenrod',linewidth=2)
#axes[1].set_title('Anomaly Score Distribution (d=4)')
axes[1].set_xlabel('Anomaly Score')
axes[1].set_ylabel('Density') #, fontsize=18
axes[1].set_yscale("log")
#axes[1].set_xlim(-0.1,120)
axes[1].set_xlim(-0.1, threshold4+1)
#axes[1].set_ylim(10E-6,10)
#axes[1].legend(fontsize=14,title='Latent Dimension = 4', title_fontsize=16, loc='best', frameon=True)
axes[1].legend(fontsize=14, loc='best', frameon=True, title='Latent Dimension = 4\n(Overflows in the Last Bin)', title_fontsize=14) #, title_fontsize=14
# Adjust layout and save

plt.tight_layout()



#plt.savefig("paper/AS_hist_comparison2016.pdf")
fig.savefig("RealData/paper_data/AD_hist_comparison2016-newdata14.pdf")


import matplotlib.transforms as mtransforms

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


save_subplot(fig, axes[0], "RealData/paper_data/AD_hist_comparison2016-a-newdata.pdf", pad=0.3)
save_subplot(fig, axes[1], "RealData/paper_data/AD_hist_comparison2016-b-newdata.pdf", pad=0.3)
plt.close(fig)


#end = time.time()
#print(f"Runtime: {end - start:.4f} seconds")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for dim = 1
threshold2 = np.percentile(dim_2["bkg_scores"],99.995)
#range=(0, threshold)

# overflow bin
n_bins = 50
bins2 = np.linspace(0, threshold2, n_bins)

# Clip scores: anything above threshold1 goes into threshold1
bkg_scores = np.clip(dim_2["bkg_scores"], None, threshold2)
AA_scores  = np.clip(dim_2["AA_scores"],  None, threshold2)
TT_scores  = np.clip(dim_2["TT_scores"],  None, threshold2)

axes[0].hist(bkg_scores, bins=bins2, alpha=1, density=True, label='test Real Data Background', histtype='step', color='tab:blue', linewidth=2)
axes[0].hist(AA_scores, bins=bins2, alpha=1, density=True, label='HToAATo4B Signal', histtype='step', color='tab:green', linewidth=2)
axes[0].hist(TT_scores, bins=bins2, alpha=1, density=True, label='TTbar Signal', histtype='step', color='goldenrod', linewidth=2)

axes[0].set_xlabel('Anomaly Score') #, fontsize=18
axes[0].set_ylabel('Density') #, fontsize=18
axes[0].set_yscale("log")
axes[0].set_xlim(-0.1, threshold2+1)
#axes[0].set_ylim(10E-6,10)

# Add note about overflows
axes[0].legend(fontsize=14, loc='best', frameon=True, title='Latent Dimension = 2\n(Overflows in the Last Bin', title_fontsize=14) #,

plt.tight_layout()



#plt.savefig("paper/AS_hist_comparison2016.pdf")
fig.savefig("RealData/paper_data/AD_hist_comparison2016-newdata2.pdf")

save_subplot(fig, axes[0], "RealData/paper_data/AD_hist_comparison2016-a-newdata2.pdf", pad=0.3)
save_subplot(fig, axes[1], "RealData/paper_data/AD_hist_comparison2016-b-newdata2.pdf", pad=0.3)
plt.close(fig)

'''



percen_9975_ht = np.percentile(HT1, 99.75)

AA_ht_passed = 100 * np.mean(HTAA > percen_9975_ht) 
TT_ht_passed = 100 * np.mean(HTtt > percen_9975_ht) 

print('AA_ht_passed:', AA_ht_passed)
print('TT_ht_passed:', TT_ht_passed)

'''
# --- Meeting mode: forza la curva e le linee ---
AA_pass_rates = [results_dim[index]["AA_pass"] for index in Dim]
TT_pass_rates = [results_dim[index]["TT_pass"] for index in Dim]  

AA_Ex_pass_rates = [results_dim[index]["AA_Ex_pass"] for index in Dim]
TT_Ex_pass_rates = [results_dim[index]["TT_Ex_pass"] for index in Dim]  

'''
AA_mean = np.mean(AA_pass_matrix, axis=0)
AA_std  = np.std(AA_pass_matrix, axis=0) / np.sqrt(K)   

TT_mean = np.mean(TT_pass_matrix, axis=0)
TT_std  = np.std(TT_pass_matrix, axis=0) / np.sqrt(K)

AA_mean_Ex = np.mean(AA_ht_pass_matrix, axis=0)
AA_std_Ex  = np.std(AA_ht_pass_matrix, axis=0) / np.sqrt(K)   

TT_mean_Ex = np.mean(TT_ht_pass_matrix, axis=0)
TT_std_Ex  = np.std(TT_ht_pass_matrix, axis=0) / np.sqrt(K)


Dim_log2 = np.log2(Dim) 



fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(18, 12), sharex=True,
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05}
)

# ---------------------------
# TOP PANEL — Efficiency
# ---------------------------

ax1.errorbar(Dim_log2, AA_mean, yerr=AA_std, fmt='o-', color='tab:green',
         label="HToAAto4B, AD Efficiency",markersize=1)
ax1.errorbar(Dim_log2, TT_mean, yerr=TT_std, fmt='o-', color='goldenrod',
         label="TTBar, AD Efficiency",markersize=1)

# HT lines
ax1.hlines(AA_ht_passed, 0, 4, color="grey", linestyles="dashed",
           label=f"HToAATo4B, HT Efficiency: {AA_ht_passed:.2f}%")
ax1.hlines(TT_ht_passed, 0, 4, color="grey", linestyles="dashed",
           label=f"TTBar, HT Efficiency: {TT_ht_passed:.2f}%")

ax1.set_ylabel("Total Efficiency (%)")
#ax1.set_ylim(0, 120)
ax1.grid(True)

ax1.legend(fontsize=16, frameon=True, loc='best',ncol=2)

# ---------------------------
# BOTTOM PANEL — Exclusive Eff
# ---------------------------
ax2.errorbar(Dim_log2, AA_mean_Ex, yerr=AA_std_Ex, fmt='o--', color='tab:green',
         label="HToAAto4B", markersize=1)
ax2.errorbar(Dim_log2, TT_mean_Ex, yerr=TT_std_Ex, fmt='o--', color='goldenrod',
         label="TTBar", markersize=1)

ax2.set_xlabel(r'$\log_{2}(\mathrm{Latent\ Dimension})$')
ax2.set_ylabel("Exclusive AD Efficiency (%)")
#ax2.set_ylim(0, 120)
ax2.grid(True)

#ax2.legend(fontsize=14, frameon=True, loc='best')

#ax2.legend(fontsize=16, frameon=True, loc='best')

# ---------------------------
# Shared X settings
# ---------------------------
ax2.set_xlim(-0.05, 4.5)
ax2.set_xticks([0,1,2,3,4])
ax2.set_xticklabels(["0", "1", "2", "3", "4"])

#fig.tight_layout()
#add_cms_header(fig)
fig.savefig("RealData/paper_data/signal_pass_vs_dimension_NewData.pdf")
plt.close(fig)



