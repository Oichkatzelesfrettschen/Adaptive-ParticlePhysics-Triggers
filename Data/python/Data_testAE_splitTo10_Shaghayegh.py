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
from pathlib import Path
import mplhep as hep
hep.style.use("CMS")

#Shaghayegh does not have this Zixin add it
SEED = 20251102  # fixing the seed
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
    encoder.add(Dense(code_size, activation='relu'))   # ← minimal fix

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape), activation='relu'))  # ← minimal fix
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
    mse_losses = np.mean(np.square(images_ - reconstructed_images_), axis=(1,2))
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


# mc_bkg_jets, mc_bkg_ht = process_h5_file0("Data/data_Run_2016_283876.h5")
mc_bkg_jets, mc_bkg_ht = process_h5_file0("../data_Run_2016_283876.h5")

#mc_bkg_jets, mc_bkg_ht = process_h5_file_mc("Data/MinBias_1.h5")
#X1 = mc_bkg_jets[::100]
#Npv1 = mc_bkg_jets[::100,0,3]
#Jets1 = mc_bkg_jets[::100, :, :-1]
#HT1 = mc_bkg_ht[::100]

X1 = mc_bkg_jets
Npv1 = mc_bkg_jets[:,0,3]
Jets1 = mc_bkg_jets[:, :, :-1]
HT1 = mc_bkg_ht

K = 10  # number of subsamples
N_total = len(X1)

indices = np.arange(N_total)
np.random.shuffle(indices)

# Split into K folds (approx equal size)
folds = np.array_split(indices, K)


print(Jets1.shape, Npv1.shape)

#Ht1_all = mc_bkg_ht[::100]
N1 = len(X1)

#X2 = np.load('Data/Jet2_bkg_h5.npy',allow_pickle=True)
#N2 = len(X2)


# mc_AA_jets, mc_AA_ht = process_h5_file0("Data/HToAATo4B.h5")
mc_AA_jets, mc_AA_ht = process_h5_file0("../HToAATo4B.h5")

#mc_AA_jets, mc_AA_ht = process_h5_file_mc("Data/HToAATo4B.h5")
X_AA = mc_AA_jets
NpvAA = mc_AA_jets[:,0,3]
JetsAA = mc_AA_jets[:,:, :-1]
#HTAA = mc_AA_ht[::100]
HTAA = mc_AA_ht

print('X_AA.shape',X_AA.shape)
N_tt = len(X_AA)
print(JetsAA.shape, NpvAA.shape)


# mc_tt_jets, mc_tt_ht = process_h5_file0("Data/TT_1.h5")
mc_tt_jets, mc_tt_ht = process_h5_file0("../TT_1.h5")

#mc_tt_jets, mc_tt_ht = process_h5_file_mc("Data/TT_1.h5")
X_tt = mc_tt_jets
Npvtt = mc_tt_jets[:,0,3]
Jetstt = mc_tt_jets[:,:, :-1]
#HTtt = mc_tt_ht[::100]
HTtt = mc_tt_ht

print('X_tt.shape',X_tt.shape)
N_tt = len(X_tt)
print(Jetstt.shape, Npvtt.shape)




X_train, X_test, Jets1_train, Jets1_test, Npv1_train, Npv1_test, HT_train, HT_test = train_test_split(X1, Jets1, Npv1, HT1, test_size=0.5, random_state=42)

#X_train, X_test, Ht_train, Ht_test = train_test_split(X1, Ht1_all, test_size=0.6, random_state=42)


# Same as (8,4), we neglect the number of instances from shape
IMG_SHAPE_X = X1.shape[1:]
IMG_SHAPE_Jets = Jets1.shape[1:]
IMG_SHAPE = [IMG_SHAPE_X, IMG_SHAPE_Jets]
Train = [X_train, Jets1_train]
Test = [X_test, Jets1_test]
AA = [X_AA, JetsAA]
TT = [X_tt, Jetstt]

Dim = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#Dim = [1,2,4,8,9,10,12]
#Dim = [1, 4]
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

for k in range(K):
    print(f"\n========== Processing subsample {k+1}/{K} ==========\n")
    idx = folds[k]

    # build the subsample
    X_sub   = X1[idx]
    Jets_sub = Jets1[idx]
    Npv_sub = Npv1[idx]
    HT_sub  = HT1[idx]

    # split train/test inside this subsample
    X_train, X_test = train_test_split(X_sub, test_size=0.5, random_state=40)

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
            verbose=0
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
        
        
        #percen_9975_ht = np.percentile(HT_test, 99.75)


        #AA_ht_passed = 100 * np.sum(HTAA > percen_9975_ht) / len(HTAA)
        #TT_ht_passed = 100 * np.sum(HTtt > percen_9975_ht) / len(HTtt)
        
        #AA_ht_pass_matrix[k, j] = AA_ht_passed
        #TT_ht_pass_matrix[k, j] = TT_ht_passed
       


        print(f"Subsample {k+1}, dim {dim}: AA={AA_pass:.2f}%  TT={TT_pass:.2f}%")


AA_mean = np.mean(AA_pass_matrix, axis=0)
AA_std  = np.std(AA_pass_matrix, axis=0) / np.sqrt(K)   

TT_mean = np.mean(TT_pass_matrix, axis=0)
TT_std  = np.std(TT_pass_matrix, axis=0) / np.sqrt(K)

#AA_mean_ht = np.mean(AA_ht_pass_matrix, axis=0)
#AA_std_ht  = np.std(AA_ht_pass_matrix, axis=0) / np.sqrt(K)   

#TT_mean_ht = np.mean(TT_ht_pass_matrix, axis=0)
#TT_std_ht  = np.std(TT_ht_pass_matrix, axis=0) / np.sqrt(K)


#print('AA_ht_passed:', AA_mean_ht, AA_std_ht)
#print('TT_ht_passed:', TT_mean_ht, TT_std_ht)

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

out_path = Path("RealData/paper_data/random_data_Shaghayegh_seeds.pdf")
out_path.parent.mkdir(parents=True, exist_ok=True)  # make RealData/paper_data if needed

plt.savefig(out_path)

'''
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
