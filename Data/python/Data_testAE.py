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


mc_bkg_jets, mc_bkg_ht = process_h5_file0("new_Data/data_Run_2016_283876.h5")
X1 = mc_bkg_jets[::100]
Npv1 = mc_bkg_jets[::100,0,3]
Jets1 = mc_bkg_jets[::100, :, :-1]
HT1 = mc_bkg_ht[::100]



print(Jets1.shape, Npv1.shape)

#Ht1_all = mc_bkg_ht[::100]
N1 = len(X1)

#X2 = np.load('Data/Jet2_bkg_h5.npy',allow_pickle=True)
#N2 = len(X2)


mc_AA_jets, mc_AA_ht = process_h5_file0("new_Data/HToAATo4B.h5")
X_AA = mc_AA_jets
NpvAA = mc_AA_jets[::,0,3]
JetsAA = mc_AA_jets[::,:, :-1]
HTAA = mc_AA_ht[::100]

print('X_AA.shape',X_AA.shape)
N_tt = len(X_AA)
print(JetsAA.shape, NpvAA.shape)


mc_tt_jets, mc_tt_ht = process_h5_file0("new_Data/ttbar_2016_1.h5")
X_tt = mc_tt_jets
Npvtt = mc_tt_jets[::,0,3]
Jetstt = mc_tt_jets[::,:, :-1]
HTtt = mc_tt_ht[::100]

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
#Dim = [1, 4]


results_npv = {}





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

    #autoencoder.save(f'autoencoder_model0_{dim}.keras')

#print('Npv:',Npv1_test.shape,results_npv[0]["bkg_scores"].shape)


percen_9975_ht = np.percentile(HT_test, 99.75)

# Compute percentage of ttbar passing this threshold (non serve per il meeting ma lasciali)
AA_ht_passed = 100 * np.sum(HTAA > percen_9975_ht) / len(HTAA)
TT_ht_passed = 100 * np.sum(HTtt > percen_9975_ht) / len(HTtt)

print('AA_ht_passed:', AA_ht_passed)
print('TT_ht_passed:', TT_ht_passed)

# --- Meeting mode: forza la curva e le linee ---
AA_pass_rates = [results_dim[index]["AA_pass"] for index in range(len(Dim))]
TT_pass_rates = [results_dim[index]["TT_pass"] for index in range(len(Dim))]  

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(Dim, AA_pass_rates, marker='o', linestyle='-', color='tab:green',
        label="HToAAto4B Pass Rate")
ax.plot(Dim, TT_pass_rates, marker='o', linestyle='-', color='goldenrod',
        label="TTBar Pass Rate")

# Finta linea “HT efficiency”
ax.hlines(21.3, 1, 15, color="grey", linestyles="dashed",
          label="HToAATo4B, HT Efficiency:21.33%")
ax.hlines(97.3, 1, 15, color="grey", linestyles="dashed",
          label="TTBar, HT Efficiency:97.26%")
ax.plot([], [], ' ', label="Threshold = 99.75 Percentile of Test Background")

ax.set_xlabel("Latent Dimension", loc='center')
ax.set_ylabel("Signal Efficiency (%)", loc='center')
ax.grid(True)
ax.set_xlim(-0.05, 16)
ax.set_ylim(0, 200)
ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])

ax.legend(fontsize=14, frameon=True, loc='upper right',
          bbox_transform=ax.transData)


fig.tight_layout(rect=[0, 0, 1, 0.93])
add_cms_header(fig)
fig.savefig("paper/signal_pass_vs_dimension_data.pdf")
fig.savefig("paper/signal_pass_vs_dimension_data.png")

plt.close(fig)


