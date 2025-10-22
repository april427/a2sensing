"""Potential benchmark: Two beamformers v and w with DNN-based design"""
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dense
import os
import scipy.io as sio

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import parse_args
args = parse_args.parse_args()
from channel_functions import *


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, dims, name):
        super(MLPBlock, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        for ii in range(num_layers - 1):
            self.layers.append(Dense(units=dims[ii], activation='relu', name=name + '_relu_' + str(ii)))
            self.layers.append(BatchNormalization())
        self.layers.append(Dense(units=dims[-1], activation='linear', name=name + '_linear'))

    def call(self, inputs, **kwargs):
        x = inputs
        for ii in range(len(self.layers)):
            if ii == 0:
                x = self.layers[ii](inputs)
            else:
                x = self.layers[ii](x)
        return x

drive_save_path = 'BD_2bDNN'
os.makedirs(drive_save_path, exist_ok=True)
#####################################################

'System Information'
N_ris = 16  # Number of BS's antennas

tau = 5 #args.tau
L = 3  # number of path
location_user = None

'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
snr_const = 0  # SNRs
snr_const = args.snr
snr_const = np.array([snr_const])
ref_dis = 5
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2
mean_true_alpha = 0.0 + 0.0j  # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)  # STD of the Gaussian noise per real dim.
#####################################################
'Learning Parameters'
initial_run = 1  # 0: Continue training; 1: Starts from the scratch
n_epochs = 5  # Num of epochs
learning_rate = 0.0001  # Learning rate
batch_per_epoch = 400  # Number of mini batches per epoch
batch_size_train = 1024  # Mini_batch_size = batch_size_order*delta_inv
batch_size_val = 10000  # Scaling the number of tests
model_path = f'{drive_save_path}/params/DNN_trainable_w_v3_L_SNR_tau_N1_N2_' + str((L, snr_const, tau, N_ris))
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
loc_input = tf.placeholder(tf.float32, shape=(None, 2, num_users), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.complex64, shape=(None,  N_ris, N_ris, num_users), name="channel_bs_irs_user")
H_SI_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_SI")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_b")

channel_true, set_location_user_train = generate_irs_user_channel(
                location_user, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
channel_complex2real(channel_true)
batch_size = tf.shape(loc_input)[0]

##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)


with tf.name_scope("channel_sensing"):
    A_T_k1 = channel_bs_irs_user[:,:,:,0] # channel within coherence time
    theta_list =[] # list of complex transmit beamforming
    v_list = [] # list of complex receive beamforming

    'RIS receive beamformer'
    V_real = tf.get_variable('V_real', shape=[1, N_ris, tau])
    V_imag = tf.get_variable('V_imag', shape=[1, N_ris, tau])
    V_tf = tf.complex(V_real, V_imag)
    V_tf = V_tf / tf.norm(V_tf, axis=1, keepdims=True)

    'RIS transmit beamformer'
    W_real = tf.get_variable('W_real', shape=[1, N_ris, tau])
    W_imag = tf.get_variable('W_imag', shape=[1, N_ris, tau])
    W_tf = tf.complex(W_real, W_imag)
    W_tf = W_tf / tf.norm(W_tf, axis=1, keepdims=True)

    y_noiseless1 = A_T_k1 @ W_tf
    noise = tf.complex(tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim), \
                       tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim))
    y_complex1 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless1 + noise
    y_complex2 = tf.transpose(tf.conj(V_tf), perm=[0, 2, 1]) @ y_complex1
    # y_complex_vec = tf.reshape(Y_complex, [-1, tau1 * tau2])
    # y_real = tf.concat([tf.real(y_complex_vec), tf.imag(y_complex_vec)], axis=1)
    y_complex1_flat = tf.reshape(y_complex1, [tf.shape(y_complex1)[0], N_ris*tau])  # (batch_size, N_ris*tau)
    y_complex2_flat = tf.reshape(y_complex2, [tf.shape(y_complex2)[0], 1*tau])      # (batch_size, 1*tau)
        
    y_real = tf.concat([tf.real(y_complex1_flat), tf.imag(y_complex1_flat), \
                        tf.real(y_complex2_flat), tf.imag(y_complex2_flat)], axis=1) 
        

    'caculate bf_gain'
    MLP_bf1 = MLPBlock(3, [1024, 1024, 2 * N_ris], name='mlp_bf1')
    MLP_bf2 = MLPBlock(3, [1024, 1024, 2 * N_ris], name='mlp_bf2')

    v_tmp = MLP_bf1(y_real)
    v_norm = tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1))
    v_tmp = tf.divide(v_tmp, v_norm)
    v_complex = tf.complex(v_tmp[:, 0:N_ris], v_tmp[:, N_ris:2 * N_ris])
    v_complex = tf.reshape(v_complex, [-1, N_ris, 1])

    w_tmp = MLP_bf2(y_real)
    w_norm = tf.reshape(tf.norm(w_tmp, axis=1), (-1, 1))
    w_tmp = tf.divide(w_tmp, w_norm)
    w_complex = tf.complex(w_tmp[:, 0:N_ris], w_tmp[:, N_ris:2 * N_ris])
    w_complex = tf.reshape(w_complex, [-1, N_ris, 1])

    H_SI_tf = H_SI_placeholder
    H_b_hat = H_b_placeholder

    sig_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex), perm=[0,2,1]) @ tf.matmul(H_b_hat, w_complex))**2
    interference_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex), perm=[0,2,1]) @ tf.matmul(H_SI_tf, w_complex))**2
    sinr = tf.reduce_mean(sig_pow / (interference_pow + 2*(noiseSTD_per_dim**2)))
    
    # bf_gain = tf.reduce_mean(tf.abs(tf.transpose(tf.conj(v_complex), perm=[0, 2, 1]) @ G @ w_complex) ** 2)

####################################################################################
####### Loss Function
loss = -tf.log(sinr)
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  Validation Set
channel_true_val, set_location_user_val = generate_irs_user_channel(
    None, location_ris_1, num_samples=batch_size_val, Rician_factor=Rician_factor)
A_T_1_real_val, _ = channel_complex2real(channel_true_val)

# Fix the H_b conversion issue
H_b_val = np.array(channel_true_val[2])
if H_b_val.ndim == 4 and H_b_val.shape[-1] == 1:
    H_b_val = H_b_val.squeeze(axis=-1)
elif H_b_val.ndim == 2:
    # Single matrix, expand to batch
    H_b_val = np.repeat(H_b_val[np.newaxis, :, :], len(set_location_user_val), axis=0)

feed_dict_val = {
    loc_input: np.array(set_location_user_val),
    channel_bs_irs_user: channel_true_val[1],  # Use complex channel directly
    lay['P']: Pvec[0],
    H_SI_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
    H_b_placeholder: H_b_val
}
###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, model_path)
    best_loss = sess.run(loss, feed_dict=feed_dict_val)
    print(-best_loss)
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on
    no_increase = 0
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            channel_true_train, set_location_user_train = generate_irs_user_channel(
                None, location_ris_1, num_samples=128, Rician_factor=Rician_factor)
            A_T_1_real, _ = channel_complex2real(channel_true_train)
            H_SI_batch = channel_true_train[0]  # Self-interference channel
            H_b_batch = np.array(channel_true_train[2])   # Backscattered channel
            
            # Ensure proper shapes for batch processing
            if H_b_batch.ndim == 4 and H_b_batch.shape[-1] == 1:
                H_b_batch = H_b_batch.squeeze(axis=-1)
            elif H_b_batch.ndim == 2:
                # Single matrix, expand to batch
                H_b_batch = np.repeat(H_b_batch[np.newaxis, :, :], len(set_location_user_train), axis=0)
            
            feed_dict_batch = {
                loc_input: np.array(set_location_user_train),
                channel_bs_irs_user: channel_true_train[1],  # Use complex channel directly
                lay['P']: Pvec[0],
                H_SI_placeholder: np.tile(H_SI_batch[np.newaxis, :, :], (len(set_location_user_train), 1, 1)),  # Broadcast to batch
                H_b_placeholder: H_b_batch  # Shape: (batch, N_ris, N_ris)
            }

            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1

        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch', epoch, '  loss_test:%2.5f' % -loss_val, ' dB:%2.3f' % (10 * np.log10(-best_loss)),
               'no_increase:', no_increase)
        if epoch % 10 == 9:  # Every 10 iterations it checks if the validation performace is improved, then saves parameters
            if loss_val < best_loss:
                save_path = saver.save(sess, model_path)
                best_loss = loss_val
                no_increase = 0
            else:
                no_increase = no_increase + 10

    # sio.savemat('./results/DNN_trainable_sensing_wi_feedback_omp_model_tau' + str((tau1, tau2)) + '.mat',
    #             {'bf_gain_dB': (10 * np.log10(-best_loss)),
    #              'bf_gain_opt_dB': (10 * np.log10(opt_loss)),
    #              'N1_N2_tau1_tau2_L': (N1, N2, tau1, tau2, L),
    #              'phi_min_max': (phi_min, phi_max), 'snr[dB]': snr_const})