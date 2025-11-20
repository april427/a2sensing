"""This is to train the AP for monostatic backscatter with alternating x_BD"""
"""Estimates beamformer v=w and angular frequency omega"""

# %%
import os
import sys
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    os.system('pip install tensorflow[and-cuda]')

try:
    import scipy.io as sio
except ImportError:
    os.system('pip install scipy')

try:
    import matplotlib.pyplot as plt
except ImportError:
    os.system('pip install matplotlib')

import numpy as np
import random
tf.disable_v2_behavior() 

# Additional M4 optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
from tensorflow.python.framework import ops
import time
from scipy import io
from scipy import special
from keras.layers import BatchNormalization, Dense
from manifold_optimization import solve_x_equals_y_fast
from parse_args import parse_args
from channel_functions import *

args = parse_args()

if False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(tf.config.list_physical_devices('CPU'))
else:
    print(tf.config.list_physical_devices('GPU'))

# Configure GPU for M4 Mac
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, running on CPU")

#####################################################
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

class RNN(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name):
        super(RNN, self).__init__()
        self.layer_Ui = Dense(units=hidden_size, activation='linear', name='Ui' + name)
        self.layer_Wi = Dense(units=hidden_size, activation='linear', name='Wi' + name)
        self.layer_Uf = Dense(units=hidden_size, activation='linear', name='Uf' + name)
        self.layer_Wf = Dense(units=hidden_size, activation='linear', name='Wf' + name)
        self.layer_Uo = Dense(units=hidden_size, activation='linear', name='Uo' + name)
        self.layer_Wo = Dense(units=hidden_size, activation='linear', name='Wo' + name)
        self.layer_Uc = Dense(units=hidden_size, activation='linear', name='Uc' + name)
        self.layer_Wc = Dense(units=hidden_size, activation='linear', name='Wc' + name)

    def call(self, inputs, **kwargs):
        (input_x, h_old, c_old) = inputs
        i_t = tf.sigmoid(self.layer_Ui(input_x) + self.layer_Wi(h_old))
        f_t = tf.sigmoid(self.layer_Uf(input_x) + self.layer_Wf(h_old))
        o_t = tf.sigmoid(self.layer_Uo(input_x) + self.layer_Wo(h_old))
        c_t = tf.tanh(self.layer_Uc(input_x) + self.layer_Wc(h_old))
        c = i_t * c_t + f_t * c_old
        h_new = o_t * tf.tanh(c)
        return h_new, c

#####################################################

drive_save_path = 'BD_angularFreq'
os.makedirs(drive_save_path, exist_ok=True)

'System Information'
N = 1   # Number of BS's antennas
delta_inv = 32 # number of epoch 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
OS_rate = 20 
delta_inv_OS = OS_rate*delta_inv 
delta_OS = 1/delta_inv_OS 
fc = 10e9
Wavelength = 3e8 / fc # Wavelength for 10 GHz

'Channel Information'
phi_min = -60*(np.pi/180)
phi_max = 60*(np.pi/180)
num_SNR = 1

# Positions
location_ris_1 = np.array([0, 0, -20])       # This RIS is our BS
num_ris = 1

# Channel parameters
mean_true_alpha = 0.0 + 0.0j
std_per_dim_alpha = np.sqrt(0.5) 
noiseSTD_per_dim = np.sqrt(0.5)

# RIS configuration
N_ris = args.N_ris  # Number of RIS elements, treat RIS as BS for the mono-static case
num_users = 1
params_system = (N_ris, N_ris, num_users)
Rician_factor = args.rician_factor  # Rician factor
location_user = None

# Sensing parameters
tau = 2 * N_ris  # Pilot length - 2 times N_ris as specified
snr_const = 10  # args.snr
snr_const = np.array([snr_const]) 
ref_dis = 5
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2
            # BD at ref_dis has received SNR of snr_const. ||v||^2 = N_ris 

# BD modulation - alternating pattern for angular frequency pi
BD_modulation = np.array([(-1)**(t) for t in range(tau)])  # [-1, 1, -1, 1, ...]
print(f"BD modulation pattern: {BD_modulation[:10]}...")  # Print first 10 values

'Learning Parameters'
initial_run = 1   # 0: Continue training; 1: Starts from scratch
n_epochs = 200  # args.n_epochs
learning_rate = 1e-3
batch_per_epoch = 128
batch_size_order = 14
val_size_order = 10
scale_factor = 1
test_size = 2000

USE_FFT = False

# Loss weights
LOS_weight = 1
NLOS_weight = 0

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Place Holders
loc_input = tf.placeholder(tf.float32, shape=(None, 2, num_users), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_ris, num_users), name="channel_bs_irs_user")
H_SI_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_SI")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_b")

# %%
channel_true, set_location_user_train = generate_irs_user_channel(
                location_user, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
channel_complex2real(channel_true)

##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    from0toN = tf.cast(tf.range(0, N, 1), tf.float32)

with tf.name_scope("channel_sensing"):
    hidden_size = 256
    layer_neuron_size = 256
    
    # Initialize RNN and MLP blocks
    RNN_layer = RNN(hidden_size, '_main')
    MLP_beamformer = MLPBlock(3, [layer_neuron_size, layer_neuron_size, 2*N_ris], name='mlp_beamformer')
    MLP_omega = MLPBlock(2, [200, 1], name='mlp_omega')
    
    snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
    snr_dB = 10* tf.log(snr) / np.log(10)
    snr_normal = snr_dB
    
    theta_list = []
    batch_size = tf.shape(loc_input)[0]
    
    # Initialize RNN states
    h_old = tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)
    c_old = tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)
    
    # Carrier frequency components for observation model
    # Time steps: t = 0, 1, 2, ..., tau-1
    time_steps = tf.constant(np.arange(tau, dtype=np.float32), dtype=tf.float32)
    
    for t in range(tau):
        'Initialization'
        if t == 0:
            v2 = tf.ones(shape=[batch_size, N_ris, 1], dtype=tf.complex64) / tf.sqrt(tf.cast(N_ris, tf.complex64))
        else:
            v2 = v_complex
        
        # Get channel based on BD modulation pattern
        # For alternating pattern: t even -> x_BD = -1, t odd -> x_BD = 1
        x_BD_t = BD_modulation[t]
        
        # Select channel based on x_BD_t
        A_T_k = channel_bs_irs_user[:, :, :, 0]
        
        # Extract channel with appropriate x_BD scaling
        # The channel is: x_BD * H_b + H_SI
        # A_T_k is in real format: [real part, -imag part; imag part, real part]
        # Shape: [batch, 2*N_ris, 2*N_ris]
        
        g_real = A_T_k[:, 0:N_ris, 0:N_ris]
        g_imag = -A_T_k[:, 0:N_ris, N_ris:2*N_ris]  # Note the negative sign
        
        x_BD_t_tf = tf.constant(x_BD_t, dtype=tf.complex64)
        
        # Get H_SI from placeholder (constant across all time steps)
        H_SI_batch = H_SI_placeholder
        
        # Separate H_b from the combined channel
        H_b_batch = H_b_placeholder  # This is the backscattered channel
        
        # Apply modulation: x_BD(t) * H_b + H_SI
        g_modulated = x_BD_t_tf * H_b_batch + H_SI_batch
        
        # BS observes the next measurement with carrier
        # y = s(t) * (x_BD(t) * H_b + H_SI) * v + n
        # where s(t) = exp(j * 2 * pi * fc * t)
        
        # Carrier phase at time t (assuming unit time steps)
        # For discrete time with sampling period Ts
        # Assuming one full carrier cycle over tau samples: Ts * fc = 1/tau
        carrier_phase = 2.0 * np.pi * t / tau
        carrier = tf.complex(tf.cos(carrier_phase), tf.sin(carrier_phase))
        
        # Noiseless observation
        y_noiseless = carrier * (g_modulated @ v2)
        
        # Add noise
        noise = tf.complex(tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=noiseSTD_per_dim), 
                          tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless + noise
        
        # Convert to real representation
        y_real = tf.concat([tf.real(y_complex[:, :, 0]), tf.imag(y_complex[:, :, 0])], axis=1)
        
        # RNN input: observation + SNR + time index
        t_feature = tf.constant(t / tau, dtype=tf.float32) * tf.ones(shape=[batch_size, 1], dtype=tf.float32)
        rnn_input = tf.concat([y_real, snr_normal, t_feature], axis=1)
        
        # Update RNN state
        h_old, c_old = RNN_layer((rnn_input, h_old, c_old))
        
        # Output beamformer from RNN state
        ris_her_unnorm = MLP_beamformer(h_old)
        
        ris_her_r = ris_her_unnorm[:, 0:N_ris]
        ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris]
        theta_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True))
        theta_real = ris_her_r / theta_tmp
        theta_imag = ris_her_i / theta_tmp
        
        v_complex = tf.complex(theta_real, theta_imag)
        v_complex = tf.reshape(v_complex, [-1, N_ris, 1])
        theta_list.append(v_complex)
    
    # After tau time steps, output angular frequency estimate and final beamformer
    # Output final beamformer
    ris_her_unnorm = MLP_beamformer(h_old)
    ris_her_r = ris_her_unnorm[:, 0:N_ris]
    ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris]
    theta_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True))
    theta_real = ris_her_r / theta_tmp
    theta_imag = ris_her_i / theta_tmp
    
    # Output angular frequency estimate (omega_hat)
    # The true angular frequency for alternating pattern is pi
    omega_hat = MLP_omega(c_old)  # Estimate omega in [0, 2*pi]
    omega_hat = tf.nn.sigmoid(omega_hat) * 2.0 * np.pi  # Constrain to [0, 2*pi]
     
    ###### Loss function -- SINR maximization
    theta_T_complex = tf.complex(theta_real, theta_imag)
    theta_T_complex = tf.reshape(theta_T_complex, [-1, N_ris, 1])

    H_SI_tf = H_SI_placeholder
    H_b_hat = H_b_placeholder

    batch_size = tf.shape(loc_input)[0]
    def compute_G_for_batch(H_SI_batch):
        s, u, v = tf.linalg.svd(H_SI_batch)
        s_inv = tf.where(s > 1e-8, 1.0 / s, tf.zeros_like(s))
        s_inv_complex = tf.cast(s_inv, tf.complex64)
        H_SI_pinv = tf.matmul(v, tf.matmul(tf.linalg.diag(s_inv_complex), tf.linalg.adjoint(u)))
        middle = tf.matmul(H_SI_batch, H_SI_pinv)
        I = tf.eye(N_ris, dtype=tf.complex64)
        return I - middle

    G = tf.map_fn(compute_G_for_batch, H_SI_tf, fn_output_signature=tf.TensorSpec(shape=(N_ris, N_ris), dtype=tf.complex64))
    G_Hb = tf.matmul(G, H_b_hat)
    G_Hb_theta = tf.matmul(G_Hb, theta_T_complex)
    power = tf.reduce_mean(lay['P'] * tf.abs(G_Hb_theta) ** 2, axis=[1, 2])
    snr_eff = power / (N_ris - 1) / 2 / (noiseSTD_per_dim**2)

    sig_pow = lay['P'] * tf.reduce_sum(tf.abs(tf.matmul(H_b_hat, theta_T_complex))**2, axis=1)
    interference_pow = lay['P'] * tf.reduce_sum(tf.abs(tf.matmul(H_SI_tf, theta_T_complex))**2, axis=1)
    sinr = sig_pow / (interference_pow + 2*(noiseSTD_per_dim**2)*N_ris)

    sp = lay['P'] * tf.abs(tf.linalg.adjoint(theta_T_complex) @ H_b_hat @ theta_T_complex)**2
    intp = lay['P'] * (tf.abs(tf.linalg.adjoint(theta_T_complex) @ H_SI_tf @ theta_T_complex)**2) 
    sinr_tr = sp / (intp + 2*(noiseSTD_per_dim**2))

    # Loss: combination of SINR maximization and angular frequency estimation
    # True angular frequency for alternating pattern is pi
    omega_true = tf.constant(np.pi, dtype=tf.float32)
    omega_loss = tf.reduce_mean(tf.square(omega_hat[:, 0] - omega_true))
    
    sinr_loss = -tf.log(tf.reduce_mean(sinr_tr))
    
    # Combined loss with weighting
    loss = sinr_loss + 0.1 * omega_loss

loss1 = sinr_loss
loss2 = omega_loss
user_loss = tf.stack([loss1, loss2], name='user_loss')

# Optimizer
global_step = tf.train.get_or_create_global_step()
l2 = 1e-4
reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss_reg = loss + l2 * reg_term
lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=500, decay_rate=0.9)
training_op = tf.train.AdamOptimizer(lr).minimize(loss_reg, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Validation Set
channel_true_val, set_location_user_val = generate_irs_user_channel(
    None, location_ris_1, num_samples=val_size_order*delta_inv, Rician_factor=Rician_factor, x_BD=1)
A_T_1_real_val, _ = channel_complex2real(channel_true_val)

# Fix H_b dimension issue
H_b_val = np.array(channel_true_val[2])
if H_b_val.ndim == 4 and H_b_val.shape[-1] == 1:
    H_b_val = H_b_val.squeeze(axis=-1)
elif H_b_val.ndim == 2:
    H_b_val = np.repeat(H_b_val[np.newaxis, :, :], len(set_location_user_val), axis=0)

feed_dict_val = {
    loc_input: np.array(set_location_user_val),
    channel_bs_irs_user: A_T_1_real_val,
    lay['P']: Pvec[0],
    H_SI_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
    H_b_placeholder: H_b_val
}

# %%
######################## Training ###################
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, f'{drive_save_path}/params_angFreq_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}')
    
    # Early stop
    best_val = 1e9
    wait = 0
    PATIENCE = 20
    
    print(tf.test.is_gpu_available())
    
    for epoch in range(n_epochs):
        batch_iter = 0
        epoch_train_losses = []
        epoch_sinr_losses = []
        epoch_omega_losses = []

        for rnd_indices in range(batch_per_epoch):
            # Generate training batch with x_BD = 1 (will be modulated in observation)
            channel_true_train, set_location_user_train = generate_irs_user_channel(
                None, location_ris_1, num_samples=batch_size_order*delta_inv,
                Rician_factor=Rician_factor, x_BD=1)
            
            A_T_1_real, _ = channel_complex2real(channel_true_train)
            
            # Fix H_b dimension
            H_b_train = np.array(channel_true_train[2])
            if H_b_train.ndim == 4 and H_b_train.shape[-1] == 1:
                H_b_train = H_b_train.squeeze(axis=-1)
            elif H_b_train.ndim == 2:
                H_b_train = np.repeat(H_b_train[np.newaxis, :, :], len(set_location_user_train), axis=0)
            
            feed_dict_batch = {
                loc_input: np.array(set_location_user_train),
                channel_bs_irs_user: A_T_1_real,
                lay['P']: Pvec[0],
                H_SI_placeholder: np.tile(channel_true_train[0][np.newaxis, :, :], (len(set_location_user_train), 1, 1)),
                H_b_placeholder: H_b_train
            }
            
            _, loss_train, sinr_loss_train, omega_loss_train = sess.run(
                [training_op, loss, loss1, loss2], feed_dict=feed_dict_batch)
            
            epoch_train_losses.append(loss_train)
            epoch_sinr_losses.append(sinr_loss_train)
            epoch_omega_losses.append(omega_loss_train)
        
        avg_train_loss = np.mean(epoch_train_losses)
        avg_sinr_loss = np.mean(epoch_sinr_losses)
        avg_omega_loss = np.mean(epoch_omega_losses)
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        
        print('epoch', epoch,
              '  train_loss:%2.7f' % avg_train_loss,
              '  sinr_loss:%2.7f' % avg_sinr_loss,
              '  omega_loss:%2.7f' % avg_omega_loss,
              '  val_loss:%2.7f' % loss_val,
              '  best_val:%2.7f' % best_val)
        
        if epoch % 4 == 3:
            omega_est = sess.run(omega_hat, feed_dict=feed_dict_val)
            print(f'    Estimated omega: mean={np.mean(omega_est):.4f}, std={np.std(omega_est):.4f}, true=π={np.pi:.4f}')
        
        if epoch % 5 == 0:
            saver.save(sess, f'{drive_save_path}/params_angFreq_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}')

        # Early Stop
        if loss_val < best_val - 1e-9:
            best_val = loss_val
            wait = 0
            saver.save(sess, f'{drive_save_path}/params_angFreq_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}')
            with open(f'{drive_save_path}/best_val_loss.txt', 'w') as f:
                f.write(str(best_val))
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

    ########################## TESTING and Saving ################
    
    sinr_test_set = []
    interference_pow_set = []
    test_loc = []
    sig_pow_set = []
    theta_test_set = []
    omega_test_set = []
    
    # optimal beamforming
    opti_theta_set = []
    opti_approx_sig_pow_set = []
    opti_approx_int_pow_set = []
    opti_approx_sinr_set = []

    # optimal beamforming using Riemannian optimization
    rieman_opti_theta_set = []
    rieman_opti_sig_pow_set = []
    rieman_opti_int_pow_set = []
    rieman_opti_sinr_set = []

    channel_true_test, set_location_user_test = generate_irs_user_channel(
        None, location_ris_1, num_samples=test_size, Rician_factor=Rician_factor, x_BD=1)
    A_T_1_real_test, _ = channel_complex2real(channel_true_test)
    
    # Fix H_b dimension
    H_b_test = np.array(channel_true_test[2])
    if H_b_test.ndim == 4 and H_b_test.shape[-1] == 1:
        H_b_test = H_b_test.squeeze(axis=-1)
    elif H_b_test.ndim == 2:
        H_b_test = np.repeat(H_b_test[np.newaxis, :, :], len(set_location_user_test), axis=0)
    
    feed_dict_test = {
        loc_input: np.array(set_location_user_test),
        channel_bs_irs_user: A_T_1_real_test,
        lay['P']: Pvec[0],
        H_SI_placeholder: np.tile(channel_true_test[0][np.newaxis, :, :], (len(set_location_user_test), 1, 1)),
        H_b_placeholder: H_b_test
    }

    _, sinr_tr_test, theta_test, sig_pow_test, interference_pow_test, omega_test = \
            sess.run([loss, sinr_tr, theta_list, sp, intp, omega_hat], 
                            feed_dict=feed_dict_test)

    # theta_test is already complex from TensorFlow, just convert to numpy array
    theta_test = np.array(theta_test)
    theta_test_cplx = theta_test  # Already complex
    
    sinr_test_set = sinr_tr_test[:, np.newaxis]
    interference_pow_set = interference_pow_test[:, np.newaxis]
    sig_pow_set = sig_pow_test[:, np.newaxis]
    test_loc = np.expand_dims(set_location_user_test, axis=1)
    theta_test_set = np.expand_dims(theta_test.transpose(1, 0, 2, 3), axis=2)
    omega_test_set = omega_test

    print(f'\nTest Results:')
    print(f'Mean SINR: {np.mean(sinr_tr_test):.4f}')
    print(f'Mean estimated omega: {np.mean(omega_test):.4f}, true omega: π = {np.pi:.4f}')
    print(f'Omega estimation error: {np.mean(np.abs(omega_test - np.pi)):.4f}')

    for j in range(test_size):
        ## Generate optimal beamformer using closed-form solution (if available)
        H_b_test = channel_true_test[2][j]
        H_SI_test = channel_true_test[0]
        c = 2 * noiseSTD_per_dim**2
        theta_star, _ = solve_x_equals_y_fast(np.sqrt(Pvec[0]) * H_b_test, np.sqrt(Pvec[0]) * H_SI_test, c)
        theta_star = theta_star.reshape(-1, 1)  # Ensure proper shape
        sig_pow_opti = Pvec[0] * np.abs(np.conj(theta_star).T @ H_b_test @ theta_star)**2
        int_pow_opti = Pvec[0] * np.abs(np.conj(theta_star).T @ H_SI_test @ theta_star)**2

        opti_theta_set.append(theta_star)
        opti_approx_sig_pow_set.append(sig_pow_opti)
        opti_approx_int_pow_set.append(int_pow_opti)
        opti_approx_sinr_set.append(sig_pow_opti / (int_pow_opti + c * N_ris))

        # Riemannian optimization can be added here if needed
        rieman_opti_theta_set.append(theta_star)
        rieman_opti_sig_pow_set.append(sig_pow_opti)
        rieman_opti_int_pow_set.append(int_pow_opti)
        rieman_opti_sinr_set.append(sig_pow_opti / (int_pow_opti + c * N_ris))

# Save the final results
model_filename = os.path.join(drive_save_path, f'TEST_angFreq_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}.mat')
sio.savemat(model_filename, dict(
    snr_const=snr_const,
    N=N, N_ris=N_ris, tau=tau,
    epoch=n_epochs, 
    theta_test=theta_test_set,
    omega_test=omega_test_set,
    omega_true=np.pi,
    loc_true=test_loc,
    sinr_test=sinr_test_set,
    interference_pow=interference_pow_set,
    sig_pow=sig_pow_set,
    opti_theta=opti_theta_set,
    opti_sig_pow=opti_approx_sig_pow_set,
    opti_int_pow=opti_approx_int_pow_set,
    opti_approx_sinr=opti_approx_sinr_set,
    rieman_opti_theta=rieman_opti_theta_set,
    rieman_opti_sig_pow=rieman_opti_sig_pow_set,
    rieman_opti_int_pow=rieman_opti_int_pow_set,
    rieman_opti_sinr=rieman_opti_sinr_set,
    BD_modulation=BD_modulation
))

print(f'\nResults saved to {model_filename}')

# %%
# Load the saved model and perform TESTING
with tf.Session() as sess:
    # Restore the trained model
    saver.restore(sess, f'{drive_save_path}/params_angFreq_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}')
    
    # Example: test on new random user locations
    num_test_samples = 2
    test_losses = []
    test_theta_list = []
    test_location_list = []
    test_omega_list = []
    
    for _ in range(num_test_samples):
        # Generate a random user location
        test_angle = np.random.uniform(-np.pi/2, np.pi/2)  # Random angle between -90 and 90 degrees
        test_distance = 5.0  # 5 meters

        # Convert to Cartesian coordinates
        x = test_distance * np.cos(test_angle)
        y = test_distance * np.sin(test_angle)
        z = -20  # Ground level
        
        location_user_test = np.array([[x, y, z]])
        channel_true_test, set_location_user_test = generate_irs_user_channel(
            location_user_test, location_ris_1, num_samples=1, Rician_factor=Rician_factor, x_BD=1)
        A_T_1_real_test, _ = channel_complex2real(channel_true_test)
        print(f"\nTest location: {set_location_user_test}")
        
        # Fix H_b dimension
        H_b_test_single = np.array(channel_true_test[2])
        if H_b_test_single.ndim == 4 and H_b_test_single.shape[-1] == 1:
            H_b_test_single = H_b_test_single.squeeze(axis=-1)
        elif H_b_test_single.ndim == 2:
            H_b_test_single = np.repeat(H_b_test_single[np.newaxis, :, :], len(set_location_user_test), axis=0)
        
        H_SI = channel_true_test[0]  # Self-interference channel
        feed_dict_test = {
            loc_input: np.array(set_location_user_test),
            channel_bs_irs_user: A_T_1_real_test,
            lay['P']: Pvec[0],
            H_SI_placeholder: H_SI[np.newaxis, :, :],  # Add batch dimension
            H_b_placeholder: H_b_test_single
        }

        mse_loss, theta_test, omega_test, sinr_test, sig_pow_test, interference_pow_test = sess.run(
                [loss, theta_list, omega_hat, sinr_tr, sp, intp], 
                feed_dict=feed_dict_test)

        test_losses.append(mse_loss)
        theta_test = np.array(theta_test)
        theta_test_cplx = theta_test  # Already complex from TensorFlow
        test_theta_list.append(theta_test_cplx)
        test_location_list.append(location_user_test)
        test_omega_list.append(omega_test[0, 0])

        print(f"  SINR [dB]: {10*np.log10(np.mean(sinr_test)):.3f}")
        print(f"  Signal Power: {np.mean(sig_pow_test):.3f}")
        print(f"  Interference Power: {np.mean(interference_pow_test):.3f}")
        print(f"  Estimated omega: {omega_test[0, 0]:.4f}, True omega: π = {np.pi:.4f}")
        print(f"  Omega estimation error: {np.abs(omega_test[0, 0] - np.pi):.4f}")

        # Visualize beam patterns
        plot_beam_patterns(
            theta_test_cplx.squeeze(), set_location_user_test,
            save_path=None  # Set to a path if you want to save the plot
        )

        ### Optimal coherent beamformer
        H_b_hat = H_b_test_single[0]
        
        if np.all(H_SI) == 0:
            # No self-interference case
            i1 = np.mod(np.arange(N_ris), N_ris)
            theta_opti = 1/np.sqrt(N_ris) * np.exp(1j * np.pi * (i1 * np.sin(set_location_user_test[0][0][0])))
        else:
            # Rayleigh quotient
            H_B = np.matmul(np.conj(H_b_hat).transpose(), H_b_hat)
            H_A = np.matmul(np.conj(H_SI.squeeze()).transpose(), H_SI.squeeze()) + \
                  2*(noiseSTD_per_dim**2)/Pvec[0] * np.eye(N_ris)
            eigv, s, _ = np.linalg.svd(np.matmul(np.linalg.inv(H_A), H_B))
            theta_opti = eigv[:, 0]

        # Optimal beamformer performance
        sinr_opti = Pvec[0] * np.sum(np.abs(H_b_hat @ theta_opti)**2) / \
                   (Pvec[0] * np.sum(np.abs(H_SI.squeeze() @ theta_opti)**2) + 2*(noiseSTD_per_dim**2)*N_ris)
        print(f"\n  Optimal beamformer:")
        print(f"    SINR [dB]: {10*np.log10(sinr_opti):.3f}")
        print(f"    Signal Power: {Pvec[0]*np.sum(np.abs(H_b_hat @ theta_opti)**2):.3f}")
        print(f"    Interference Power: {Pvec[0]*np.sum(np.abs(H_SI.squeeze() @ theta_opti)**2):.3f}")

        # vr = vt formulation
        spopti = Pvec[0] * (np.abs(np.transpose(np.conj(theta_opti)) @ H_b_hat @ theta_opti)**2)
        intpopt = Pvec[0] * (np.abs(np.transpose(np.conj(theta_opti)) @ H_SI.squeeze() @ theta_opti)**2)
        sinr2_opt = spopti / (intpopt + 2*(noiseSTD_per_dim**2))

        print(f"    SINR (vr=vt) [dB]: {10*np.log10(sinr2_opt):.3f}")
        print(f"    Signal Power (vr=vt): {np.mean(spopti):.3f}")
        print(f"    Interference Power (vr=vt): {np.mean(intpopt):.3f}")

        # Visualize optimal beam pattern
        plot_beam_patterns(
            theta_opti[np.newaxis, :], set_location_user_test,
            save_path=None
        )

# %%
