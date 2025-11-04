"""Bi-Static SIMO that maximizes SINR"""
"""Only one beamformer"""
"""N_ris is the number of receiver antennas"""

# %%
import os
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    os.system('pip install tensorflow[and-cuda]')

try:
    import scipy
except ImportError:
    os.system('pip install scipy')

try:
    import matplotlib
except ImportError:
    os.system('pip install matplotlib')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# Additional M4 optimization
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.linalg import eig
import os
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
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU memory growth enabled for: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
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

drive_save_path = 'Bi_simo_sinr_max'
os.makedirs(drive_save_path, exist_ok=True)

'System Information'
N_bs = 1   # Number of BS's antennas
delta_inv = 32 #number of epoch 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
OS_rate = 20 
delta_inv_OS = OS_rate*delta_inv 
delta_OS = 1/delta_inv_OS 
fc = 2.4e9
Wavelength = 3e8 / fc # Wavelength for 2.4 GHz

BD_modulation = np.array([-1,1] )  # BPSK modulation for BD

'Channel Information'
phi_min = -60*(np.pi/180)
phi_max = 60*(np.pi/180)
num_SNR = 1

# Positions
location_bs = np.array([10, 0, -20])        # BS location -- Tx
location_ris = np.array([0, 0, -20])       # This RIS is Rx 
num_ris = 1

# Channel parameters
noiseSTD_per_dim = np.sqrt(0.5)

# RIS configuration
N_ris = args.N_ris  # Number of RIS elements, treat RIS as BS for the mono-static case
num_users = 1
params_system = (1, N_ris, num_users)
Rician_factor = args.rician_factor  # Rician factor
location_user = None

# Sensing parameters
tau = 10#args.tau  # Pilot length
snr_const = 10#args.snr
snr_const = np.array([snr_const]) 
ref_dis = 5
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2
            # BD at ref_dis has received SNR of snr_const. ||v||^2 = N_ris 

'Learning Parameters'
initial_run = 1   # 0: Continue training; 1: Starts from scratch
n_epochs = 200#args.n_epochs
learning_rate = 1e-3
batch_per_epoch = 128
batch_size_order = 14
val_size_order = 10
scale_factor = 1
test_size = 2000

USE_FFT = False

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Place Holders
# loc_input = tf.placeholder(tf.float32, shape=(None, num_users, 3), name="loc_input")
loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_users), name="loc_input")
channel_g1 = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_bs, num_users), name="channel_g1")
channel_g2 = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_bs, num_users), name="channel_g2")
H_d_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_bs), name="H_d") # direct path channel N_bs x N_ris
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_bs), name="H_b") # backscattered channel N_bs x N_ris
s_placeholder = tf.placeholder(tf.complex64, shape=(None, N_bs), name="ambient_signal") # ambient signal transmitted from the BS, unit power

# %%
channel_true, set_location_user_train = generate_bistatic_channels(
                location_user, location_bs, location_ris, num_samples=1, Rician_factor=Rician_factor, x_BD = BD_modulation)
channel_bistatic_complex2real(channel_true)
##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0) # tf.placeholder(tf.float32, shape=(), name="power") #
    # from0toN = tf.cast(tf.range(0, N_bs, 1), tf.float32)
    # BD pilot sequence: alternating 0 and 1 of length tau
    bd_seq = tf.constant([BD_modulation[int(i % 2)] for i in range(tau)], dtype=tf.float32)  # shape (tau,)


with tf.name_scope("channel_sensing"):
    hidden_size = 256
    RNN1 = RNN(hidden_size, name='RNN_g1')
    RNN2 = RNN(hidden_size, name='RNN_g2')

    MLP_receiver = MLPBlock(3, [hidden_size, hidden_size, 2 * N_ris], name='MLP_receiver')

    snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
    snr_dB = 10* tf.log(snr) / np.log(10)
    snr_normal = snr_dB#(snr_dB - 1) / np.sqrt(1.6666)  # normalization mean=1, std=1.6666
    # Broadcast BD sequence so each time-step entry has shape [batch, 1] for use in the RNN
    x_BD = [tf.tile(tf.reshape(bd_seq[t], [1, 1]), [tf.shape(loc_input)[0], 1]) for t in range(tau)]
    v_list = []
    Y1 = []
    Y2 = []
    z1 = []
    z2 = []

    for t in range(tau): # training pilot length
        if t == 0: # Initialization
            y_real = tf.ones([tf.shape(loc_input)[0], 32])
            z_real_imag = tf.ones([tf.shape(loc_input)[0], 2])
            h_old = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # hidden state
            c_old = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # cell state
            h_old, c_old = RNN1((tf.concat([y_real, z_real_imag, x_BD[t], snr_normal], axis=1), h_old, c_old))
        elif t % 2 == 0:
            h_old, c_old = RNN1((tf.concat([y_real, z_real_imag, x_BD[t], snr_normal], axis=1), h_old, c_old))
        else:
            h_old, c_old = RNN2((tf.concat([y_real, z_real_imag, x_BD[t], snr_normal], axis=1), h_old, c_old))

        # BS beamforming 
        ris_her_unnorm = MLP_receiver(h_old)
        ris_her_r = ris_her_unnorm[:, 0:N_ris]  # real part
        ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris] # imaginary part
        v_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True)) # normalization per sample
        v_real = tf.divide(ris_her_r , v_tmp)
        v_imag = tf.divide(ris_her_i , v_tmp)
        v = tf.concat([v_real, v_imag], axis=1)   
        v_T = tf.reshape(v, [-1, 1, 2 * N_ris])
        v_list.append(v_T[:, 0, :])

                
        # BS next observation
        # uses the backscattered channel RIS-BD-RIS, BD has an alternating pilot sequence
        if t % 2 == 0:
            A_T_k = channel_g1[:, :, 0, 0] # ((batch_size,  N_BS, 2 * N_ris))
        else:
            A_T_k = channel_g2[:, :, 0, 0] # ((batch_size,  N_BS, 2 * N_ris))
        g_real = A_T_k[:, 0:N_ris]
        g_imag = -A_T_k[:,  N_ris:2*N_ris]

        noise = tf.complex(tf.random_normal(tf.shape(g_real), mean=0.0, stddev=noiseSTD_per_dim), 
                           tf.random_normal(tf.shape(g_imag), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.complex(g_real, g_imag) * s_placeholder + noise   # y = gs + n
        z = tf.matmul(tf.complex(v_real[:,tf.newaxis,:], -v_imag[:,tf.newaxis,:]), y_complex[:,:,tf.newaxis])  # v^H y
        z_real_imag = tf.concat([tf.real(z), tf.imag(z)], axis=1) 
        z_real_imag = tf.reshape(z_real_imag[:,:,0], [-1, 2])  # Reshape to match the input size of RNN

        y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1) 
        y_real = tf.reshape(y_real, [-1, 32])  # Reshape to match the input size of RNN

        # store y 
        if t % 2 == 0:
            Y1.append(y_real)
            z1.append(z_real_imag)
        else:
            Y2.append(y_real)
            z2.append(z_real_imag)

    h_old, c_old = RNN2((tf.concat([y_real, z_real_imag, x_BD[t], snr_normal], axis=1), h_old, c_old))
    c_old = Dense(units=200, activation='linear')(c_old)
    
    # Ouput RIS weights
    MLP_bf1 = MLPBlock(3, [hidden_size, hidden_size, 2 * N_ris], name='MLP_bf1')
    ris_her_unnorm = MLP_bf1(c_old)
    ris_her_r = ris_her_unnorm[:, 0:N_ris]  # real part
    ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris] # imaginary part
    v_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True)) # normalization per sample
    v_real = ris_her_r / v_tmp
    v_imag = ris_her_i / v_tmp

    ## output two dimension: angle and distance of the BD
    loc_hat = Dense(units=2, activation='relu')(c_old)  
     
    ###### Loss function -- maximize the ratio of v^H g0 g0^H v / v^H g1 g1^H v  ######
    v_complex = tf.complex(v_real, v_imag)  # (batch, N_ris)
    v_complex = tf.reshape(v_complex, [-1, N_ris, 1])  # (batch, N_ris, 1)

    # known channels
    g1 = tf.complex(channel_g1[:,  0:N_ris, 0, :], channel_g1[:,  N_ris : 2*N_ris, 0, : ])
    g2 = tf.complex(channel_g2[:, 0:N_ris, 0, :], channel_g2[:,  N_ris : 2*N_ris, 0, :]) # ((batch_size,  N_BS, 2 * N_ris))
    
    sigma1_sq = tf.abs(tf.matmul(tf.linalg.adjoint(v_complex), g1))**2  # (batch, 1, 1)
    sigma2_sq = tf.abs(tf.matmul(tf.linalg.adjoint(v_complex), g2))**2  # (batch, 1, 1)
    sigma1_sq = tf.squeeze(sigma1_sq)  # (batch,)
    sigma2_sq = tf.squeeze(sigma2_sq)  # (batch,)

    ratio_1_over_2 = sigma1_sq / (sigma2_sq + 1e-8)
    ratio_2_over_1 = sigma2_sq / (sigma1_sq + 1e-8)
    max_ratio = tf.maximum(ratio_1_over_2, ratio_2_over_1)

    Th = sigma1_sq*sigma2_sq / tf.abs(sigma1_sq - sigma2_sq + 1e-8) * tf.log(sigma1_sq/sigma2_sq + 1e-8) 
    pe = 0.5 + tf.exp(-Th/sigma2_sq)/2 - tf.exp(-Th/sigma1_sq)/2

    # estimated loss functions -- iteration 1 we assume known channels
    z1_stacked = tf.stack(z1, axis=1)  # (batch, tau/2, 2)
    z2_stacked = tf.stack(z2, axis=1)  # (batch, tau/2, 2)

    z1_complex = tf.complex(z1_stacked[:, :, 0], z1_stacked[:, :, 1])  # (batch, tau/2)
    z2_complex = tf.complex(z2_stacked[:, :, 0], z2_stacked[:, :, 1])  # (batch, tau/2)

    sigma1_sq_est = tf.reduce_sum(tf.abs(z1_complex)**2, axis=1)  # (batch,)
    sigma2_sq_est = tf.reduce_sum(tf.abs(z2_complex)**2, axis=1)  # (batch,)

    ratio_1_over_2_est = sigma1_sq_est / (sigma2_sq_est + 1e-8)
    ratio_2_over_1_est = sigma2_sq_est / (sigma1_sq_est + 1e-8)
    max_ratio_est = tf.maximum(ratio_1_over_2_est, ratio_2_over_1_est)  # (batch,)

    Th = sigma1_sq_est*sigma2_sq_est / tf.abs(sigma1_sq_est - sigma2_sq_est + 1e-8) * tf.log(sigma1_sq_est/sigma2_sq_est + 1e-8) 
    pe_est = 0.5 + tf.exp(-Th/sigma2_sq_est)/2 - tf.exp(-Th/sigma1_sq_est)/2


loss = -tf.log(tf.reduce_mean(max_ratio) + 1e-8)  # maximize the average ratio

# xy_pred = loc_hat[:, ]
# xy_true = loc_input[:, 0,]
# loss = tf.reduce_mean(tf.square(xy_pred - xy_true))
user_loss = tf.stack(loss, name='ratio')

# 优化器
global_step = tf.train.get_or_create_global_step()
l2 = 1e-4
reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss_reg = loss + l2 * reg_term
lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=500, decay_rate=0.9)
training_op = tf.train.AdamOptimizer(lr).minimize(loss_reg, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Validation Set
channel_true_val, set_location_user_val = generate_bistatic_channels(
    None, location_bs, location_ris, num_samples=val_size_order*delta_inv, Rician_factor=Rician_factor, x_BD = BD_modulation )
A_T_1_real_val, A_T_2_real_val = channel_bistatic_complex2real(channel_true_val)

#ambient signal
s_signal = (np.random.randn(val_size_order*delta_inv, 1) + 1j * np.random.randn(val_size_order*delta_inv, 1)) / np.sqrt(2)

feed_dict_val = {
    loc_input: np.array(set_location_user_val),
    channel_g1: A_T_1_real_val,
    channel_g2: A_T_2_real_val,
    lay['P']: Pvec[0],
    H_d_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
    H_b_placeholder: channel_true_val[3],
    s_placeholder: s_signal
}
# %%
#################################### Training #################################
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, f'{drive_save_path}/params_RiK10_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}')
    
    # Early stop
    best_val = 1e9
    wait = 0
    PATIENCE = 20
    
    print(tf.test.is_gpu_available())
    
    for epoch in range(n_epochs):
        batch_iter = 0
        epoch_train_losses = []
        s_signal = (np.random.randn(batch_size_order*delta_inv, 1) + 1j * np.random.randn(batch_size_order*delta_inv, 1)) / np.sqrt(2)
        
        for rnd_indices in range(batch_per_epoch):
            # Training set
            channel_true_train, set_location_user_train = generate_bistatic_channels(
                None, location_bs, location_ris, num_samples=batch_size_order*delta_inv, \
                    Rician_factor=Rician_factor, x_BD = BD_modulation) # 16*32 samples
            A_T_1_real, A_T_2_real = channel_bistatic_complex2real(channel_true_train)
            H_d_batch = channel_true_train[0]  # Self-interference channel
            H_b_batch = channel_true_train[3]   # Backscattered channel
            
            feed_dict_batch = {
                loc_input: np.array(set_location_user_train),
                channel_g1: A_T_1_real,
                channel_g2: A_T_2_real,
                lay['P']: Pvec[0],
                H_d_placeholder: np.tile(H_d_batch[np.newaxis, :, :], (len(set_location_user_train), 1, 1)),  # Broadcast to batch
                H_b_placeholder: H_b_batch,  # Shape: (batch, N_ris, N_ris)
                s_placeholder: s_signal
            }

            _, train_loss_val, error_probability = sess.run(
                                    [training_op, loss, pe], feed_dict=feed_dict_batch
            )
            
            epoch_train_losses.append(train_loss_val)
            batch_iter += 1
        
        avg_train_loss = np.mean(epoch_train_losses)
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        
        print('epoch', epoch,
              '  train_loss:%2.7f' % avg_train_loss,
              '  val_loss:%2.7f' % loss_val,
              '  best_val:%2.7f' % best_val)
        
        if epoch % 3 == 0:
            print('error_probability:', np.mean(error_probability))

        # Early Stop
        if loss_val < best_val - 1e-9:
            best_val = loss_val
            wait = 0
            saver.save(sess, f'{drive_save_path}/params_RiK10_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}')
            with open(os.path.join(drive_save_path, "best_val_loss.txt"), "w") as f:
                f.write(str(best_val))
        else:
            wait += 1
            if wait == PATIENCE:
                print("Early stopping at epoch", epoch)
                break

## Validation samples from training set
    # num_test_samples = 10
    # sample_indices = random.sample(range(len(set_location_user_val)), num_test_samples)
    
    # train_losses = []
    # theta_test_list = []
    # location_list = []
    
    # for sample_index in sample_indices:
    #     location_user_target = set_location_user_val[sample_index]
    #     A_T_1_real_test = A_T_1_real_val[sample_index]
        
    #     feed_dict_test = {
    #         loc_input: np.expand_dims(location_user_target, axis=0),
    #         channel_g: np.expand_dims(A_T_1_real_test, axis=0),
    #         lay['P']: Pvec[0],
    #         H_d_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
    #         H_b_placeholder: channel_true_val[3],
    #         s_placeholder: s_signal
    #     }
        
    #     mse_loss, phi_hat_test, theta_test = sess.run([loss, loc_hat, theta_list], feed_dict=feed_dict_test)
    #     train_losses.append(mse_loss)
    #     theta_test = np.array(theta_test)
    #     theta_test_cplx = theta_test[:, :, 0:N_ris] + 1j * theta_test[:, :, N_ris:2 * N_ris]
    #     theta_test_list.append(theta_test_cplx)
    #     location_list.append(location_user_target)
    
########################## TESTING  and Saving ################
    
    ratio_test = []
  
    channel_true_test, set_location_user_test = generate_bistatic_channels(
        None, location_bs, location_ris, num_samples=test_size, Rician_factor=Rician_factor, x_BD = BD_modulation)
    A_T_1_real_test, A_T_2_real_test = channel_bistatic_complex2real(channel_true_test)
    s_signal = (np.random.randn(test_size, 1) + 1j * np.random.randn(test_size, 1)) / np.sqrt(2)
    
    feed_dict_test = {
        loc_input: np.array(set_location_user_test),
        channel_g1: A_T_1_real_test,
        channel_g2: A_T_2_real_test,
        lay['P']: Pvec[0],
        H_d_placeholder: np.tile(channel_true_test[0][np.newaxis, :, :], (len(set_location_user_test), 1, 1)),
        H_b_placeholder: channel_true_test[3],
        s_placeholder: s_signal
    }

    _,  ratio_test = \
            sess.run([loss, max_ratio], 
                            feed_dict=feed_dict_test)

    g1 = tf.complex(channel_g1[:,  0:N_ris, 0, :], channel_g1[:,  N_ris : 2*N_ris, 0, : ])
    g2 = tf.complex(channel_g2[:, 0:N_ris, 0, :], channel_g2[:,  N_ris : 2*N_ris, 0, :]) # ((batch_size,  N_BS, 2 * N_ris))
    A = lay['P'] * tf.matmul(tf.linalg.adjoint(g1), g1) + noiseSTD_per_dim**2 * tf.eye(N_ris, dtype=tf.complex64)
    B = lay['P'] * tf.matmul(tf.linalg.adjoint(g2), g2) + noiseSTD_per_dim**2 * tf.eye(N_ris, dtype=tf.complex64)
    opti_v = eig(A, B)
    

# Save the final results
# model_filename = os.path.join(drive_save_path, f'TEST_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}.mat')
# sio.savemat(model_filename, dict(
#     snr_const = snr_const,
#     N_bs = N_bs, N_ris = N_ris, tau = tau,
#     epoch = n_epochs, 
#     theta_test = theta_test_set,
#     loc_true = test_loc,
#     sinr_test = sinr_test_set,
#     interference_pow = interference_pow_set,
#     sig_pow = sig_pow_set,
#     opti_theta = opti_theta_set,
#     opti_sig_pow = opti_approx_sig_pow_set,
#     opti_int_pow = opti_approx_int_pow_set,
#     opti_approx_sinr = opti_approx_sinr_set,
#     rieman_opti_theta = rieman_opti_theta_set,
#     rieman_opti_sig_pow = rieman_opti_sig_pow_set,
#     rieman_opti_int_pow = rieman_opti_int_pow_set,
#     rieman_opti_sinr = rieman_opti_sinr_set
# ))



# %% 
# Load the saved model and perform TESTING
with tf.Session() as sess:
    # Restore the trained model
    saver.restore(sess, f'{drive_save_path}/params_RiK10_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}') #
    
    # Example: test on new random user locations
    num_test_samples = 2
    test_losses = []
    test_theta_list = []
    test_location_list = []
    
    for _ in range(num_test_samples):
        # Generate a random user location
        test_angle = 1.00988728#np.random.uniform(-np.pi/2, np.pi/2)  # Random angle between -90 and 90 degrees
        test_distance = 5.0  # 5 meters

        # Convert to Cartesian coordinates
        x = test_distance * np.cos(test_angle)
        y = test_distance * np.sin(test_angle)
        z = -20  # Ground level
        
        location_user_test = np.array([[x, y, z]])#generate_location(num_users)
        channel_true_test, set_location_user_test = generate_bistatic_channels(
            location_user_test, location_bs, location_ris, num_samples=1, Rician_factor=Rician_factor, x_BD = BD_modulation)
        A_T_1_real_test, A_T_2_real_test = channel_bistatic_complex2real(channel_true_test)
        print(set_location_user_test)
        H_d = channel_true_test[0]  # Self-interference channel
        s_signal = (np.random.randn(1, 1) + 1j * np.random.randn(1, 1)) / np.sqrt(2)

        feed_dict_test = {
            loc_input: np.array(set_location_user_test),
            channel_g1: A_T_1_real_test,
            channel_g2: A_T_2_real_test,
            lay['P']: Pvec[0],
            H_d_placeholder: H_d[np.newaxis, :, :],  # Add batch dimension
            H_b_placeholder: channel_true_test[3] ,   # Add batch dimension
            s_placeholder: s_signal
        }

        mse_loss, v_complex = sess.run(
                [loss, v_complex], 
                feed_dict=feed_dict_test)
    
        test_losses.append(mse_loss)

        # visualize 
        plot_beam_patterns(
            v_complex.squeeze(), set_location_user_test,
            save_path=None#f'{drive_save_path}/demo_beam_patterns_{i+1}.png'
        )  

# %%
