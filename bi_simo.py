"""Bi-Static SIMO"""
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
delta_inv = 32 #number of epoch 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
OS_rate = 20 
delta_inv_OS = OS_rate*delta_inv 
delta_OS = 1/delta_inv_OS 
fc = args.fc
Wavelength = 3e8 / fc # Wavelength for 2.4 GHz

BD_modulation = np.array([-1,1] )  # BPSK modulation for BD

'Channel Information'
phi_min = -60*(np.pi/180)
phi_max = 60*(np.pi/180)
num_SNR = 1

# Positions
location_bs = np.array([Wavelength*100, 0, -20])        # BS location -- Tx
location_ris = np.array([0, 0, -20])       # This RIS is Rx 
num_ris = 1

# Channel parameters
noiseSTD_per_dim = np.sqrt(0.5)

# RIS configuration
N_bs = 1
N_ris = args.N_ris  # Number of RIS elements, treat RIS as BS for the mono-static case
num_users = 1
params_system = (1, N_ris, num_users)
Rician_factor = args.rician_factor  # Rician factor
location_user = None

# Sensing parameters
tau = 2*args.N_ris #args.tau  # Pilot length
K = 1 # Number of samples per BD symbol
snr_const = 25#args.snr
snr_const = np.array([snr_const]) 
ref_dis = 5
Pvec = 10**(snr_const/10) #/ (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2
            # BD at ref_dis has received SNR of snr_const. ||v||^2 = N_ris 

# BD modulation - alternating pattern for angular frequency pi
BD_modulation = np.array([(-1)**(t) for t in range(tau)])  # [-1, 1, -1, 1, ...]
print(f"BD modulation pattern: {BD_modulation[:10]}...")  # Print first 10 values

# Tx signal - OFDM
# 5G NR signal generation parameters
BW = 5e6  # 5 MHz bandwidth
subcarrier_spacing = 15e3  # 15 kHz subcarrier spacing
num_subcarriers = int(BW / subcarrier_spacing)  # 333 subcarriers
num_symbols_ofdm = tau*K  # Number of OFDM symbols equals pilot length

# QPSK modulation constellation
qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

# Generate random QPSK symbols for each subcarrier and OFDM symbol
nr_signal_freq = np.zeros((num_symbols_ofdm, num_subcarriers), dtype=complex)
for t in range(num_symbols_ofdm):
    qpsk_indices = np.random.randint(0, 4, num_subcarriers)
    nr_signal_freq[t, :] = qpsk_constellation[qpsk_indices]

nr_signal_time = np.fft.ifft(nr_signal_freq, axis=1)

# Add cyclic prefix (7.2% of symbol duration, typical for NR)
cp_length = int(0.072 * num_subcarriers)
nr_signal_with_cp = np.zeros((num_symbols_ofdm, num_subcarriers + cp_length), dtype=complex)
for t in range(num_symbols_ofdm):
    nr_signal_with_cp[t, :cp_length] = nr_signal_time[t, -cp_length:]
    nr_signal_with_cp[t, cp_length:] = nr_signal_time[t, :]

tx_signal = nr_signal_with_cp.flatten()

'Learning Parameters'
initial_run = 1   # 0: Continue training; 1: Starts from scratch
n_epochs = 200#args.n_epochs
learning_rate = 1e-3
batch_per_epoch = 128
batch_size_order = 4
val_size_order = 10
scale_factor = 1
test_size = 2000

USE_FFT = False

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()
 
# Feature options
USE_MIXED_Z_FEATURE = True   # Use both fixed-beam and learned-beam power features
USE_LOG_POWER = True         # Log-compress |v^H y|^2 features for numerical stability
# Weights and dropout to quickly ablate contributions
Z_FIXED_W = 1.0              # 1.0 => include fixed-beam power; 0.0 => ignore
Z_LEARN_W = 1.0              # 1.0 => include learned-beam power; 0.0 => ignore
Z_LEARN_DROPOUT = 0.0        # Dropout rate for learned-beam feature (0.0 ~ 0.2)

# Place Holders
# loc_input = tf.placeholder(tf.float32, shape=(None, num_users, 3), name="loc_input")
loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_users), name="loc_input")
channel_g1 = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_bs, num_users), name="channel_g1")
channel_g2 = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_bs, num_users), name="channel_g2")
H_d_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_bs), name="H_d") # direct path channel N_bs x N_ris
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_bs), name="H_b") # backscattered channel N_bs x N_ris
s_placeholder = tf.placeholder(tf.complex64, shape=(None, N_bs, K*num_subcarriers), name="ambient_signal") # ambient signal transmitted from the BS, unit power

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
    hidden_size = 128
    RNN1 = RNN(hidden_size, name='RNN_g1')
    RNN2 = RNN(hidden_size, name='RNN_g2')

    MLP_receiver = MLPBlock(3, [hidden_size*2, hidden_size*2, 2 * N_ris], name='MLP_receiver')

    snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
    snr_dB = 10* tf.log(snr) / np.log(10)
    snr_normal = snr_dB
    # Broadcast BD sequence so each time-step entry has shape [batch, 1] for use in the RNN
    x_BD = [tf.tile(tf.reshape(bd_seq[t], [1, 1]), [tf.shape(loc_input)[0], 1]) for t in range(tau)]
    v_list = []
    Y1 = []
    Y2 = []
    z1 = []
    z2 = []
    
    # FIXED observation beamformer: Equal-gain combining (breaks feedback loop)
    # Network learns from observations WITHOUT depending on its own learned beamformer
    v_obs_real = tf.ones([tf.shape(loc_input)[0], N_ris]) / tf.sqrt(tf.cast(N_ris, tf.float32))
    v_obs_imag = tf.zeros([tf.shape(loc_input)[0], N_ris])

    for t in range(tau): # training pilot length
        if t == 0: # Initialization
            y_real = tf.ones([tf.shape(loc_input)[0], N_ris*2])
            # Initialize z features (two scalars if mixed, else one)
            z_feat = tf.ones([tf.shape(loc_input)[0], 2 if USE_MIXED_Z_FEATURE else 1])
            h_old1 = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # hidden state
            c_old1 = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # cell state
            h_old2 = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # hidden state
            c_old2 = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # cell state
            h_old1, c_old1 = RNN1((tf.concat([y_real, z_feat, x_BD[t], snr_normal], axis=1), h_old1, c_old1))
        elif t % 2 == 0:
            h_old1, c_old1 = RNN1((tf.concat([y_real, z_feat, x_BD[t], snr_normal], axis=1), h_old1, c_old1))
        else:
            h_old2, c_old2 = RNN2((tf.concat([y_real, z_feat, x_BD[t], snr_normal], axis=1), h_old2, c_old2))

        # BS beamforming 
        ris_her_unnorm = MLP_receiver(tf.concat([h_old1, h_old2], axis=1))
        ris_her_r = ris_her_unnorm[:, 0:N_ris]  # real part
        ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris] # imaginary part
        # Safe normalization (avoid division by zero -> NaNs)
        v_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True) + 1e-8)
        v_real = ris_her_r / v_tmp
        v_imag = ris_her_i / v_tmp
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

        noise = tf.complex(tf.random_normal((tf.shape(g_real)[0], K*num_subcarriers), mean=0.0, stddev=noiseSTD_per_dim), 
                           tf.random_normal((tf.shape(g_imag)[1], K*num_subcarriers), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.complex(g_real, g_imag) * s_placeholder + noise   # Y = gs + n (N_ris, K*num_subcarriers)
        
        # Use FIXED observation beamformer (not learned v) to break feedback loop
        z_fixed_c = tf.matmul(
            tf.complex(v_obs_real[:, tf.newaxis, :], -v_obs_imag[:, tf.newaxis, :]),
            y_complex[:, :, tf.newaxis]
        )  # (batch, K*num_subcarriers, 1)
        z_fixed = tf.reduce_mean(tf.abs(z_fixed_c) ** 2, axis=1)
        z_fixed = tf.cast(tf.squeeze(z_fixed, axis=2), tf.float32)  # (batch, 1)

        # Optionally also use learned per-step beam for an auxiliary feature (with stop_gradient to avoid instability)
        if USE_MIXED_Z_FEATURE:
            # Build complex learned beam as column vector and compute v^H y safely
            v_step_col = tf.reshape(tf.complex(v_real, v_imag), [-1, N_ris, 1])  # (batch, N_ris, 1)
            v_step_col = tf.stop_gradient(v_step_col)  # detach gradient to avoid feedback loop
            z_learn_c = tf.matmul(tf.linalg.adjoint(v_step_col), y_complex[:, :, tf.newaxis])  # (batch, K*num_subcarriers, 1)
            z_learn = tf.reduce_mean(tf.abs(z_learn_c) ** 2, axis=1)
            z_learn = tf.cast(tf.squeeze(z_learn, axis=2), tf.float32)  # (batch, 1)

            if USE_LOG_POWER:
                z_fixed_scaled = tf.log1p(z_fixed)
                z_learn_scaled = tf.log1p(z_learn)
            else:
                z_fixed_scaled = z_fixed
                z_learn_scaled = z_learn
            # Optional dropout on learned feature for robustness
            z_learn_scaled = tf.nn.dropout(z_learn_scaled, keep_prob=tf.constant(1.0 - Z_LEARN_DROPOUT, dtype=tf.float32))
            # Weighted concat to enable quick ablations
            z_feat = tf.concat([Z_FIXED_W * z_fixed_scaled, Z_LEARN_W * z_learn_scaled], axis=1)  # (batch, 2)
        else:
            z_feat = tf.log1p(z_fixed) if USE_LOG_POWER else z_fixed     # (batch, 1)

        y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1) 
        y_real = tf.cast(tf.reshape(y_real, [-1, 2 * N_ris]), tf.float32)  # Ensure float32 dtype

        # store y 
        if t % 2 == 0:
            Y1.append(y_real)
            z1.append(z_fixed if not USE_LOG_POWER else tf.log1p(z_fixed))
        else:
            Y2.append(y_real)
            z2.append(z_fixed if not USE_LOG_POWER else tf.log1p(z_fixed))

    h_old2, c_old2 = RNN2((tf.concat([y_real, z_feat, x_BD[t], snr_normal], axis=1), h_old2, c_old2))
    c_old2 = Dense(units=200, activation='linear')(c_old2)
    
    # Ouput RIS weights
    MLP_bf1 = MLPBlock(3, [2*hidden_size, 2*hidden_size, 2 * N_ris], name='MLP_bf1')
    ris_her_unnorm = MLP_bf1(tf.concat([h_old1, h_old2], axis=1))
    ris_her_r = ris_her_unnorm[:, 0:N_ris]  # real part
    ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris] # imaginary part
    v_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True) + 1e-8) # safe normalization
    v_real = ris_her_r / v_tmp
    v_imag = ris_her_i / v_tmp

    ## output two dimension: angle and distance of the BD
    # loc_hat = Dense(units=2, activation='relu')(c_old2)  
     
    ###### Loss function -- maximize the ratio of v^H g0 g0^H v / v^H g1 g1^H v  ######
    v_complex = tf.complex(v_real, v_imag)  # (batch, N_ris)
    v_complex = tf.reshape(v_complex, [-1, N_ris, 1])  # (batch, N_ris, 1)

    ########################## known channels ##########################
    g1 = tf.complex(channel_g1[:,  0:N_ris, 0, :], -channel_g1[:,  N_ris : 2*N_ris, 0, : ])
    g2 = tf.complex(channel_g2[:, 0:N_ris, 0, :], -channel_g2[:,  N_ris : 2*N_ris, 0, :]) # ((batch_size,  N_BS, 2 * N_ris))
    
    sigma1_sq = tf.abs(tf.matmul(tf.linalg.adjoint(v_complex), g1))**2  # (batch, 1, 1)
    sigma2_sq = tf.abs(tf.matmul(tf.linalg.adjoint(v_complex), g2))**2  # (batch, 1, 1)
    sigma1_sq = tf.squeeze(sigma1_sq) * lay['P'] + 1  # (batch,)
    sigma2_sq = tf.squeeze(sigma2_sq) * lay['P'] + 1  # (batch,)

    ratio_1_over_2 = sigma1_sq / (sigma2_sq + 1e-10)
    ratio_2_over_1 = sigma2_sq / (sigma1_sq + 1e-10)
    max_ratio = tf.maximum(ratio_1_over_2, ratio_2_over_1)

    Th = sigma1_sq*sigma2_sq / (sigma1_sq - sigma2_sq + 1e-10) * tf.log(sigma1_sq / sigma2_sq) 
    
    pe = tf.where(
        sigma1_sq > sigma2_sq,
        0.5 + 0.5 * tf.exp(-Th/sigma2_sq) - 0.5 * tf.exp(-Th/sigma1_sq),
        0.5 + 0.5 * tf.exp(-Th/sigma1_sq) - 0.5 * tf.exp(-Th/sigma2_sq)
    )

    #### Simplified receiver: project g2 onto orthogonal complement of g1
    g1_col = tf.reshape(g1, [-1, N_ris, 1])
    g2_col = tf.reshape(g2, [-1, N_ris, 1])
    batch_size = tf.shape(g1_col)[0]

    # projection matrix onto span{g1}: P = g1 g1^H / ||g1||^2
    g1_energy = tf.real(tf.squeeze(tf.matmul(tf.linalg.adjoint(g1_col), g1_col), axis=[1, 2]))  # (batch,)
    g1_energy = tf.maximum(g1_energy, 1e-12)
    proj = tf.matmul(g1_col, tf.linalg.adjoint(g1_col)) / tf.cast(tf.reshape(g1_energy, [-1, 1, 1]), tf.complex64)

    # Identity per-batch
    I_batch = tf.tile(tf.expand_dims(tf.eye(N_ris, dtype=tf.complex64), axis=0), [batch_size, 1, 1])

    # Orthogonal projector (I - P)
    orth_proj = I_batch - proj

    # Apply to g2 and normalize
    v_simp_col = tf.matmul(orth_proj, g2_col)  # (batch, N_ris, 1)
    v_simp_norm_sq = tf.maximum(tf.reduce_sum(tf.abs(v_simp_col)**2, axis=[1,2], keepdims=True), 1e-12)
    v_simp = v_simp_col / tf.cast(tf.sqrt(v_simp_norm_sq), tf.complex64)  # (batch, N_ris, 1) normalized complex beamformer

    # Calculate SINR components using simplified beamformer v_simp
    sigma1_sq_simp = tf.abs(tf.matmul(tf.linalg.adjoint(v_simp), g1))**2  # (batch, 1, 1)
    sigma2_sq_simp = tf.abs(tf.matmul(tf.linalg.adjoint(v_simp), g2))**2  # (batch, 1, 1)
    sigma1_sq_simp = tf.squeeze(sigma1_sq_simp) * lay['P'] + 1  # (batch,)
    sigma2_sq_simp = tf.squeeze(sigma2_sq_simp) * lay['P'] + 1  # (batch,)

    # Threshold for simplified beamformer
    Th_simp_full = sigma1_sq_simp*sigma2_sq_simp / (sigma1_sq_simp - sigma2_sq_simp + 1e-10) * tf.log(sigma1_sq_simp / sigma2_sq_simp) 
    
    # Error probability using simplified beamformer
    pe_simp_full = tf.where(
        sigma1_sq_simp > sigma2_sq_simp,
        0.5 + 0.5 * tf.exp(-Th_simp_full/sigma2_sq_simp) - 0.5 * tf.exp(-Th_simp_full/sigma1_sq_simp),
        0.5 + 0.5 * tf.exp(-Th_simp_full/sigma1_sq_simp) - 0.5 * tf.exp(-Th_simp_full/sigma2_sq_simp)
    )

    # squeeze to (batch,)
    v_simp_norm_sq = tf.squeeze(v_simp_norm_sq, axis=[1,2])     

    # Threshold: (1 + P * v_simp_norm^2) / (P * v_simp_norm^2) * log(1 + P * v_simp_norm^2)
    Th_simp = (1.0 + lay['P'] * v_simp_norm_sq) / (lay['P'] * v_simp_norm_sq + 1e-12) \
                * tf.log(1.0 + lay['P'] * v_simp_norm_sq)
    pe_simp = tf.exp(-Th_simp / (1.0)) / 2 + 0.5 -  tf.exp(-Th_simp / (1.0 + lay['P'] * v_simp_norm_sq)) / 2

    #### Optimum receiver
    g2_energy = tf.real(tf.squeeze(tf.matmul(tf.linalg.adjoint(g2_col), g2_col), axis=[1, 2]))  # (batch,)
    g2_energy = tf.maximum(g2_energy, 1e-12)
    
    # Coefficients for optimal detector (batch,) -> reshape for broadcasting
    c1 = lay['P'] * g1_energy / (lay['P'] * g1_energy + 1)  # (batch,)
    c2 = lay['P'] * g2_energy / (lay['P'] * g2_energy + 1)  # (batch,)
    c1_batch = tf.cast(tf.reshape(c1, [-1, 1, 1]), tf.complex64)  # (batch, 1, 1)
    c2_batch = tf.cast(tf.reshape(c2, [-1, 1, 1]), tf.complex64)  # (batch, 1, 1)
    
    # Beamforming matrix: Mg = c1 * g1*g1^H - c2 * g2*g2^H
    Mg = c1_batch * tf.matmul(g1_col, tf.linalg.adjoint(g1_col)) \
         - c2_batch * tf.matmul(g2_col, tf.linalg.adjoint(g2_col))  # (batch, N_ris, N_ris)
    
    # Covariance matrices under H0 and H1
    S1 = tf.cast(lay['P'], tf.complex64) * tf.matmul(g1_col, tf.linalg.adjoint(g1_col)) + I_batch  # (batch, N_ris, N_ris)
    S2 = tf.cast(lay['P'], tf.complex64) * tf.matmul(g2_col, tf.linalg.adjoint(g2_col)) + I_batch  # (batch, N_ris, N_ris)
    
    # Cholesky decomposition: S = L * L^H (lower triangular)
    L1 = tf.linalg.cholesky(S1)  # (batch, N_ris, N_ris)
    L2 = tf.linalg.cholesky(S2)  # (batch, N_ris, N_ris)
    
    # Compute L1 * Mg * L1^H (this is Hermitian)
    Mg_transformed1 = tf.matmul(tf.matmul(L1, Mg), tf.linalg.adjoint(L1))  # (batch, N_ris, N_ris)
    eigvals1, _ = tf.linalg.eigh(Mg_transformed1)  # (batch, N_ris) ascending order
    
    # Compute L2 * Mg * L2^H (this is Hermitian)
    Mg_transformed2 = tf.matmul(tf.matmul(L2, Mg), tf.linalg.adjoint(L2))  # (batch, N_ris, N_ris)
    eigvals2, _ = tf.linalg.eigh(Mg_transformed2)  # (batch, N_ris) ascending order
    
    # Optimal threshold (log-likelihood ratio threshold at equal priors)
    Th_opt = tf.log(lay['P'] * g1_energy + 1) - tf.log(lay['P'] * g2_energy + 1)  # (batch,) float32
    
    # Error probability using Chernoff bound with largest eigenvalues
    # Extract eigenvalues and ensure they are real (eigh returns real eigenvalues but as float32)
    lambda1_max = tf.real(eigvals1[:, -1])  # (batch,) largest eigenvalue under H1
    lambda1_2nd = tf.real(eigvals1[:, 0])  # (batch,) 2nd largest eigenvalue under H1
    lambda2_max = tf.real(eigvals2[:, 0])  # (batch,) largest eigenvalue under H2
    lambda2_2nd = tf.real(eigvals2[:, -1])  # (batch,) 2nd largest eigenvalue under H2
    
    # Compute error probability components with numerical stability
    pe_opt = tf.where(
        Th_opt < 0,
        0.5 - 0.5 * (lambda1_max / (lambda1_2nd - lambda1_max + 1e-10)) * tf.exp(-Th_opt / (lambda1_max + 1e-10)) \
            + 0.5 * (lambda2_max / (lambda2_2nd - lambda2_max + 1e-10)) * tf.exp(-Th_opt / (lambda2_max + 1e-10)),
        0.5 - 0.5 * (lambda1_2nd / (lambda1_2nd - lambda1_max + 1e-10)) * tf.exp(-Th_opt / (lambda1_2nd + 1e-10)) \
            + 0.5 * (lambda2_2nd / (lambda2_2nd - lambda2_max + 1e-10)) * tf.exp(-Th_opt / (lambda2_2nd + 1e-10))
    )



log_max_ratio = tf.log(tf.maximum(max_ratio, 1.0 + 1e-9))  # ensure argument >=1+eps
loss = -tf.reduce_mean(log_max_ratio)  # maximize average ratio in stable way
# loss = tf.cast(tf.reduce_mean(tf.linalg.norm(v_complex - v_simp, axis=1)**2), dtype=tf.float32)  # minimize the difference between learned and simplified beamformer
# xy_pred = loc_hat[:, ]
# xy_true = loc_input[:, 0,]
# loss = tf.reduce_mean(tf.square(xy_pred - xy_true))
user_loss = tf.stack(loss, name='ratio')

# optimizer
global_step = tf.train.get_or_create_global_step()
l2 = 1e-4
reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss_reg = loss + l2 * reg_term
lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=500, decay_rate=0.9)
optimizer = tf.train.AdamOptimizer(lr)
grads_vars = optimizer.compute_gradients(loss_reg)
safe_grads = []
vars_list = []
for g, v in grads_vars:
    if g is None:
        safe_grads.append(None)
        vars_list.append(v)
    else:
        g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))  # replace non-finite grads
        safe_grads.append(g)
        vars_list.append(v)
clipped_grads, global_norm = tf.clip_by_global_norm([g for g in safe_grads if g is not None], 5.0)
# Reconstruct list with clipped grads
final_grads = []
clip_index = 0
for g in safe_grads:
    if g is None:
        final_grads.append(None)
    else:
        final_grads.append(clipped_grads[clip_index])
        clip_index += 1
nan_checks = [
    tf.check_numerics(tf.real(v_complex), 'v_complex real NaN'),
    tf.check_numerics(tf.imag(v_complex), 'v_complex imag NaN'),
    tf.check_numerics(log_max_ratio, 'log_max_ratio NaN')
]
with tf.control_dependencies(nan_checks):
    training_op = optimizer.apply_gradients(list(zip(final_grads, vars_list)), global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Validation Set
channel_true_val, set_location_user_val = generate_bistatic_channels(
    None, location_bs, location_ris, num_samples=val_size_order*delta_inv, Rician_factor=Rician_factor, x_BD = BD_modulation )
A_T_1_real_val, A_T_2_real_val = channel_bistatic_complex2real(channel_true_val)

# Ambient signal: QPSK modulation (constant amplitude, random phase)
# This provides cleaner observations than Gaussian (variable amplitude)
qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)  # Normalized QPSK
s_signal = qpsk_symbols[np.random.randint(0, 4, size=(val_size_order*delta_inv, 1))]

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
        # QPSK ambient signal (constant amplitude, random phase)
        qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        s_signal = qpsk_symbols[np.random.randint(0, 4, size=(batch_size_order*delta_inv, 1))]
        
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

            _, train_loss_val,  error_probability, max_ratio_val = sess.run(
                                    [training_op, loss, pe, max_ratio], feed_dict=feed_dict_batch
            )

            threshold, error_probability, ch_g1, ch_g2, vec_v, pe_simplified, pe_optimum = sess.run(
                                    [Th, pe, g1, g2, v_complex, pe_simp_full, pe_opt], feed_dict=feed_dict_batch
            )

            eigvalue1, eigvalue2 = sess.run( [eigvals1, eigvals2], feed_dict=feed_dict_batch)
            
            epoch_train_losses.append(train_loss_val)
            batch_iter += 1
        
        avg_train_loss = np.mean(epoch_train_losses)
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        
        print('epoch', epoch,
              '  train_loss:%2.7f' % avg_train_loss,
              '  val_loss:%2.7f' % loss_val,
              '  best_val:%2.7f' % best_val)
        print('threshold:', np.mean(threshold))
        print('max_ratio:', np.mean(max_ratio_val))
        print('fir 5 max_ratio:', max_ratio_val[:5])
        print('sigma1^2:', Pvec[0] * np.mean(np.abs(np.matmul(np.conj(vec_v).transpose(0, 2, 1), ch_g1)**2), axis = 0) + 1 )
        print('sigma2^2:', Pvec[0] * np.mean(np.abs(np.matmul(np.conj(vec_v).transpose(0, 2, 1), ch_g2)**2), axis = 0) + 1 ) 

        if epoch % 1 == 0:
            print('pe_activesensing:', np.mean(error_probability))
            print('pe_simplified:', np.mean(pe_simplified))
            print('pe_optimum:', np.mean(pe_optimum))

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
    # QPSK ambient signal for testing
    qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    s_signal = qpsk_symbols[np.random.randint(0, 4, size=(test_size, 1))]
    
    feed_dict_test = {
        loc_input: np.array(set_location_user_test),
        channel_g1: A_T_1_real_test,
        channel_g2: A_T_2_real_test,
        lay['P']: Pvec[0],
        H_d_placeholder: np.tile(channel_true_test[0][np.newaxis, :, :], (len(set_location_user_test), 1, 1)),
        H_b_placeholder: channel_true_test[3],
        s_placeholder: s_signal
    }

    _,  ratio_test = sess.run([loss, max_ratio], feed_dict=feed_dict_test)

    # Build complex channels (consistent sign) and generalized-eigen beamformer in TF
    P_c = tf.cast(lay['P'], tf.complex64)
    N0_c = tf.cast(noiseSTD_per_dim**2, tf.complex64)

    g1_c = tf.complex(channel_g1[:, 0:N_ris, 0, :], -channel_g1[:, N_ris:2*N_ris, 0, :])  # (B, N_ris, 1)
    g2_c = tf.complex(channel_g2[:, 0:N_ris, 0, :], -channel_g2[:, N_ris:2*N_ris, 0, :])  # (B, N_ris, 1)

    g1_col = g1_c
    g2_col = g2_c
    Bsz = tf.shape(g1_col)[0]
    I_batch_c = tf.tile(tf.expand_dims(tf.eye(N_ris, dtype=tf.complex64), axis=0), [Bsz, 1, 1])

    # Use g g^H (N_ris x N_ris), not g^H g (1 x 1)
    A_tf = P_c * tf.matmul(g1_col, tf.linalg.adjoint(g1_col)) + N0_c * I_batch_c
    B_tf = P_c * tf.matmul(g2_col, tf.linalg.adjoint(g2_col)) + N0_c * I_batch_c

    # Solve generalized Hermitian EVP: A v = λ B v via whitening
    L = tf.linalg.cholesky(B_tf)                             # B = L L^H
    Y = tf.linalg.triangular_solve(L, A_tf, lower=True)      # Y = L^{-1} A
    Tm = tf.linalg.triangular_solve(L, Y, lower=True, adjoint=True)  # T = L^{-H} L^{-1} A = L^{-H} A L^{-1}
    eigvals_opt, eigvecs_opt = tf.linalg.eigh(Tm)            # batch-eigh
    u_max = eigvecs_opt[:, :, -1:]                           # (B, N_ris, 1)
    v_opt = tf.linalg.triangular_solve(L, u_max, lower=True, adjoint=True)  # v = L^{-H} u
    v_opt = v_opt / tf.sqrt(tf.maximum(tf.reduce_sum(tf.abs(v_opt)**2, axis=[1,2], keepdims=True), 1e-12))

    # Evaluate v_opt for your feed
    v_opt_np = sess.run(v_opt, feed_dict=feed_dict_test)
    # Optionally, compare its pe:
    sigma1_sq_opt = np.squeeze(np.abs(np.matmul(np.conj(v_opt_np).transpose(0,2,1),
                                                sess.run(g1_col, feed_dict=feed_dict_test)))**2) * Pvec[0] + 1
    sigma2_sq_opt = np.squeeze(np.abs(np.matmul(np.conj(v_opt_np).transpose(0,2,1),
                                                sess.run(g2_col, feed_dict=feed_dict_test)))**2) * Pvec[0] + 1
    Th_opt_num = sigma1_sq_opt*sigma2_sq_opt / (sigma1_sq_opt - sigma2_sq_opt + 1e-10) * np.log(sigma1_sq_opt / sigma2_sq_opt)
    pe_opt_v = np.where(
        sigma1_sq_opt > sigma2_sq_opt,
        0.5 + 0.5 * np.exp(-Th_opt_num/sigma2_sq_opt) - 0.5 * np.exp(-Th_opt_num/sigma1_sq_opt),
        0.5 + 0.5 * np.exp(-Th_opt_num/sigma1_sq_opt) - 0.5 * np.exp(-Th_opt_num/sigma2_sq_opt)
    )
    print('pe_opt_v (gen-eig):', np.mean(pe_opt_v))
    

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
        # QPSK ambient signal
        qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        s_signal = qpsk_symbols[np.random.randint(0, 4, size=(1, 1))]

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
