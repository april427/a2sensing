"""
Bi-Static MIMO with BD SINR Maximization Objective

There exist other static scatters. 
BD alters its states to distinguish itself from other reflectors.

Objective: max_v SINR_BD = P|w^H H_b v|^2 / (P|w^H (H_d+H_r) v|^2 + noise_var)

"""

import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    os.system('pip install tensorflow[and-cuda]')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    os.system('pip install matplotlib')
try:
    import scipy.io as sio
except ImportError:
    os.system('pip install scipy')
import scipy.io as sio
from scipy.linalg import eig
from keras.layers import BatchNormalization, Dense
from manifold_optimization import solve_with_random_restarts
from parse_args import parse_args
from channel_functions import *

args = parse_args()

# GPU configuration
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(tf.config.list_physical_devices('CPU'))
else:
    print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU memory growth enabled for: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, running on CPU")


#####################################################
# Network Components
#####################################################

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, dims, name):
        super(MLPBlock, self).__init__()
        self.layers_list = []
        self.num_layers = num_layers
        for ii in range(num_layers - 1):
            self.layers_list.append(Dense(units=dims[ii], activation='relu', name=name + '_relu_' + str(ii)))
            self.layers_list.append(BatchNormalization())
        self.layers_list.append(Dense(units=dims[-1], activation='linear', name=name + '_linear'))

    def call(self, inputs, **kwargs):
        x = inputs
        for ii in range(len(self.layers_list)):
            x = self.layers_list[ii](x)
        return x


class LSTM_Cell(tf.keras.layers.Layer):
    """LSTM cell for sequential observation processing"""
    def __init__(self, hidden_size, name):
        super(LSTM_Cell, self).__init__()
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
# System Configuration
#####################################################

drive_save_path = 'Bi_mimo_sinr'
os.makedirs(drive_save_path, exist_ok=True)

# System parameters
fc = args.fc
Wavelength = 3e8 / fc


# Channel parameters
noiseSTD_per_dim = np.sqrt(0.5)
noise_var = 2 * noiseSTD_per_dim**2  # Total noise variance

# Tx/Rx configuration
    # Positions
d0 = 80*Wavelength  # Distance between Tx and Rx
location_tx = np.array([0, 0, 0])                     # Tx center location 
location_rx = np.array([0, 0, d0])                    # Rx center location
    # Antenna array, square array 
N_tx = 4 
N_rx = args.N_ris
N_ris = N_rx  # Alias for backward compatibility
N_tx_h_ch = int(np.sqrt(N_tx))
N_tx_v_ch = int(np.sqrt(N_tx))
N_rx_h_ch = int(np.sqrt(N_rx))
N_rx_v_ch = int(np.sqrt(N_rx))
num_users = 1
num_scatters = args.N_scatterers  # excluding BD
Rician_factor = args.rician_factor
location_bd = None

# Sensing parameters
tau = 32  # Pilot length (also number of BD interactions)
K = 1  # Number of OFDM symbols per BD state
snr_const = 25
snr_const = np.array([snr_const])
ref_dis = 5
Pvec = 10 ** (snr_const / 10)

# BD modulation - alternating pattern
BD_modulation = np.array([(-1) ** t for t in range(tau)])
print(f"BD modulation pattern: {BD_modulation[:10]}...")

#####################################################
# Generate ambient signal
#####################################################
BW = 5e6  # 5 MHz bandwidth
subcarrier_spacing = 15e3  # 15 kHz subcarrier spacing
num_subcarriers = int(BW / subcarrier_spacing)  # 333 subcarriers (samples)
num_ofdm_symbols = tau*K  # Number of OFDM symbols within pilot

# QPSK modulation constellation
qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

# Generate random QPSK symbols for each subcarrier and OFDM symbol
nr_signal_freq = np.zeros((num_ofdm_symbols, num_subcarriers), dtype=complex)
for t in range(num_ofdm_symbols):
    qpsk_indices = np.random.randint(0, 4, num_subcarriers)
    nr_signal_freq[t, :] = qpsk_constellation[qpsk_indices]

nr_signal_time = np.fft.ifft(nr_signal_freq, axis=1)

# Add cyclic prefix (7.2% of symbol duration, typical for NR)
cp_length = int(0.072 * num_subcarriers)
nr_signal_with_cp = np.zeros((num_ofdm_symbols, num_subcarriers + cp_length), dtype=complex)
for t in range(num_ofdm_symbols):
    nr_signal_with_cp[t, :cp_length] = nr_signal_time[t, -cp_length:]
    nr_signal_with_cp[t, cp_length:] = nr_signal_time[t, :]

tx_signal = nr_signal_with_cp.flatten()

####  Learning parameters
initial_run = 1
n_epochs = 50
learning_rate = 5e-4
batch_per_epoch = 128
batch_size_order = 4
val_size_order = 10
test_size = 200

#####################################################
# Build Computation Graph
#####################################################

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Placeholders
loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_users), name="loc_input")
scatter_loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_scatters), name="scatter_loc_input")
H_d_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_d")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_b")
H_r_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_r")
H_d_rev_placeholder = tf.placeholder(tf.complex64, shape=(None, N_tx, N_ris), name="H_d_rev")
H_b_rev_placeholder = tf.placeholder(tf.complex64, shape=(None, N_tx, N_ris), name="H_b_rev")
H_r_rev_placeholder = tf.placeholder(tf.complex64, shape=(None, N_tx, N_ris), name="H_r_rev")
s_placeholder = tf.placeholder(tf.complex64, shape=(None, 1, num_subcarriers), name="ambient_signal")

##################### ACTIVE SENSING NETWORK #####################

with tf.name_scope("system_parameters"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    bd_seq = tf.constant(BD_modulation.astype(np.float32), dtype=tf.float32)

with tf.name_scope("active_sensing_agent"):
    hidden_size1 = 256
    hidden_size2 = 512
    
    # Two LSTM cells for alternating BD states
    LSTM1 = LSTM_Cell(hidden_size1, name='LSTM_1')  
    LSTM2 = LSTM_Cell(hidden_size2, name='LSTM_2') 
    mlp_ris_tx = MLPBlock(3, [hidden_size1 * 2, hidden_size1 * 2, 2 * N_tx], name='RIS_transmitter')
    mlp_ris_rx = MLPBlock(3, [hidden_size1 * 2, hidden_size2 * 2, 2 * N_tx], name='RIS_receiver')
    mlp_re_tx = MLPBlock(3, [hidden_size2 * 2, hidden_size2 * 2, 2 * N_rx], name='Receiver_transmitter')
    mlp_re_rx = MLPBlock(3, [hidden_size2 * 2, hidden_size2 * 2, 2 * N_rx], name='Receiver_receiver')
    
    # SNR feature
    snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
    snr_dB = 10 * tf.log(snr) / np.log(10)
    snr_normal = snr_dB
    
    # BD modulation sequence for each time step
    x_BD = [tf.tile(tf.reshape(bd_seq[t], [1, 1]), [tf.shape(loc_input)[0], 1]) for t in range(tau)]
    
    # Storage
    v_list = []
    Y1_list = []
    Y2_list = []
    z1_list = []
    z2_list = []
    
    #####################################################
    # Active Sensing Loop (Ping-Pong Style):
    # User 1 (Tx) transmits with w, User 2 (Rx) receives with v
    # Channel: H = x_BD[t] * H_b + H_d + H_r
    # Received signal: y = sqrt(P) * v^H * H * w * s + noise
    #####################################################
    
    batch_size = tf.shape(loc_input)[0]
    
    for t in range(tau):
        'Initialization at t=0'
        if t == 0:
            # Initialize LSTM states
            h_old1 = tf.zeros([batch_size, hidden_size1])  # RIS state
            c_old1 = tf.zeros([batch_size, hidden_size1])
            h_old2 = tf.zeros([batch_size, hidden_size2])  # Rx state
            c_old2 = tf.zeros([batch_size, hidden_size2])
            
            # Initialize first RIS transmit beamformer w 
            w_init_real = tf.get_variable("w_init_real", shape=(1, N_tx, 1), trainable=True)
            w_init_imag = tf.get_variable("w_init_imag", shape=(1, N_tx, 1), trainable=True)
            w_complex_init = tf.complex(w_init_real, w_init_imag)
            w1 = w_complex_init / tf.cast(tf.norm(w_complex_init, axis=1, keepdims=True), tf.complex64)
            
            # Initialize first Rx receive beamformer v
            v_init_real = tf.get_variable("v_init_real", shape=(1, N_rx, 1), trainable=True)
            v_init_imag = tf.get_variable("v_init_imag", shape=(1, N_rx, 1), trainable=True)
            v_complex_init = tf.complex(v_init_real, v_init_imag)
            v1 = v_complex_init / tf.cast(tf.norm(v_complex_init, axis=1, keepdims=True), tf.complex64)

            w2 = w1  # Initialize RIS receive beamformer
            
            # Store for random initialization comparison
            bf_gain_init_w = w1
            bf_gain_init_v = v1
        
        'Construct effective channel H(t) = x_BD[t] * H_b + H_d + H_r'
        # H_d, H_r: (batch, N_rx, N_tx), H_b: (batch, N_rx, N_tx)
        # x_BD[t] is scalar (+1 or -1)
        x_bd_t = tf.reshape(tf.cast(x_BD[t], tf.complex64), [-1, 1, 1])  # (batch, 1, 1)
        H_eff = x_bd_t * H_b_placeholder + H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
        H_eff_rev = x_bd_t * H_b_rev_placeholder + H_d_rev_placeholder + H_r_rev_placeholder  # (batch, N_tx, N_rx)
        
        'Rx observes K samples: y = sqrt(P) * v^H * H * w * s + noise'
        y_noiseless_before_rx = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.matmul(H_eff, w1)  # (batch, N_rx, 1)
        y_noiseless_before_rx = tf.tile(y_noiseless_before_rx, [1, 1, K])  # (batch, N_rx, K)
        noise_rx = tf.complex(
            tf.random_normal([batch_size, N_rx, K], mean=0.0, stddev=noiseSTD_per_dim),
            tf.random_normal([batch_size, N_rx, K], mean=0.0, stddev=noiseSTD_per_dim)
        )
        y_complex_nobf = y_noiseless_before_rx + noise_rx
        y_complex_rx_bf = tf.matmul(tf.linalg.adjoint(v1), y_complex_nobf)  # (batch, 1, K)
        y_complex_rx_bf = tf.squeeze(y_complex_rx_bf, axis=1)  # (batch, K)
        
        # Stack and compute sufficient statistics (mean and power)
        y_mean = tf.reduce_mean(y_complex_rx_bf, axis=1, keepdims=True)  # (batch, 1)
        y_power = tf.reduce_mean(tf.abs(y_complex_rx_bf) ** 2, axis=1, keepdims=True)  # (batch, 1)
        y_samples_flat = tf.reduce_mean(y_complex_nobf, axis=2)  # (batch, N_rx)

        # Feature for Rx LSTM: [Re(y_mean), Im(y_mean), y_power, x_BD[t], snr]
        y_real2 = tf.concat([
            tf.cast(tf.real(y_samples_flat), tf.float32),
            tf.cast(tf.imag(y_samples_flat), tf.float32),
            tf.cast(tf.real(y_mean), tf.float32),
            tf.cast(tf.imag(y_mean), tf.float32),
            tf.cast(y_power, tf.float32)
        ], axis=1) #/ tf.sqrt(lay['P'])  # (batch, 2*N_rx + 3)
        
        'Rx updates LSTM state and designs beamformers'
        h_old2, c_old2 = LSTM2((tf.concat([y_real2, x_BD[t], snr_normal], axis=1), h_old2, c_old2))
        
        # Rx designs receive beamformer v
        v_her = mlp_re_rx(h_old2)
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm + 1e-8)
        v1 = tf.complex(v_her[:, 0:N_rx], v_her[:, N_rx:2 * N_rx])
        v1 = tf.reshape(v1, [-1, N_rx, 1])
        
        # Rx designs transmit beamformer for reverse link (used by Tx to observe)
        v_tx_her = mlp_re_tx(h_old2)  # Reusing mlp_tx for Rx->Tx transmission design
        v_tx_norm = tf.reshape(tf.norm(v_tx_her, axis=1), (-1, 1))
        v_tx_her = tf.divide(v_tx_her, v_tx_norm + 1e-8)
        v2 = tf.complex(v_tx_her[:, 0:N_rx], v_tx_her[:, N_rx:2 * N_rx])  # Tx dim for reverse
        v2 = tf.reshape(v2, [-1, N_rx, 1])
        
        'Tx observes reverse link: y = sqrt(P) * w^H * H_rev * v2 * s + noise'
        y_noiseless_before_tx = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.matmul(H_eff_rev, v2)  # (batch, N_tx, 1)
        y_noiseless_before_tx = tf.tile(y_noiseless_before_tx, [1, 1, K])  # (batch, N_tx, K)
        noise_tx = tf.complex(
            tf.random_normal([batch_size, N_tx, K], mean=0.0, stddev=noiseSTD_per_dim),
            tf.random_normal([batch_size, N_tx, K], mean=0.0, stddev=noiseSTD_per_dim)
        )
        ytx_complex_nobf = y_noiseless_before_tx + noise_tx
        ytx_sample_flat = tf.reduce_mean(ytx_complex_nobf, axis=2)  # (batch, N_tx)
        
        y_complex_tx_bf = tf.matmul(tf.linalg.adjoint(w2), ytx_complex_nobf)
        y_complex_tx_bf = tf.squeeze(y_complex_tx_bf, axis=1)  # (batch, K)
        
        y_mean_rev = tf.reduce_mean(y_complex_tx_bf, axis=1)[:,tf.newaxis]
        y_power_rev = tf.reduce_mean(tf.abs(y_complex_tx_bf) ** 2, axis=1)[:,tf.newaxis]
        
        y_real1 = tf.concat([
            tf.cast(tf.real(ytx_sample_flat), tf.float32),
            tf.cast(tf.imag(ytx_sample_flat), tf.float32),
            tf.cast(tf.real(y_mean_rev), tf.float32),
            tf.cast(tf.imag(y_mean_rev), tf.float32),
            tf.cast(y_power_rev, tf.float32)
        ], axis=1) #/ tf.sqrt(lay['P'])
        
        'Tx updates LSTM state and designs beamformers'
        h_old1, c_old1 = LSTM1((tf.concat([y_real1, x_BD[t], snr_normal], axis=1), h_old1, c_old1))

        # Tx designs receive beamformer for next round
        w2_her = mlp_ris_rx(h_old1)
        w2_norm = tf.reshape(tf.norm(w2_her, axis=1), (-1, 1))
        w2_her = tf.divide(w2_her, w2_norm + 1e-8)
        w2 = tf.complex(w2_her[:, 0:N_tx], w2_her[:, N_tx:2 * N_tx])
        w2 = tf.reshape(w2, [-1, N_tx, 1])

        # Tx designs transmit beamformer w for next round
        w_her = mlp_ris_tx(h_old1)
        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm + 1e-8)
        w1 = tf.complex(w_her[:, 0:N_tx], w_her[:, N_tx:2 * N_tx])
        w1 = tf.reshape(w1, [-1, N_tx, 1])
        
        # Store beamformers
        v_list.append(tf.concat([tf.real(tf.squeeze(v1, axis=2)), tf.imag(tf.squeeze(v1, axis=2))], axis=1))
    
    #####################################################
    # Output Final Beamformers after tau interactions
    #####################################################
    MLP_bf_w = MLPBlock(3, [2 * hidden_size1, 2 * hidden_size1, 2 * N_tx], name='MLP_bf_w')
    MLP_bf_v = MLPBlock(3, [2 * hidden_size2, 2 * hidden_size2, 2 * N_rx], name='MLP_bf_v')
    
    # Final RIS transmit beamformer w 
    w_tmp = MLP_bf_w(c_old1)
    w_norm = tf.reshape(tf.norm(w_tmp, axis=1), (-1, 1))
    w_tmp = tf.divide(w_tmp, w_norm + 1e-8)
    w_complex = tf.complex(w_tmp[:, 0:N_tx], w_tmp[:, N_tx:2 * N_tx])
    w_complex = tf.reshape(w_complex, [-1, N_tx, 1])
    
    # Final Rx receive beamformer v 
    v_tmp = MLP_bf_v(c_old2)
    v_norm = tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1))
    v_tmp = tf.divide(v_tmp, v_norm + 1e-8)
    v_complex = tf.complex(v_tmp[:, 0:N_rx], v_tmp[:, N_rx:2 * N_rx])
    v_complex = tf.reshape(v_complex, [-1, N_rx, 1])


#####################################################
# Loss Function: Maximize BD SINR
# SINR = P_s * |v^H H_b w|^2 / (P_s * |v^H (H_d + H_r) w|^2 + noise_power)
#####################################################

with tf.name_scope("sinr_computation"):
    
    # BD signal power: P * |v^H H_b w|^2
    # v^H: (batch, 1, N_rx), H_b: (batch, N_rx, N_tx), w: (batch, N_tx, 1)
    sig_BD = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_b_placeholder, w_complex))  # (batch, 1, 1)
    sig_BD = tf.squeeze(tf.abs(sig_BD) ** 2) * lay['P']  # (batch,)
    
    # Interference channel: H_d + H_r
    H_interference = H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
    
    # Interference power: P * |v^H (H_d + H_r) w|^2
    sig_int = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_interference, w_complex))  # (batch, 1, 1)
    sig_int = tf.squeeze(tf.abs(sig_int) ** 2) * lay['P']  # (batch,)
    
    # BD SINR = P * |v^H H_b w|^2 / (P * |v^H (H_d + H_r) w|^2 + noise_var)
    sinr_BD = sig_BD / (sig_int + noise_var + 1e-10)
    sinr_BD_clipped = tf.clip_by_value(sinr_BD, 1e-4, 1e4)
    
    # Log SINR with per-sample clipping to prevent outliers from dominating
    log_sinr_BD_raw = tf.log(sinr_BD_clipped + 1e-9)
    log_sinr_BD = tf.clip_by_value(log_sinr_BD_raw, -8.0, 8.0)  # ~±35 dB range
    
    # For backward compatibility, define sig_ref as interference
    sig_ref = sig_int
    
    # Compute sigma1_sq and sigma2_sq for BD state detection metrics
    # g1 = -H_b + H_d + H_r (BD = -1), g2 = H_b + H_d + H_r (BD = +1)
    H_g1 = -H_b_placeholder + H_d_placeholder + H_r_placeholder
    H_g2 = H_b_placeholder + H_d_placeholder + H_r_placeholder
    
    sigma1_sq = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_g1, w_complex))
    sigma2_sq = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_g2, w_complex))
    sigma1_sq = tf.squeeze(tf.abs(sigma1_sq) ** 2) * lay['P'] + noise_var
    sigma2_sq = tf.squeeze(tf.abs(sigma2_sq) ** 2) * lay['P'] + noise_var
    
    # Threshold and error probability (for evaluation)
    Th = sigma1_sq * sigma2_sq / (sigma1_sq - sigma2_sq + 1e-10) * tf.log(sigma1_sq / (sigma2_sq + 1e-10) + 1e-10)
    pe = tf.where(
        sigma1_sq > sigma2_sq,
        0.5 + 0.5 * tf.exp(-Th / (sigma2_sq + 1e-10)) - 0.5 * tf.exp(-Th / (sigma1_sq + 1e-10)),
        0.5 + 0.5 * tf.exp(-Th / (sigma1_sq + 1e-10)) - 0.5 * tf.exp(-Th / (sigma2_sq + 1e-10))
    )


#####################################################
# Optimal Beamformer (SVD-based solution for MIMO)
# Maximize SINR = |v^H H_b w|^2 / (|v^H H_int w|^2 + noise_var/P)
#####################################################

with tf.name_scope("optimal_beamformer"):
    batch_size_opt = tf.shape(H_b_placeholder)[0]
    
    P_c = tf.cast(lay['P'], tf.complex64)
    N0_c = tf.cast(noise_var, tf.complex64)
    
    # Use tf.py_func to call NumPy-based optimization
    def compute_optimal_beamformers_np(H_b, H_int, noise_var_val, P_val):
        """Compute optimal beamformers for a batch using manifold optimization"""
        batch_size = H_b.shape[0]
        v_opt_batch = np.zeros((batch_size, H_b.shape[1], 1), dtype=np.complex64)
        w_opt_batch = np.zeros((batch_size, H_b.shape[2], 1), dtype=np.complex64)
        
        for i in range(batch_size):
            H_b_i = H_b[i]  # (N_rx, N_tx)
            H_int_i = H_int[i]  # (N_rx, N_tx)
            
            # Scale by sqrt(P)
            A = np.sqrt(P_val) * H_b_i
            B = np.sqrt(P_val) * H_int_i
            
            try:
                w_opt_i, v_opt_i, _ = solve_with_random_restarts(A, B, c=noise_var_val, restarts=10)
                v_opt_batch[i, :, 0] = v_opt_i
                w_opt_batch[i, :, 0] = w_opt_i
            except Exception as e:
                # Fallback to SVD if optimization fails
                U, S, Vh = np.linalg.svd(H_b_i)
                v_opt_batch[i, :, 0] = U[:, 0]
                w_opt_batch[i, :, 0] = Vh[0, :]
        
        return v_opt_batch.astype(np.complex64), w_opt_batch.astype(np.complex64)
    
    # Wrap in tf.py_func
    v_opt, w_opt = tf.py_func(
        lambda H_b, H_int: compute_optimal_beamformers_np(H_b, H_int, noise_var, Pvec[0]),
        [H_b_placeholder, H_interference],
        [tf.complex64, tf.complex64]
    )
    
    # Set shapes explicitly
    v_opt = tf.reshape(v_opt, [-1, N_rx, 1])
    w_opt = tf.reshape(w_opt, [-1, N_tx, 1])
    # Optimal SINR with SVD-based beamformers
    sig_BD_opt = tf.matmul(tf.linalg.adjoint(v_opt), tf.matmul(H_b_placeholder, w_opt))
    sig_BD_opt = tf.squeeze(tf.abs(sig_BD_opt) ** 2) * lay['P']
    
    sig_int_opt = tf.matmul(tf.linalg.adjoint(v_opt), tf.matmul(H_interference, w_opt))
    sig_int_opt = tf.squeeze(tf.abs(sig_int_opt) ** 2) * lay['P']
    
    sinr_BD_opt = sig_BD_opt / (sig_int_opt + noise_var + 1e-10)
    
    # Random beamformer baseline
    bf_gain_rnd = tf.reduce_mean(tf.abs(tf.matmul(
        tf.linalg.adjoint(bf_gain_init_v), 
        tf.matmul(H_b_placeholder, bf_gain_init_w)
    )) ** 2)


#####################################################
# Loss and Optimizer
#####################################################

# Primary loss: Maximize BD SINR (minimize negative log SINR)
loss = -tf.reduce_mean(log_sinr_BD)

# Regularization
global_step = tf.train.get_or_create_global_step()
l2 = 1e-5
reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss_reg = loss + l2 * reg_term


# Add warmup and slower decay
warmup_steps = 300
global_step_float = tf.cast(global_step, tf.float32)
warmup_lr = learning_rate * tf.minimum(1.0, global_step_float / warmup_steps)

decayed_lr = tf.train.exponential_decay(
    learning_rate, 
    global_step, 
    decay_steps=2000,  # Slower decay
    decay_rate=0.96,    # Gentler decay
    staircase=True
)

lr = tf.where(global_step < warmup_steps, warmup_lr, decayed_lr)
# Learning rate schedule
# lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Gradient computation with safety checks
grads_vars = optimizer.compute_gradients(loss_reg)
safe_grads = []
vars_list = []
for g, v in grads_vars:
    if g is None:
        safe_grads.append(None)
        vars_list.append(v)
    else:
        g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        safe_grads.append(g)
        vars_list.append(v)

clipped_grads, global_norm = tf.clip_by_global_norm([g for g in safe_grads if g is not None], 1.0)

final_grads = []
clip_index = 0
for g in safe_grads:
    if g is None:
        final_grads.append(None)
    else:
        final_grads.append(clipped_grads[clip_index])
        clip_index += 1

# NaN checks
nan_checks = [
    tf.check_numerics(tf.real(v_complex), 'v_complex real NaN'),
    tf.check_numerics(tf.imag(v_complex), 'v_complex imag NaN'),
    tf.check_numerics(tf.real(w_complex), 'w_complex real NaN'),
    tf.check_numerics(tf.imag(w_complex), 'w_complex imag NaN'),
    tf.check_numerics(log_sinr_BD, 'log_sinr_BD NaN')
]

with tf.control_dependencies(nan_checks):
    training_op = optimizer.apply_gradients(list(zip(final_grads, vars_list)), global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def sample_bistatic_batch(num_samples):
    """Generate one batched bi-static dataset sample for training/evaluation."""
    bd_locations = generate_location_mimo_batch(num_samples, 1, 'u')[:, 0, :]
    if num_scatters > 0:
        scatter_locations = generate_location_mimo_batch(num_samples, num_scatters, 's')
    else:
        scatter_locations = np.zeros((num_samples, 0, 3), dtype=np.float32)

    (_, H_d, H_r, H_b,
     _, H_d_rev, H_r_rev, H_b_rev) = generate_bistatic_mimo_channel_batch(
        location_tx,
        location_rx,
        scatter_locations,
        bd_locations,
        N_tx_h=N_tx_h_ch,
        N_tx_v=N_tx_v_ch,
        N_rx_h=N_rx_h_ch,
        N_rx_v=N_rx_v_ch,
        Rician_factor=Rician_factor
    )
    loc_features = build_mimo_location_features(bd_locations, location_tx, location_rx)
    return loc_features, H_d, H_b, H_r, H_d_rev, H_b_rev, H_r_rev, bd_locations, scatter_locations


#####################################################
# Data Generation
#####################################################

# Validation set
num_val_samples = val_size_order * 32

set_location_user_val, H_d_val, H_b_val, H_r_val, H_d_val_rev, H_b_val_rev, H_r_val_rev, _, _ = sample_bistatic_batch(num_val_samples)

# QPSK ambient signal
qpsk_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2)
s_signal_val = qpsk_symbols[np.random.randint(0, 4, size=(val_size_order * 32, 1, num_subcarriers))].astype(np.complex64)

# Prepare batch-sized channel matrices
H_d_val_batch = H_d_val  # Already (num_val_samples, N_rx, N_tx)
H_r_val_batch = H_r_val  # Already (num_val_samples, N_rx, N_tx)

feed_dict_val = {
    loc_input: set_location_user_val,
    lay['P']: Pvec[0],
    H_d_placeholder: H_d_val_batch,
    H_b_placeholder: H_b_val,
    H_r_placeholder: H_r_val_batch,
    H_d_rev_placeholder: H_d_val_rev,
    H_b_rev_placeholder: H_b_val_rev,
    H_r_rev_placeholder: H_r_val_rev,
    s_placeholder: s_signal_val
}


#####################################################
# Training Loop
#####################################################

print("\n" + "=" * 60)
print("BD SINR Maximization via Active Sensing")
print("=" * 60)
print(f"N_tx: {N_tx}, N_rx: {N_rx}, tau: {tau}, K: {K}, SNR: {snr_const[0]} dB")
print("=" * 60 + "\n")

with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, f'{drive_save_path}/params_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}')
    
    # Early stopping
    best_val = 1e9
    wait = 0
    PATIENCE = 20
    
    for epoch in range(n_epochs):
        batch_iter = 0
        epoch_train_losses = []
        epoch_sinr_values = []
        num_train_samples = batch_size_order * 32
        
        # QPSK ambient signal
        s_signal = qpsk_symbols[np.random.randint(0, 4, size=(num_train_samples, 1, num_subcarriers))].astype(np.complex64)
        
        for rnd_indices in range(batch_per_epoch):
            (set_location_user_train, H_d_train, H_b_train, H_r_train,
             H_d_train_rev, H_b_train_rev, H_r_train_rev, _, _) = sample_bistatic_batch(num_train_samples)

            H_d_train_batch = H_d_train  # Already correct shape
            H_r_train_batch = H_r_train  # Already correct shape

            feed_dict_batch = {
                loc_input: set_location_user_train,
                lay['P']: Pvec[0],
                H_d_placeholder: H_d_train_batch,
                H_b_placeholder: H_b_train,
                H_r_placeholder: H_r_train_batch,
                H_d_rev_placeholder: H_d_train_rev,
                H_b_rev_placeholder: H_b_train_rev,
                H_r_rev_placeholder: H_r_train_rev,
                s_placeholder: s_signal
            }
            
            _, train_loss, sinr_values, pe_values = sess.run(
                [training_op, loss, sinr_BD, pe], feed_dict=feed_dict_batch
            )
            
            epoch_train_losses.append(train_loss)
            epoch_sinr_values.append(np.mean(sinr_values))
            batch_iter += 1
        
        # Validation
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_sinr = np.mean(epoch_sinr_values)
        
        loss_val, sinr_val, sinr_opt_val, pe_val, \
            sig_bd_val, sig_ref_val, sig_bd_opt_val, sig_int_opt_val = sess.run(
            [loss, sinr_BD, sinr_BD_opt, pe, \
             sig_BD, sig_ref, sig_BD_opt, sig_int_opt], feed_dict=feed_dict_val
        )
        
        print(f'Epoch {epoch:3d} | '
              f'Train Loss: {avg_train_loss:8.4f} | '
              f'Val Loss: {loss_val:8.4f} | '
              f'Best: {best_val:8.4f}')
        print(f'         | '
              f'SINR_BD (learned): {10 * np.log10(np.mean(sinr_val) + 1e-10):6.2f} dB | '
              f'SINR_BD (optimal): {10 * np.log10(np.mean(sinr_opt_val) + 1e-10):6.2f} dB')
        print(f'         | '
              f'Sig_BD: {np.mean(sig_bd_val):8.4f} | '
              f'Sig_int: {np.mean(sig_ref_val):8.4f} | '
              f'Sig_BD_opt: {np.mean(sig_bd_opt_val):8.4f} | '
              f'Sig_int_opt: {np.mean(sig_int_opt_val):8.4f} | '
              f'PE: {np.mean(pe_val):6.4f}')
        print()
        
        # Early stopping
        if loss_val < best_val - 1e-9:
            best_val = loss_val
            wait = 0
            saver.save(sess, f'{drive_save_path}/params_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}')
            with open(os.path.join(drive_save_path, "best_val_loss.txt"), "w") as f:
                f.write(str(best_val))
        else:
            wait += 1
            if wait == PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    #####################################################
    # Testing
    #####################################################
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)
    
    # Generate test batch
    (set_location_user_test, H_d_test, H_b_test, H_r_test,
     H_d_test_rev, H_b_test_rev, H_r_test_rev, BD_loc, Scatter_loc) = sample_bistatic_batch(test_size)

    H_d_test_batch = H_d_test
    H_r_test_batch = H_r_test
    
    s_signal_test = qpsk_symbols[np.random.randint(0, 4, size=(test_size, 1, num_subcarriers))].astype(np.complex64)
    
    feed_dict_test = {
        loc_input: set_location_user_test,
        lay['P']: Pvec[0],
        H_d_placeholder: H_d_test_batch,
        H_b_placeholder: H_b_test,
        H_r_placeholder: H_r_test_batch,
        H_d_rev_placeholder: H_d_test_rev,
        H_b_rev_placeholder: H_b_test_rev,
        H_r_rev_placeholder: H_r_test_rev,
        s_placeholder: s_signal_test
    }
    
    sinr_test, sinr_opt_test, pe_test, v_learned, w_learned, v_optimal, w_optimal = sess.run(
        [sinr_BD, sinr_BD_opt, pe, v_complex, w_complex, v_opt, w_opt], feed_dict=feed_dict_test
    )
    
    print(f"Test SINR_BD (learned):  {10 * np.log10(np.mean(sinr_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_BD (optimal):  {10 * np.log10(np.mean(sinr_opt_test) + 1e-10):6.2f} dB")
    print(f"Test PE (learned):       {np.mean(pe_test):6.4f}")
    print(f"Gap to optimal:          {10 * np.log10((np.mean(sinr_opt_test) + 1e-10) / (np.mean(sinr_test) + 1e-10)):6.2f} dB")
    
    # Save results
    model_filename = os.path.join(drive_save_path, f'TEST_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}.mat')
    sio.savemat(model_filename, dict(
        snr_const=snr_const,
        N_tx=N_tx,
        N_rx=N_rx,
        tau=tau,
        K=K,
        BD_location=BD_loc,
        Scatter_location=Scatter_loc if num_scatters > 0 else [],
        sinr_learned=sinr_test,
        sinr_optimal=sinr_opt_test,
        pe_learned=pe_test,
        v_learned=v_learned,
        w_learned=w_learned,
        v_optimal=v_optimal,
        w_optimal=w_optimal
    ))
    print(f"\nResults saved to {model_filename}")




# %%    
    #####################################################
    # 3D Beam Pattern Visualization
    #####################################################
print("\n" + "=" * 60)
print("Beam Pattern Visualization")
print("=" * 60)

from mpl_toolkits.mplot3d import Axes3D

# Select one test instance for visualization
idx = 8
bd_loc_vis = BD_loc[idx]
if num_scatters > 0:
    scatter_loc_vis = Scatter_loc[idx]
    scatter_loc_vis = np.asarray(scatter_loc_vis)
    if scatter_loc_vis.ndim > 1:
        scatter_loc_vis = scatter_loc_vis[0]
else:
    scatter_loc_vis = np.array([0, 0, 0])

# Reshape beamformers to 2D arrays for UPA
N_tx_h = int(np.sqrt(N_tx))
N_tx_v = int(np.sqrt(N_tx))
N_rx_h = int(np.sqrt(N_rx))
N_rx_v = int(np.sqrt(N_rx))

# Function to compute beam pattern for UPA
def compute_beam_pattern(beamformer, N_h, N_v, wavelength, num_points=100):
    """Compute 3D beam pattern for a UPA beamformer"""
    azimuth = np.linspace(-np.pi, np.pi, num_points)
    elevation = np.linspace(-np.pi/2, np.pi/2, num_points)
    AZ, EL = np.meshgrid(azimuth, elevation)
    
    pattern = np.zeros_like(AZ, dtype=np.float64)  # Use float, not complex
    bf = beamformer.flatten()
    
    for i_el in range(num_points):
        for i_az in range(num_points):
            az = azimuth[i_az]
            el = elevation[i_el]
            # Steering vector for UPA
            a = np.zeros(N_h * N_v, dtype=complex)
            for m in range(N_h):
                for n in range(N_v):
                    a[m * N_v + n] = np.exp(1j * np.pi * (m * np.sin(az) * np.cos(el) + n * np.sin(el)))
            pattern[i_el, i_az] = np.abs(np.dot(np.conj(bf), a)) ** 2
    
    # Normalize
    pattern = pattern / (np.max(pattern) + 1e-10)
    return AZ, EL, pattern

# Convert spherical to Cartesian for beam pattern visualization
def spherical_to_cartesian(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

# Create figure with 2 rows, 2 columns
fig = plt.figure(figsize=(18, 12))

# Get learned and optimal beamformers for this instance
w_learned_vis = w_learned[idx]  # (N_tx, 1)
v_learned_vis = v_learned[idx]  # (N_rx, 1)
w_optimal_vis = w_optimal[idx]  # (N_tx, 1)
v_optimal_vis = v_optimal[idx]  # (N_rx, 1)

# Compute beam patterns for learned beamformers
AZ_tx_learned, EL_tx_learned, pattern_tx_learned = compute_beam_pattern(w_learned_vis, N_tx_h, N_tx_v, Wavelength, num_points=50)
AZ_rx_learned, EL_rx_learned, pattern_rx_learned = compute_beam_pattern(v_learned_vis, N_rx_h, N_rx_v, Wavelength, num_points=50)

# Compute beam patterns for optimal beamformers
AZ_tx_optimal, EL_tx_optimal, pattern_tx_optimal = compute_beam_pattern(w_optimal_vis, N_tx_h, N_tx_v, Wavelength, num_points=50)
AZ_rx_optimal, EL_rx_optimal, pattern_rx_optimal = compute_beam_pattern(v_optimal_vis, N_rx_h, N_rx_v, Wavelength, num_points=50)

# ===== ROW 1: LEARNED BEAMFORMERS =====
# ===== Row 1, Col 1: 3D Scene with Learned Beam Patterns =====
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Plot Tx array (as a small grid of points)
tx_antenna_pos = []
for m in range(N_tx_h):
    for n in range(N_tx_v):
        pos = location_tx + np.array([n - (N_tx_h+1)/2, m - (N_tx_v+1)/2, 0]) * Wavelength/2
        tx_antenna_pos.append(pos)
tx_antenna_pos = np.array(tx_antenna_pos)
ax1.scatter(tx_antenna_pos[:, 0], tx_antenna_pos[:, 1], tx_antenna_pos[:, 2], 
            c='blue', marker='s', s=100, label='Tx Array', alpha=0.8, zorder=10)

# Plot Rx array
rx_antenna_pos = []
for m in range(N_rx_h):
    for n in range(N_rx_v):
        pos = location_rx + np.array([n - (N_rx_h+1)/2, m - (N_rx_v+1)/2, 0]) * Wavelength/2
        rx_antenna_pos.append(pos)
rx_antenna_pos = np.array(rx_antenna_pos)
ax1.scatter(rx_antenna_pos[:, 0], rx_antenna_pos[:, 1], rx_antenna_pos[:, 2], 
            c='green', marker='^', s=100, label='Rx Array', alpha=0.8, zorder=10)

# Plot BD location
ax1.scatter(bd_loc_vis[0], bd_loc_vis[1], bd_loc_vis[2], 
            c='red', marker='*', s=400, label='BD', edgecolors='black', linewidth=2, zorder=10)

# Plot Scatter location
ax1.scatter(scatter_loc_vis[0], scatter_loc_vis[1], scatter_loc_vis[2], 
            c='orange', marker='o', s=250, label='Scatterer', edgecolors='black', linewidth=1.5, zorder=10)

# Draw lines showing signal paths
# ax1.plot([location_tx[0], bd_loc_vis[0]], [location_tx[1], bd_loc_vis[1]], 
#             [location_tx[2], bd_loc_vis[2]], 'r--', alpha=0.6, linewidth=2, label='Tx-BD path', zorder=5)
# ax1.plot([bd_loc_vis[0], location_rx[0]], [bd_loc_vis[1], location_rx[1]], 
#             [bd_loc_vis[2], location_rx[2]], 'r--', alpha=0.6, linewidth=2, zorder=5)
# ax1.plot([location_tx[0], scatter_loc_vis[0]], [location_tx[1], scatter_loc_vis[1]], 
#             [location_tx[2], scatter_loc_vis[2]], 'orange', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)
# ax1.plot([scatter_loc_vis[0], location_rx[0]], [scatter_loc_vis[1], location_rx[1]], 
#             [scatter_loc_vis[2], location_rx[2]], 'orange', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)

# ===== Add Learned Tx Beam Pattern to Scene =====
beam_scale = 5  # Scale factor for beam visualization
X_tx_learned, Y_tx_learned, Z_tx_learned = spherical_to_cartesian(AZ_tx_learned, EL_tx_learned, pattern_tx_learned * beam_scale)
X_tx_learned += location_tx[0]
Y_tx_learned += location_tx[1]
Z_tx_learned += location_tx[2]

# Plot Tx beam pattern with semi-transparent surface
surf_tx = ax1.plot_surface(X_tx_learned, Y_tx_learned, Z_tx_learned, cmap='Blues', alpha=0.4, 
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# ===== Add Learned Rx Beam Pattern to Scene =====
X_rx_learned, Y_rx_learned, Z_rx_learned = spherical_to_cartesian(AZ_rx_learned, EL_rx_learned, pattern_rx_learned * beam_scale)
X_rx_learned += location_rx[0]
Y_rx_learned += location_rx[1]
Z_rx_learned += location_rx[2]

# Plot Rx beam pattern with semi-transparent surface
surf_rx = ax1.plot_surface(X_rx_learned, Y_rx_learned, Z_rx_learned, cmap='Greens', alpha=0.4,
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# Mark direction to BD from Tx
dir_to_bd = (bd_loc_vis - location_tx) / np.linalg.norm(bd_loc_vis - location_tx)
ax1.quiver(location_tx[0], location_tx[1], location_tx[2], 
            dir_to_bd[0]*4, dir_to_bd[1]*4, dir_to_bd[2]*4, 
            color='darkblue', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

# Mark direction to BD from Rx
dir_to_bd_rx = (bd_loc_vis - location_rx) / np.linalg.norm(bd_loc_vis - location_rx)
ax1.quiver(location_rx[0], location_rx[1], location_rx[2], 
            dir_to_bd_rx[0]*4, dir_to_bd_rx[1]*4, dir_to_bd_rx[2]*4, 
            color='darkgreen', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

ax1.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
ax1.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
ax1.set_title('Learned Beamformers - 3D Scene\n(Blue: Tx Beam, Green: Rx Beam)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Add coordinate axis arrows at origin for reference
axis_length = 3
ax1.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax1.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax1.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax1.text(axis_length*1.1, 0, 0, 'X', color='red', fontsize=10, fontweight='bold')
ax1.text(0, axis_length*1.1, 0, 'Y', color='green', fontsize=10, fontweight='bold')
ax1.text(0, 0, axis_length*1.1, 'Z', color='blue', fontsize=10, fontweight='bold')

# Set better viewing angle
ax1.view_init(elev=20, azim=45)

# ===== Row 1, Col 2: 2D Beam Pattern for Learned Beamformers =====
ax2 = fig.add_subplot(2, 2, 2)

# Take a horizontal cut (elevation = 0)
el_idx = pattern_tx_learned.shape[0] // 2
azimuth_deg = np.linspace(-180, 180, pattern_tx_learned.shape[1])

ax2.plot(azimuth_deg, 10*np.log10(pattern_tx_learned[el_idx, :] + 1e-10), 'b-', 
            linewidth=2.5, label='Tx Beam', alpha=0.8)
ax2.plot(azimuth_deg, 10*np.log10(pattern_rx_learned[el_idx, :] + 1e-10), 'g-', 
            linewidth=2.5, label='Rx Beam', alpha=0.8)

# Mark BD direction
bd_azimuth_tx = np.arctan2(bd_loc_vis[1] - location_tx[1], bd_loc_vis[0] - location_tx[0]) * 180/np.pi
bd_azimuth_rx = np.arctan2(bd_loc_vis[1] - location_rx[1], bd_loc_vis[0] - location_rx[0]) * 180/np.pi
ax2.axvline(bd_azimuth_tx, color='darkblue', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Tx: {bd_azimuth_tx:.1f}°')
ax2.axvline(bd_azimuth_rx, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Rx: {bd_azimuth_rx:.1f}°')

# Mark scatter direction
sc_azimuth_tx = np.arctan2(scatter_loc_vis[1] - location_tx[1], scatter_loc_vis[0] - location_tx[0]) * 180/np.pi
sc_azimuth_rx = np.arctan2(scatter_loc_vis[1] - location_rx[1], scatter_loc_vis[0] - location_rx[0]) * 180/np.pi
ax2.axvline(sc_azimuth_tx, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Scatter from Tx: {sc_azimuth_tx:.1f}°')

ax2.set_xlabel('Azimuth Angle (degrees)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Normalized Gain (dB)', fontsize=10, fontweight='bold')
ax2.set_title('Learned Beamformers - Azimuth Cut (El = 0°)', fontsize=11, fontweight='bold')
ax2.set_xlim([-180, 180])
ax2.set_ylim([-30, 5])
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Add text annotations for better understanding
ax2.text(0.02, 0.98, f'N_tx = {N_tx}, N_rx = {N_rx}\nτ = {tau}, SNR = {snr_const[0]} dB', 
         transform=ax2.transAxes, fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ===== ROW 2: OPTIMAL BEAMFORMERS =====
# ===== Row 2, Col 1: 3D Scene with Optimal Beam Patterns =====
ax3 = fig.add_subplot(2, 2, 3, projection='3d')

# Plot Tx array (as a small grid of points)
tx_antenna_pos = []
for m in range(N_tx_h):
    for n in range(N_tx_v):
        pos = location_tx + np.array([n - (N_tx_h+1)/2, m - (N_tx_v+1)/2, 0]) * Wavelength/2
        tx_antenna_pos.append(pos)
tx_antenna_pos = np.array(tx_antenna_pos)
ax3.scatter(tx_antenna_pos[:, 0], tx_antenna_pos[:, 1], tx_antenna_pos[:, 2], 
            c='blue', marker='s', s=100, label='Tx Array', alpha=0.8, zorder=10)

# Plot Rx array
rx_antenna_pos = []
for m in range(N_rx_h):
    for n in range(N_rx_v):
        pos = location_rx + np.array([n - (N_rx_h+1)/2, m - (N_rx_v+1)/2, 0]) * Wavelength/2
        rx_antenna_pos.append(pos)
rx_antenna_pos = np.array(rx_antenna_pos)
ax3.scatter(rx_antenna_pos[:, 0], rx_antenna_pos[:, 1], rx_antenna_pos[:, 2], 
            c='green', marker='^', s=100, label='Rx Array', alpha=0.8, zorder=10)

# Plot BD location
ax3.scatter(bd_loc_vis[0], bd_loc_vis[1], bd_loc_vis[2], 
            c='red', marker='*', s=400, label='BD', edgecolors='black', linewidth=2, zorder=10)

# Plot Scatter location
ax3.scatter(scatter_loc_vis[0], scatter_loc_vis[1], scatter_loc_vis[2], 
            c='orange', marker='o', s=250, label='Scatterer', edgecolors='black', linewidth=1.5, zorder=10)

# Draw lines showing signal paths
ax3.plot([location_tx[0], bd_loc_vis[0]], [location_tx[1], bd_loc_vis[1]], 
            [location_tx[2], bd_loc_vis[2]], 'r--', alpha=0.6, linewidth=2, label='Tx-BD path', zorder=5)
ax3.plot([bd_loc_vis[0], location_rx[0]], [bd_loc_vis[1], location_rx[1]], 
            [bd_loc_vis[2], location_rx[2]], 'r--', alpha=0.6, linewidth=2, zorder=5)
ax3.plot([location_tx[0], scatter_loc_vis[0]], [location_tx[1], scatter_loc_vis[1]], 
            [location_tx[2], scatter_loc_vis[2]], 'orange', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)
ax3.plot([scatter_loc_vis[0], location_rx[0]], [scatter_loc_vis[1], location_rx[1]], 
            [scatter_loc_vis[2], location_rx[2]], 'orange', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)

# ===== Add Optimal Tx Beam Pattern to Scene =====
X_tx_optimal, Y_tx_optimal, Z_tx_optimal = spherical_to_cartesian(AZ_tx_optimal, EL_tx_optimal, pattern_tx_optimal * beam_scale)
X_tx_optimal += location_tx[0]
Y_tx_optimal += location_tx[1]
Z_tx_optimal += location_tx[2]

# Plot Tx beam pattern with semi-transparent surface
surf_tx_opt = ax3.plot_surface(X_tx_optimal, Y_tx_optimal, Z_tx_optimal, cmap='Blues', alpha=0.4, 
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# ===== Add Optimal Rx Beam Pattern to Scene =====
X_rx_optimal, Y_rx_optimal, Z_rx_optimal = spherical_to_cartesian(AZ_rx_optimal, EL_rx_optimal, pattern_rx_optimal * beam_scale)
X_rx_optimal += location_rx[0]
Y_rx_optimal += location_rx[1]
Z_rx_optimal += location_rx[2]

# Plot Rx beam pattern with semi-transparent surface
surf_rx_opt = ax3.plot_surface(X_rx_optimal, Y_rx_optimal, Z_rx_optimal, cmap='Greens', alpha=0.4,
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# Mark direction to BD from Tx
dir_to_bd = (bd_loc_vis - location_tx) / np.linalg.norm(bd_loc_vis - location_tx)
ax3.quiver(location_tx[0], location_tx[1], location_tx[2], 
            dir_to_bd[0]*4, dir_to_bd[1]*4, dir_to_bd[2]*4, 
            color='darkblue', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

# Mark direction to BD from Rx
dir_to_bd_rx = (bd_loc_vis - location_rx) / np.linalg.norm(bd_loc_vis - location_rx)
ax3.quiver(location_rx[0], location_rx[1], location_rx[2], 
            dir_to_bd_rx[0]*4, dir_to_bd_rx[1]*4, dir_to_bd_rx[2]*4, 
            color='darkgreen', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

ax3.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
ax3.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
ax3.set_title('Optimal Beamformers - 3D Scene\n(Blue: Tx Beam, Green: Rx Beam)', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Add coordinate axis arrows at origin for reference
axis_length = 3
ax3.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax3.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax3.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax3.text(axis_length*1.1, 0, 0, 'X', color='red', fontsize=10, fontweight='bold')
ax3.text(0, axis_length*1.1, 0, 'Y', color='green', fontsize=10, fontweight='bold')
ax3.text(0, 0, axis_length*1.1, 'Z', color='blue', fontsize=10, fontweight='bold')

# Set better viewing angle
ax3.view_init(elev=20, azim=45)

# ===== Row 2, Col 2: 2D Beam Pattern for Optimal Beamformers =====
ax4 = fig.add_subplot(2, 2, 4)

# Take a horizontal cut (elevation = 0)
el_idx = pattern_tx_optimal.shape[0] // 2
azimuth_deg = np.linspace(-180, 180, pattern_tx_optimal.shape[1])

ax4.plot(azimuth_deg, 10*np.log10(pattern_tx_optimal[el_idx, :] + 1e-10), 'b-', 
            linewidth=2.5, label='Tx Beam', alpha=0.8)
ax4.plot(azimuth_deg, 10*np.log10(pattern_rx_optimal[el_idx, :] + 1e-10), 'g-', 
            linewidth=2.5, label='Rx Beam', alpha=0.8)

# Mark BD direction
bd_azimuth_tx = np.arctan2(bd_loc_vis[1] - location_tx[1], bd_loc_vis[0] - location_tx[0]) * 180/np.pi
bd_azimuth_rx = np.arctan2(bd_loc_vis[1] - location_rx[1], bd_loc_vis[0] - location_rx[0]) * 180/np.pi
ax4.axvline(bd_azimuth_tx, color='darkblue', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Tx: {bd_azimuth_tx:.1f}°')
ax4.axvline(bd_azimuth_rx, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Rx: {bd_azimuth_rx:.1f}°')

# Mark scatter direction
sc_azimuth_tx = np.arctan2(scatter_loc_vis[1] - location_tx[1], scatter_loc_vis[0] - location_tx[0]) * 180/np.pi
sc_azimuth_rx = np.arctan2(scatter_loc_vis[1] - location_rx[1], scatter_loc_vis[0] - location_rx[0]) * 180/np.pi
ax4.axvline(sc_azimuth_tx, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Scatter from Tx: {sc_azimuth_tx:.1f}°')

ax4.set_xlabel('Azimuth Angle (degrees)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Normalized Gain (dB)', fontsize=10, fontweight='bold')
ax4.set_title('Optimal Beamformers - Azimuth Cut (El = 0°)', fontsize=11, fontweight='bold')
ax4.set_xlim([-180, 180])
ax4.set_ylim([-30, 5])
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Add text annotations
ax4.text(0.02, 0.98, f'N_tx = {N_tx}, N_rx = {N_rx}\nτ = {tau}, SNR = {snr_const[0]} dB', 
         transform=ax4.transAxes, fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
# fig_filename = os.path.join(drive_save_path, f'beam_pattern_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}.png')
# plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
# print(f"Beam pattern figure saved to {fig_filename}")

plt.show()

# %%
