"""
Mono-Static MIMO with BD SINR Maximization Objective

There exist other static scatters. 
BD alters its states to distinguish itself from other reflectors.

Objective: max_v SINR_BD = P|v^H H_b v|^2 / (P|v^H (H_d+H_r) v|^2 + noise_var)
w = v

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
import queue
import threading

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
from manifold_optimization import solve_x_equals_y_fast
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

# Set random seeds for reproducibility
seed = getattr(args, 'seed', 42)
np.random.seed(seed)
tf.set_random_seed(seed)
print(f"Random seed set to: {seed}")


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

drive_save_path = 'Mo_mimo_sinr_1b'
os.makedirs(drive_save_path, exist_ok=True)

# System parameters
fc = args.fc
Wavelength = 3e8 / fc


# Channel parameters
noiseSTD_per_dim = np.sqrt(0.5)
noise_var = 2 * noiseSTD_per_dim**2  # Total noise variance

# Tx/Rx configuration
location_tx = np.array([0, 0, 0])                     # Tx center location 
location_rx = np.array([0, 0, 0])                    # Tx and Rx are co-located
    # Antenna array, square array 
N_tx = args.N_ris 
N_rx = args.N_ris
N_ris = N_rx       # Numbers of antennas are all the same
num_users = 1
num_scatters = args.N_scatterers  # excluding BD
Rician_factor = args.rician_factor
location_bd = None

# Sensing parameters
tau = args.tau  # Pilot length (also number of BD interactions)
K = getattr(args, "N_symbols", 1)  # Number of OFDM symbols per BD state
snr_const = args.snr
snr_const = np.array([snr_const])
ref_dis = 15*Wavelength
Pvec = 10 ** (snr_const / 10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4)  / N_tx / N_rx

# BD modulation - alternating pattern
BD_modulation = np.array([(-1) ** t for t in range(tau)])
print(f"BD modulation pattern: {BD_modulation[:10]}...")

#####################################################
# Optimized Data Generation Helpers (Performance Optimization)
#####################################################

def _generate_single_sample(loc_tx, loc_rx, n_scatters, n_tx, n_rx, rician):
    """Generate a single channel sample"""
    # Generate random locations
    bd_loc = generate_location_mimo(1, 'u')[0]
    scatter_loc = generate_location_mimo(n_scatters, 's')
    
    # Generate MIMO channels
    _, H_d, H_r, H_b = generate_mimo_channel(
        loc_tx, loc_rx, scatter_loc, bd_loc,
        N_tx_h=n_tx, N_tx_v=1,
        N_rx_h=n_rx, N_rx_v=1,
        Rician_factor=rician
    )
    
    # Compute location info
    d_bd_rx = np.linalg.norm(bd_loc - loc_rx)
    d_bd_tx = np.linalg.norm(bd_loc - loc_tx)
    azimuth_bd = np.arctan2(bd_loc[1] - loc_rx[1], bd_loc[0] - loc_rx[0])
    loc_info = np.array([azimuth_bd, d_bd_rx, d_bd_tx])[:, np.newaxis]
    
    return H_d, H_b, H_r, loc_info, bd_loc, scatter_loc

def generate_batch_parallel(num_samples, loc_tx, loc_rx, n_scatters, n_tx, n_rx, rician):
    """Generate a batch of channel data - optimized sequential (ThreadPool adds overhead for fast ops)"""
    H_d_list = []
    H_b_list = []
    H_r_list = []
    loc_list = []
    bd_locs = []
    scatter_locs = []
    
    for _ in range(num_samples):
        H_d, H_b, H_r, loc_info, bd_loc, scatter_loc = _generate_single_sample(
            loc_tx, loc_rx, n_scatters, n_tx, n_rx, rician
        )
        H_d_list.append(H_d)
        H_b_list.append(H_b)
        H_r_list.append(H_r)
        loc_list.append(loc_info)
        bd_locs.append(bd_loc)
        scatter_locs.append(scatter_loc)
    
    return (np.array(H_d_list), np.array(H_b_list), np.array(H_r_list), 
            np.array(loc_list), bd_locs, scatter_locs)

def compute_optimal_beamformers_batch_parallel(H_b_batch, H_int_batch, noise_var_val, P_val, num_restarts=10):
    """Compute optimal beamformers for a batch - reduced restarts for 2x speedup"""
    batch_size = H_b_batch.shape[0]
    
    w_opt_batch = np.zeros((batch_size, H_b_batch.shape[2], 1), dtype=np.complex64)
    
    for i in range(batch_size):
        H_b_i = H_b_batch[i]
        H_int_i = H_int_batch[i]
        A = np.sqrt(P_val) * H_b_i
        B = np.sqrt(P_val) * H_int_i
        try:
            # Reduced restarts from 10 to 5 for ~2x speedup with minimal quality loss
            w_opt_i, _ = solve_x_equals_y_fast(A, B, c=noise_var_val, restarts=num_restarts)

            w_opt_batch[i, :, 0] = w_opt_i
        except:
            # Fallback to SVD
            U, S, Vh = np.linalg.svd(H_b_i)

            w_opt_batch[i, :, 0] = Vh[0, :]
    
    return w_opt_batch.astype(np.complex64)

# Background data generation queue (unused - ThreadPool overhead not worth it for fast ops)
class BackgroundDataGenerator:
    """Pre-generates batches in background thread for smoother training"""
    def __init__(self, num_samples, loc_tx, loc_rx, n_scatters, n_tx, n_rx, rician, queue_size=3):
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.num_samples = num_samples
        self.loc_tx = loc_tx
        self.loc_rx = loc_rx
        self.n_scatters = n_scatters
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.rician = rician
        self.thread = threading.Thread(target=self._generate_loop, daemon=True)
        
    def start(self):
        self.thread.start()
        
    def _generate_loop(self):
        while not self.stopped:
            try:
                batch = generate_batch_parallel(
                    self.num_samples, self.loc_tx, self.loc_rx,
                    self.n_scatters, self.n_tx, self.n_rx, self.rician
                )
                self.queue.put(batch, timeout=1)
            except queue.Full:
                continue
            except Exception as e:
                print(f"Background generation error: {e}")
                
    def get_batch(self, timeout=30):
        return self.queue.get(timeout=timeout)
    
    def stop(self):
        self.stopped = True

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

# For LS_based channel estimation and beam sweeping
def dft_codebook(n, m):
    """Create a unit-norm DFT codebook of shape (n, m)."""
    dft_matrix = np.ones((n, m), dtype=np.complex64)
    for i in range(m):
        dft_matrix[:, i] = (1 / np.sqrt(n)) * np.array(
            [np.exp(-1j * 2 * np.pi * j * i / m) for j in range(n)],
            dtype=np.complex64
        )
    return dft_matrix


pilot_codebook = dft_codebook(N_tx, max(N_tx, tau))
s_pilot = pilot_codebook[:, :tau].copy()

# Beam-sweeping codebook: ensure exactly 2*tau beams are available.
sweep_codebook = dft_codebook(N_tx, max(N_tx, 2 * tau))[:, : (2 * tau)].copy()
    
#####################################################
# Learning parameters and Computation Graph
#####################################################
#  
initial_run = 1 if args.n_epochs > 0 else 0
n_epochs = args.n_epochs
learning_rate = 5e-4
batch_per_epoch = 128
batch_size_order = 4
val_size_order = 20
test_size = 800

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Placeholders
loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_users), name="loc_input")
H_d_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_d")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_b")
H_r_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_r")

##################### ACTIVE SENSING NETWORK #####################

with tf.name_scope("system_parameters"):
    lay = {}
    lay['P'] = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(), name="tx_power")
    bd_seq = tf.constant(BD_modulation.astype(np.float32), dtype=tf.float32)

with tf.name_scope("active_sensing_agent"):
    hidden_size = 128  # Shared hidden size for both nodes
    
    LSTM1 = LSTM_Cell(hidden_size, name='LSTM_1')
    LSTM2 = LSTM_Cell(hidden_size, name='LSTM_2')
    
    # Tx and Rx share the same beamformer
    mlp_tx_rx = MLPBlock(3, [hidden_size * 2, hidden_size * 2, 2 * N_rx], name='Receiver_receiver')
    
    # SNR feature
    snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
    snr_dB = 10 * tf.log(snr) / np.log(10)
    snr_normal = snr_dB
    
    # BD modulation sequence for each time step
    x_BD = [tf.tile(tf.reshape(bd_seq[t], [1, 1]), [tf.shape(loc_input)[0], 1]) for t in range(tau)]
    
    # Storage
    v_list = []
    Y1 = tf.zeros([tf.shape(loc_input)[0], N_rx], dtype=tf.complex64) 
    Y2 = tf.zeros([tf.shape(loc_input)[0], N_rx], dtype=tf.complex64)

    batch_size = tf.shape(loc_input)[0]
    
    for t in range(tau):
        'Initialization at t=0'
        if t == 0:
            # Initialize shared LSTM states (both nodes see the same state)
            h_old = tf.zeros([batch_size, hidden_size])
            c_old = tf.zeros([batch_size, hidden_size])
            h_old2 = tf.zeros([batch_size, hidden_size])
            c_old2 = tf.zeros([batch_size, hidden_size])
            
            # Initialize first beamformer
            v_init_real = tf.get_variable("v_init_real", shape=(1, N_rx, 1), trainable=True)
            v_init_imag = tf.get_variable("v_init_imag", shape=(1, N_rx, 1), trainable=True)
            v_complex_init = tf.complex(v_init_real, v_init_imag)
            v1 = v_complex_init / tf.cast(tf.norm(v_complex_init, axis=1, keepdims=True), tf.complex64)
        
        'Construct effective channel H(t) = x_BD[t] * H_b + H_d + H_r'
        x_bd_t = tf.reshape(tf.cast(x_BD[t], tf.complex64), [-1, 1, 1])  # (batch, 1, 1)
        H_eff = x_bd_t * H_b_placeholder + H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
        H_eff2 = -x_bd_t * H_b_placeholder + H_d_placeholder + H_r_placeholder
        
        y_noiseless1 = tf.complex(tf.sqrt(lay['P']), 0.0) *  tf.matmul(H_eff, v1)  # (batch, N_rx, 1)
        # Broadcast to K samples
        y_noiseless1 = tf.tile(y_noiseless1, [1, 1, K])  # (batch, N_rx, K)
        y_noiseless2 = tf.tile(tf.complex(tf.sqrt(lay['P']), 0.0) *  tf.matmul(H_eff2, v1), [1, 1, K])  # (batch, N_rx, K)
        
        noise = tf.complex(
            tf.random_normal([2, batch_size, N_rx, K], mean=0.0, stddev=noiseSTD_per_dim),
            tf.random_normal([2, batch_size, N_rx, K], mean=0.0, stddev=noiseSTD_per_dim)
        )

        y_complex1 = tf.add(y_noiseless1, noise[0])  # (batch, N_rx, K) Before beamforming
        y_complex2 = tf.add(y_noiseless2, noise[1])  # (batch, N_rx, K) Before beamforming
        

        Y1 =  tf.reduce_mean(x_bd_t * y_complex1 - x_bd_t * y_complex2, axis=2, keepdims=False)
        Y2 =  tf.reduce_mean(y_complex1 + y_complex2, axis=2, keepdims=False)  # Accumulate over time steps
        Y1_after = tf.reduce_mean(tf.matmul(tf.linalg.adjoint(v1), tf.reshape(Y1, [-1, N_rx, 1])), axis=2, keepdims=False)
        Y2_after = tf.reduce_mean(tf.matmul(tf.linalg.adjoint(v1), tf.reshape(Y2, [-1, N_rx, 1])), axis=2, keepdims=False)
        y_real = tf.concat([
            tf.cast(tf.real(Y1), tf.float32),
            tf.cast(tf.imag(Y1), tf.float32),
            tf.cast(tf.real(Y1_after), tf.float32),
            tf.cast(tf.imag(Y1_after), tf.float32),
        ], axis=1)  # (batch, 4*N_rx + 4)

        y_real2 = tf.concat([
            tf.cast(tf.real(Y2), tf.float32),
            tf.cast(tf.imag(Y2), tf.float32),
            tf.cast(tf.real(Y2_after), tf.float32),
            tf.cast(tf.imag(Y2_after), tf.float32),
        ], axis=1)
        
        'Update shared LSTM state - both RIS and Rx can see the result'
        h_old, c_old = LSTM1((tf.concat([y_real,  snr_normal], axis=1), h_old, c_old))
        h_old2, c_old2 = LSTM2((tf.concat([y_real2,  snr_normal], axis=1), h_old2, c_old2))
        
        'Rx designs receive beamformer v based on shared hidden state'
        v_her = mlp_tx_rx(tf.concat([h_old, h_old2], axis=1))
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm + 1e-8)
        v1 = tf.complex(v_her[:, 0:N_rx], v_her[:, N_rx:2 * N_rx])
        v1 = tf.reshape(v1, [-1, N_rx, 1])
        
        # Store beamformers for analysis
        v_list.append(tf.concat([tf.real(tf.squeeze(v1, axis=2)), tf.imag(tf.squeeze(v1, axis=2))], axis=1))
    
    #####################################################
    # Output Final Beamformers after tau interactions
    # Both RIS and Rx use the shared LSTM state c_old
    #####################################################
    MLP_bf_v = MLPBlock(3, [2 * hidden_size, 2 * hidden_size, 2 * N_rx], name='MLP_bf_v')
    
    # Final eamformer v (from shared state)
    v_tmp = MLP_bf_v(tf.concat([c_old, c_old2], axis=1))
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
    sig_BD = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_b_placeholder, v_complex))  # (batch, 1, 1)
    sig_BD = tf.squeeze(tf.abs(sig_BD) ** 2) * lay['P']  # (batch,)
    
    # Interference channel: H_d + H_r
    H_interference = H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
    
    # Interference power: P * |v^H (H_d + H_r) w|^2
    sig_int = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_interference, v_complex))  # (batch, 1, 1)
    sig_int = tf.squeeze(tf.abs(sig_int) ** 2) * lay['P']  # (batch,)
    
    sinr_BD = sig_BD / (sig_int + noise_var + 1e-10)
    sinr_BD_clipped = tf.clip_by_value(sinr_BD, 1e-4, 1e4)
    
    # Log SINR with per-sample clipping to prevent outliers from dominating
    log_sinr_BD_raw = tf.log(sinr_BD_clipped + 1e-9)
    log_sinr_BD = tf.clip_by_value(log_sinr_BD_raw, -8.0, 8.0)  # ~±35 dB range
    
    # For backward compatibility, define sig_ref as interference
    sig_ref = sig_int

    # Calculate the SINR towards the scatters
    sig_scatter = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_r_placeholder, v_complex))  # (batch, 1, 1)
    int_scatter = tf.matmul(tf.linalg.adjoint(v_complex), \
                            tf.matmul(H_d_placeholder+H_b_placeholder, v_complex))  # (batch, 1, 1)
    sinr_scatter = tf.squeeze(tf.abs(sig_scatter) ** 2) * lay['P'] / \
                    (tf.squeeze(tf.abs(int_scatter) ** 2) * lay['P'] + noise_var + 1e-10)


#####################################################
# Optimal Beamformer (SVD-based solution for MIMO)
#####################################################

with tf.name_scope("optimal_beamformer"):
    batch_size_opt = tf.shape(H_b_placeholder)[0]
    
    P_c = tf.cast(lay['P'], tf.complex64)
    N0_c = tf.cast(noise_var, tf.complex64)
    
    # Use tf.py_func to call NumPy-based optimization (with parallel processing)
    def compute_optimal_beamformers_np(H_b, H_int, noise_var_val, P_val):
        """Compute optimal beamformers for a batch using parallel manifold optimization"""
        return compute_optimal_beamformers_batch_parallel(
            H_b, H_int, noise_var_val, P_val, 
            num_restarts=10
        )
    
    # Wrap in tf.py_func
    v_opt = tf.py_func(
        lambda H_b, H_int: compute_optimal_beamformers_np(H_b, H_int, noise_var, Pvec[0]),
        [H_b_placeholder, H_interference],
           tf.complex64
    )
    
    # Set shapes explicitly
    v_opt = tf.reshape(v_opt, [-1, N_rx, 1])

    # Optimal SINR with SVD-based beamformers
    sig_BD_opt = tf.matmul(tf.linalg.adjoint(v_opt), tf.matmul(H_b_placeholder, v_opt))
    sig_BD_opt = tf.squeeze(tf.abs(sig_BD_opt) ** 2) * lay['P']
    
    sig_int_opt = tf.matmul(tf.linalg.adjoint(v_opt), tf.matmul(H_interference, v_opt))
    sig_int_opt = tf.squeeze(tf.abs(sig_int_opt) ** 2) * lay['P']
    
    sinr_BD_opt = sig_BD_opt / (sig_int_opt + noise_var + 1e-10)

    scatter_sig_opt = tf.matmul(tf.linalg.adjoint(v_opt), tf.matmul(H_r_placeholder, v_opt))  # (batch, 1, 1)
    scatter_sig_opt = tf.squeeze(tf.abs(scatter_sig_opt) ** 2) * lay['P']
    scatter_int_opt = tf.matmul(tf.linalg.adjoint(v_opt), \
                            tf.matmul(H_d_placeholder+H_b_placeholder, v_opt))  # (batch, 1, 1)
    scatter_int_opt = tf.squeeze(tf.abs(scatter_int_opt) ** 2) * lay['P']
    sinr_scatter_opt = scatter_sig_opt / (scatter_int_opt + noise_var + 1e-10)
    
with tf.name_scope("sp_beamformer"):
    # Signal processing baseline from noisy pilot measurements.
    # System model: Y = sqrt(P) * H * S + N
    s_pilot_tf = tf.constant(s_pilot, dtype=tf.complex64)
    s_pilot_h = tf.linalg.adjoint(s_pilot_tf)  # (tau, N_tx)
    ss_h = tf.matmul(s_pilot_tf, s_pilot_h)  # (N_tx, N_tx) = SS^H
    sqrt_p = tf.complex(tf.sqrt(lay['P']), 0.0)
    
    reg = tf.cast(1e-1, tf.complex64) * tf.eye(N_tx, dtype=tf.complex64)
    ss_h_reg = ss_h + reg
    s_pilot_pinv = tf.linalg.adjoint(tf.linalg.solve(ss_h_reg, s_pilot_tf))  # (tau, N_tx)

    H_eff0 = (-1) * H_b_placeholder + H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
    Y_0_clean = tf.matmul(H_eff0, s_pilot_tf)  # (batch, N_rx, tau)
    noise_0 = tf.complex(
        tf.random_normal(tf.shape(Y_0_clean), mean=0.0, stddev=noiseSTD_per_dim),
        tf.random_normal(tf.shape(Y_0_clean), mean=0.0, stddev=noiseSTD_per_dim)
    )
    Y_0 = sqrt_p * Y_0_clean + noise_0
    H_eff_hat0 = tf.matmul(Y_0, s_pilot_pinv) / (sqrt_p + 1e-10)

    H_eff1 = (1) * H_b_placeholder + H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
    Y_1_clean = tf.matmul(H_eff1, s_pilot_tf)  # (batch, N_rx, tau)
    noise_1 = tf.complex(
        tf.random_normal(tf.shape(Y_1_clean), mean=0.0, stddev=noiseSTD_per_dim),
        tf.random_normal(tf.shape(Y_1_clean), mean=0.0, stddev=noiseSTD_per_dim)
    )
    Y_1 = sqrt_p * Y_1_clean + noise_1
    H_eff_hat1 = tf.matmul(Y_1, s_pilot_pinv) / (sqrt_p + 1e-10)  # (batch, N_rx, N_tx)

    H_int_estimated = (H_eff_hat0 + H_eff_hat1) / tf.cast(2, tf.complex64)
    H_bd_estimated = (H_eff_hat1 - H_eff_hat0) / tf.cast(2, tf.complex64)

    v_sp = tf.py_func(
        lambda H_b, H_int: compute_optimal_beamformers_np(H_b, H_int, noise_var, Pvec[0]),
        [H_bd_estimated, H_int_estimated],
           tf.complex64
    )
    
    # Set shapes explicitly
    v_sp = tf.reshape(v_sp, [-1, N_rx, 1])
    
    # # Compute SINR with signal processing beamformers
    sig_BD_sp = tf.matmul(tf.linalg.adjoint(v_sp), tf.matmul(H_b_placeholder, v_sp))
    sig_BD_sp = tf.squeeze(tf.abs(sig_BD_sp) ** 2) * lay['P']
    
    sig_int_sp = tf.matmul(tf.linalg.adjoint(v_sp), tf.matmul(H_interference, v_sp))
    sig_int_sp = tf.squeeze(tf.abs(sig_int_sp) ** 2) * lay['P']
    
    sinr_BD_sp = sig_BD_sp / (sig_int_sp + noise_var + 1e-10)

    #### Beam sweeping baseline (selection from estimated channels; evaluation on true channels).
    sweep_codebook_tf = tf.constant(sweep_codebook, dtype=tf.complex64)
    
    # Beam sweeping from estimated channels
    Hbd_est_W = tf.matmul(H_bd_estimated, sweep_codebook_tf)  # (batch, N_rx, num_beams)
    codebook_H = tf.linalg.adjoint(sweep_codebook_tf)         # (num_beams, N_tx), N_tx == N_rx
    codebook_H_expanded = tf.expand_dims(codebook_H, 0)       # (1, num_beams, N_rx)
    bi_matrix = tf.abs(tf.matmul(codebook_H_expanded, Hbd_est_W)) ** 2  # (batch, num_beams, num_beams)
    sweep_metric = tf.linalg.diag_part(bi_matrix)  # (batch, num_beams)
    best_v_idx = tf.argmax(sweep_metric, axis=1, output_type=tf.int32)

    # Gather selected Tx beam
    codebook_T = tf.transpose(sweep_codebook_tf)  # (num_beams, N_tx)
    v_sweep = tf.gather(codebook_T, best_v_idx)  # (batch, N_tx)
    v_sweep = tf.reshape(v_sweep, [-1, N_tx, 1])
    v_sweep = v_sweep / tf.cast(tf.norm(v_sweep, axis=1, keepdims=True) + 1e-10, tf.complex64)

    
    # Compute final SINR using selected beams on TRUE channels
    sig_BD_sweep = tf.matmul(tf.linalg.adjoint(v_sweep), tf.matmul(H_b_placeholder, v_sweep))
    sig_BD_sweep = tf.squeeze(tf.abs(sig_BD_sweep) ** 2) * lay['P']
    
    sig_int_sweep = tf.matmul(tf.linalg.adjoint(v_sweep), tf.matmul(H_interference, v_sweep))
    sig_int_sweep = tf.squeeze(tf.abs(sig_int_sweep) ** 2) * lay['P']
    
    sinr_BD_sweep = sig_BD_sweep / (sig_int_sweep + noise_var + 1e-10)


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
    tf.check_numerics(log_sinr_BD, 'log_sinr_BD NaN')
]

with tf.control_dependencies(nan_checks):
    training_op = optimizer.apply_gradients(list(zip(final_grads, vars_list)), global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


#####################################################
# Data Generation
#####################################################

# Validation set (generated in parallel for speed)
num_val_samples = val_size_order * 32


H_d_val, H_b_val, H_r_val, set_location_user_val, _, _ = generate_batch_parallel(
    num_val_samples, location_tx, location_rx, num_scatters, N_tx, N_rx, Rician_factor
)


feed_dict_val = {
    loc_input: set_location_user_val,
    lay['P']: Pvec[0],
    H_d_placeholder: H_d_val,
    H_b_placeholder: H_b_val,
    H_r_placeholder: H_r_val
}


#####################################################
# Training Loop
#####################################################

print("\n" + "=" * 60)
print("BD SINR Maximization via Active Sensing")
print("=" * 60)
print(f"N_tx: {N_tx}, N_rx: {N_rx}, tau: {tau}, K: {K}, SNR: {snr_const[0]} dB")
print("=" * 60 + "\n")

model_ckpt = f'{drive_save_path}/params_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}_K_{K}'

with tf.Session() as sess:
    if initial_run == 1:
        init.run()
        best_val = np.inf
    else:
        saver.restore(sess, model_ckpt)
        best_val = float(sess.run(loss, feed_dict=feed_dict_val))
        print(f"Restored validation loss baseline: {best_val:.6f}")
    
    # Early stopping
    wait = 0
    PATIENCE = 20
    
    for epoch in range(n_epochs):
        batch_iter = 0
        epoch_train_losses = []
        epoch_sinr_values = []
        
        for rnd_indices in range(batch_per_epoch):
            # Generate training batch using parallel processing
            num_train_samples = batch_size_order * 32
            
            # Use parallel batch generation (much faster than sequential)
            H_d_train, H_b_train, H_r_train, set_location_user_train, _, _ = generate_batch_parallel(
                num_train_samples, location_tx, location_rx, num_scatters, N_tx, N_rx, Rician_factor
            )

            feed_dict_batch = {
                loc_input: set_location_user_train,
                lay['P']: Pvec[0],
                H_d_placeholder: H_d_train,
                H_b_placeholder: H_b_train,
                H_r_placeholder: H_r_train
            }
            
            _, train_loss, sinr_values, sinr_scatter_values = sess.run(
                [training_op, loss, sinr_BD, sinr_scatter], feed_dict=feed_dict_batch
            )
            
            epoch_train_losses.append(train_loss)
            epoch_sinr_values.append(np.mean(sinr_values))
            batch_iter += 1
        
        # Validation
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_sinr = np.mean(epoch_sinr_values)
        
        loss_val, sinr_val, sinr_scatter_val, sig_bd_val, sig_ref_val = sess.run(
            [loss, sinr_BD, sinr_scatter, sig_BD, sig_ref], feed_dict=feed_dict_val
        )

        # Optimal beamformer performance
        if epoch == 0:
            sinr_opt_val, sinr_scatter_opt_val, sig_bd_opt_val, sig_int_opt_val = sess.run(
            [sinr_BD_opt, sinr_scatter_opt, sig_BD_opt, sig_int_opt], feed_dict=feed_dict_val)

            # sp-based beamformer performance
            # sinr_sp_val, sig_bd_sp_val, sig_int_sp_val = sess.run(
            #             [sinr_BD_sp, sig_BD_sp, sig_int_sp], feed_dict=feed_dict_val)

        
        print(f'Epoch {epoch:3d} | '
              f'Train Loss: {avg_train_loss:8.4f} | '
              f'Val Loss: {loss_val:8.4f} | '
              f'Best: {best_val:8.4f}')
        # print(f'         | '
        #       f'SINR_BD (sp): {10 * np.log10(np.mean(sinr_sp_val) + 1e-10):6.2f} dB | ')
        print(f'         | '
              f'SINR_BD (learned): {10 * np.log10(np.mean(sinr_val) + 1e-10):6.2f} dB | '
              f'SINR_BD (optimal): {10 * np.log10(np.mean(sinr_opt_val) + 1e-10):6.2f} dB')
        print(f'         | '
              f'Sig_BD: {np.mean(sig_bd_val):8.4f} | '
              f'Sig_int: {np.mean(sig_ref_val):8.4f} | '
              f'Sig_BD_opt: {np.mean(sig_bd_opt_val):8.4f} | '
              f'Sig_int_opt: {np.mean(sig_int_opt_val):8.4f} | ')
        print(f'         | '
              f'SINR_scatter: {10 * np.log10(np.mean(sinr_scatter_val) + 1e-10):6.2f} dB | '
              f'SINR_scatter_opt: {10 * np.log10(np.mean(sinr_scatter_opt_val) + 1e-10):6.2f} dB')
        print()
        
        # Early stopping
        if loss_val < best_val - 1e-8:
            best_val = loss_val
            wait = 0
            saver.save(sess, model_ckpt)
            with open(os.path.join(drive_save_path, "best_val_loss.txt"), "w") as f:
                f.write(str(best_val))
        else:
            wait += 1
            if wait == PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test the best validation model, not the weights left in memory after
    # the final training epoch.
    print(f"Restoring best validation checkpoint for testing: {model_ckpt}")
    saver.restore(sess, model_ckpt)
    
    #####################################################
    # Testing
    #####################################################
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)
    
    # Reset seed before generating test data to ensure reproducibility across runs
    # This guarantees the same test set regardless of training parameters (tau, epochs, etc.)
    test_seed = seed + 10000  # Offset to separate from training data
    np.random.seed(test_seed)
    print(f"Test data seed: {test_seed}")
    
    # Generate test batch using parallel processing

    
    H_d_test, H_b_test, H_r_test, set_location_user_test, BD_loc, Scatter_loc = generate_batch_parallel(
        test_size, location_tx, location_rx, num_scatters, N_tx, N_rx, Rician_factor
    )
    

    feed_dict_test = {
        loc_input: set_location_user_test,
        lay['P']: Pvec[0],
        H_d_placeholder: H_d_test,
        H_b_placeholder: H_b_test,
        H_r_placeholder: H_r_test
    }
    
    sinr_test, sinr_opt_test, sinr_scatter_test, v_learned, v_optimal = sess.run(
        [sinr_BD, sinr_BD_opt, sinr_scatter, v_complex,  v_opt], feed_dict=feed_dict_test
    )

    sinr_sp_test, sinr_sweep_test = sess.run([sinr_BD_sp, sinr_BD_sweep], feed_dict=feed_dict_test)
    
    print(f"Test SINR_BD (learned):  {10 * np.log10(np.mean(sinr_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_BD (optimal):  {10 * np.log10(np.mean(sinr_opt_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_scatter (learned):       {10 * np.log10(np.mean(sinr_scatter_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_BD (signalprocessing): {10 * np.log10(np.mean(sinr_sp_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_BD (sweeping): {10 * np.log10(np.mean(sinr_sweep_test) + 1e-10):6.2f} dB")
    print(f"Gap to optimal:          {10 * np.log10((np.mean(sinr_opt_test) + 1e-10) / (np.mean(sinr_test) + 1e-10)):6.2f} dB")
    
    # Save results
    model_filename = os.path.join(drive_save_path, \
            f'TEST_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}_K_{K}_Nsca_{num_scatters}.mat')
    sio.savemat(model_filename, dict(
        snr_const=snr_const,
        N_tx=N_tx,
        N_rx=N_rx,
        tau=tau,
        K=K,
        BD_location=BD_loc,
        Scatter_location=Scatter_loc if Scatter_loc[0] is not None else [],
        sinr_learned=sinr_test,
        sinr_optimal=sinr_opt_test,
        sinr_sp=sinr_sp_test,
        sinr_sweep=sinr_sweep_test,
        sinr_scatter_learned=sinr_scatter_test,
        v_learned=v_learned,
        v_optimal=v_optimal,
    ))
    print(f"\nResults saved to {model_filename}")




# %%    
    #####################################################
    # 2D Beam Pattern Visualization (Azimuth only, Elevation = 0)
    #####################################################
print("\n" + "=" * 60)
print("Beam Pattern Visualization (2D - Azimuth)")
print("=" * 60)

# Allow running this section alone (e.g., in an IDE/Jupyter cell).
if 'np' not in globals():
    import numpy as np
if 'plt' not in globals():
    import matplotlib.pyplot as plt
if 'sio' not in globals():
    import scipy.io as sio
if 'os' not in globals():
    import os

N_tx = int(globals().get('N_tx', 36))
N_rx = int(globals().get('N_rx', 36))
tau = int(globals().get('tau', 7))
K = int(globals().get('K', 1))
num_scatters = int(globals().get('num_scatters', 3))
snr_const = np.atleast_1d(np.squeeze(np.asarray(globals().get('snr_const', np.array([10.0], dtype=np.float64)))))

# This section can run standalone by loading a saved .mat result file.
def _decode_saved_locations(raw):
    arr = np.asarray(raw)
    if arr.size == 0:
        return []
    if arr.dtype != object:
        arr = np.squeeze(arr)
        # Common numeric formats from savemat:
        # (num_realizations, 3) for single point per realization
        # (num_realizations, num_scatterers, 3) for multiple points per realization
        if arr.ndim == 2 and arr.shape[-1] == 3:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 2 and arr.shape[0] == 3:
            return [arr[:, i] for i in range(arr.shape[1])]
        return [np.squeeze(arr)]

    out = []
    for elem in arr.ravel():
        if elem is None:
            out.append(None)
            continue
        val = np.squeeze(np.asarray(elem))
        out.append(None if val.size == 0 else val)

    # Some object-array saves wrap all realizations in a single element:
    # [ (num_realizations, num_scatterers, 3) ].
    if len(out) == 1 and out[0] is not None:
        first = np.squeeze(np.asarray(out[0]))
        if first.ndim == 3 and first.shape[-1] == 3:
            return [first[i] for i in range(first.shape[0])]
        if first.ndim == 2 and first.shape[-1] == 3:
            return [first[i] for i in range(first.shape[0])]
        if first.ndim == 2 and first.shape[0] == 3 and first.shape[1] != 3:
            return [first[:, i] for i in range(first.shape[1])]
    return out

def _pick_result_file(result_dir):
    override_file = globals().get('visualization_result_file', None)
    if override_file is not None and os.path.isfile(override_file):
        return override_file

    candidate = None
    snr_arr = np.atleast_1d(np.squeeze(np.asarray(globals()['snr_const'])))
    snr0 = int(snr_arr[0])
    candidate = os.path.join(
        result_dir,
        f"TEST_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{snr0}_K_{K}_Nsca_{num_scatters}.mat"
    )
    if not os.path.isfile(candidate):
        candidate = [
            os.path.join(result_dir, f) for f in os.listdir(result_dir)
            if f.startswith('TEST_sinr_') and f.endswith('.mat')
        ]
        if not candidate:
            raise FileNotFoundError(
                f"No saved TEST_sinr*.mat files found in {result_dir}. "
                "Run testing once or set visualization_result_file."
            )
        candidate.sort(key=os.path.getmtime)
        candidate = candidate[-1]
    return candidate

required_vis_vars = [
    'v_learned', 'v_optimal',
    'sinr_test', 'sinr_opt_test', 'sinr_sp_test', 'sinr_sweep_test',
    'BD_loc', 'Scatter_loc'
]
force_reload = bool(globals().get('force_load_visualization_data', False))
if force_reload or any(name not in globals() for name in required_vis_vars):
    result_dir = globals().get('drive_save_path', 'Mo_mimo_sinr_1b')
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Result directory not found: {result_dir}")
    result_file = _pick_result_file(result_dir)
    loaded = sio.loadmat(result_file)
    print(f"Loaded visualization data from: {result_file}")

    if 'N_tx' in loaded:
        N_tx = int(np.squeeze(loaded['N_tx']))
    if 'N_rx' in loaded:
        N_rx = int(np.squeeze(loaded['N_rx']))
    if 'tau' in loaded:
        tau = int(np.squeeze(loaded['tau']))
    if 'snr_const' in loaded:
        snr_const = np.atleast_1d(np.squeeze(loaded['snr_const']))

    v_learned = loaded['v_learned']
    v_optimal = loaded['v_optimal']

    sinr_test = np.squeeze(loaded.get('sinr_learned', np.array([np.nan])))
    sinr_opt_test = np.squeeze(loaded.get('sinr_optimal', np.array([np.nan])))
    sinr_sp_test = np.squeeze(loaded.get('sinr_sp', np.full_like(np.atleast_1d(sinr_test), np.nan, dtype=np.float64)))
    sinr_sweep_test = np.squeeze(loaded.get('sinr_sweep', np.full_like(np.atleast_1d(sinr_test), np.nan, dtype=np.float64)))

    BD_loc = _decode_saved_locations(loaded.get('BD_location', np.array([])))
    Scatter_loc = _decode_saved_locations(loaded.get('Scatter_location', np.array([])))

if 'location_tx' not in globals():
    location_tx = np.array([0, 0, 0])
if 'snr_const' not in globals():
    snr_const = np.array([np.nan], dtype=np.float64)
snr_display = np.atleast_1d(np.squeeze(np.asarray(snr_const)))[0]


v_learned = np.asarray(v_learned)
v_optimal = np.asarray(v_optimal)


if v_learned.ndim == 2:
    v_learned = v_learned[np.newaxis, ...]

if v_optimal.ndim == 2:
    v_optimal = v_optimal[np.newaxis, ...]

num_realizations = int(v_learned.shape[0])
if len(BD_loc) == 0:
    BD_loc = [np.array([0, 0, 0], dtype=np.float64) for _ in range(num_realizations)]
if len(Scatter_loc) == 0:
    Scatter_loc = [None for _ in range(num_realizations)]
if len(BD_loc) < num_realizations:
    BD_loc += [np.array([0, 0, 0], dtype=np.float64) for _ in range(num_realizations - len(BD_loc))]
if len(Scatter_loc) < num_realizations:
    Scatter_loc += [None for _ in range(num_realizations - len(Scatter_loc))]

# Select one test instance for visualization (random unless beam_pattern_idx is set)
idx_default = np.random.randint(num_realizations)
idx = int(globals().get('beam_pattern_idx', idx_default))
idx = max(0, min(idx, num_realizations - 1))
bd_loc_vis = np.squeeze(np.asarray(BD_loc[idx]))

# Handle multiple scatterers - ensure 2D array shape (num_scatterers, 3)
scatter_entry = Scatter_loc[idx] if idx < len(Scatter_loc) else None
if scatter_entry is None:
    scatter_loc_vis = np.zeros((0, 3), dtype=np.float64)
else:
    scatter_loc_vis = np.asarray(scatter_entry)
    scatter_loc_vis = np.squeeze(scatter_loc_vis)
    if scatter_loc_vis.ndim == 1:
        scatter_loc_vis = scatter_loc_vis[np.newaxis, :]
    if scatter_loc_vis.ndim == 2 and scatter_loc_vis.shape[0] == 3 and scatter_loc_vis.shape[1] != 3:
        scatter_loc_vis = scatter_loc_vis.T
    if scatter_loc_vis.ndim > 2 and scatter_loc_vis.size % 3 == 0:
        scatter_loc_vis = scatter_loc_vis.reshape(-1, 3)
    if scatter_loc_vis.shape[-1] != 3:
        scatter_loc_vis = np.zeros((0, 3), dtype=np.float64)
num_scatterers_vis = scatter_loc_vis.shape[0]

# Shared beamformer case: the same beam is used on transmit and receive (w = v).
N_shared_h = N_rx

def fold_ula_azimuth(angle_rad, sector_start=-np.pi / 2):
    """Fold azimuth angle(s) into ULA-equivalent sector [sector_start, sector_start + pi)."""
    return np.mod(angle_rad - sector_start, np.pi) + sector_start

# Function to compute 2D beam pattern (azimuth only, elevation = 0)
def compute_beam_pattern_2d(beamformer, N_h, num_points=360):
    """Compute 2D beam pattern for azimuth in [-pi/2, pi/2] (elevation = 0)
    
    For elevation = 0:
    - cos(el) = 1, sin(el) = 0
    - steering_vector = exp(1j * pi * i * sin(az))
    This simplifies to a ULA pattern in azimuth.
    """
    azimuth = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    
    pattern = np.zeros(num_points, dtype=np.float64)
    bf = beamformer.flatten()
    
    # For ULA-like behavior in azimuth (elevation = 0)
    indices = np.arange(len(bf))
    
    for i_az in range(num_points):
        az = azimuth[i_az]
        sin_azimuth = np.sin(az)
        
        # Steering vector for elevation = 0: a = exp(1j * pi * i * sin(az))
        a = np.exp(1j * np.pi * indices * sin_azimuth)
        pattern[i_az] = np.abs(np.dot(np.conj(bf), a)) ** 2
    
    # Normalize
    pattern = pattern / (np.max(pattern) + 1e-10)
    return azimuth, pattern

# Get learned and optimal beamformers for this instance
shared_beam_learned_vis = v_learned[idx]  # (N_rx, 1)
shared_beam_optimal_vis = v_optimal[idx]  # (N_rx, 1)

# Compute 2D beam patterns
az_shared_learned, pattern_shared_learned = compute_beam_pattern_2d(shared_beam_learned_vis, N_shared_h, num_points=360)
az_shared_optimal, pattern_shared_optimal = compute_beam_pattern_2d(shared_beam_optimal_vis, N_shared_h, num_points=360)

# Calculate target directions and fold to ULA-equivalent azimuth sector [-pi/2, pi/2)
bd_azimuth = fold_ula_azimuth(np.arctan2(bd_loc_vis[1] - location_tx[1], bd_loc_vis[0] - location_tx[0]))

# Calculate azimuth for each scatterer
scatter_azimuths = []
for s in range(num_scatterers_vis):
    scatter_az = np.arctan2(scatter_loc_vis[s, 1] - location_tx[1], scatter_loc_vis[s, 0] - location_tx[0])
    scatter_azimuths.append(scatter_az)
scatter_azimuths = fold_ula_azimuth(np.array(scatter_azimuths))

def configure_upper_half_polar_axis(ax):
    """Show -pi/2..pi/2 as an upward-facing semicircle with radian tick labels."""
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_xticks([-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2])
    ax.set_xticklabels([r'$-\pi/2$', r'$-\pi/4$', '0', r'$\pi/4$', r'$\pi/2$'])
    ax.set_ylim([0, 1])

# Create figure with 2 rows, 3 columns
fig = plt.figure(figsize=(18, 12))
# --- SINR summary (for the selected idx) ---
def _scalar_at(x, i):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    arr = np.ravel(arr)
    i = max(0, min(i, arr.size - 1))
    return float(np.squeeze(arr[i]))

opt_sinr_val = _scalar_at(sinr_opt_test, idx)
learned_sinr_val = _scalar_at(sinr_test, idx)
sp_sinr_val = _scalar_at(sinr_sp_test, idx)
sweep_sinr_val = _scalar_at(sinr_sweep_test, idx)

sinr_text = (
    f"Opt SINR: {10*np.log10(opt_sinr_val + 1e-12):.2f} dB   |   "
    f"Learned SINR: {10*np.log10(learned_sinr_val + 1e-12):.2f} dB   |   "
    f"SP SINR: {10*np.log10(sp_sinr_val + 1e-12):.2f} dB   |   "
    f"Sweep SINR: {10*np.log10(sweep_sinr_val + 1e-12):.2f} dB"
)

fig.text(
    0.5, 0.985, sinr_text,
    ha='center', va='top', fontsize=12,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='gray')
)
plt.subplots_adjust(top=0.90)
ax1 = fig.add_subplot(2, 3, 1)

# Plot Tx/Rx location (co-located)
ax1.scatter(location_tx[0], location_tx[1], c='blue', marker='s', s=200, label='Tx/Rx Array', zorder=10, edgecolors='black')

# Plot BD location
ax1.scatter(bd_loc_vis[0], bd_loc_vis[1], c='red', marker='*', s=400, label='BD', zorder=10, edgecolors='black', linewidth=1.5)

# Plot Scatter locations (multiple scatterers)
scatter_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, num_scatterers_vis))
for s in range(num_scatterers_vis):
    label = 'Scatterers' if s == 0 else None
    ax1.scatter(scatter_loc_vis[s, 0], scatter_loc_vis[s, 1], c=[scatter_colors[s]], marker='o', s=250, 
                label=label, zorder=10, edgecolors='black', linewidth=1.5)

# Draw lines showing signal paths
ax1.plot([location_tx[0], bd_loc_vis[0]], [location_tx[1], bd_loc_vis[1]], 'r--', alpha=0.6, linewidth=2, label='BD path')
for s in range(num_scatterers_vis):
    label = 'Scatter paths' if s == 0 else None
    ax1.plot([location_tx[0], scatter_loc_vis[s, 0]], [location_tx[1], scatter_loc_vis[s, 1]], 
             color=scatter_colors[s], linestyle=':', alpha=0.6, linewidth=2, label=label)

# Add direction arrows
dir_to_bd = (bd_loc_vis[:2] - location_tx[:2]) / np.linalg.norm(bd_loc_vis[:2] - location_tx[:2])
arrow_scale = 3
ax1.annotate('', xy=(location_tx[0] + dir_to_bd[0]*arrow_scale, location_tx[1] + dir_to_bd[1]*arrow_scale),
             xytext=(location_tx[0], location_tx[1]),
             arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
for s in range(num_scatterers_vis):
    dir_to_scatter = (scatter_loc_vis[s, :2] - location_tx[:2]) / (np.linalg.norm(scatter_loc_vis[s, :2] - location_tx[:2]) + 1e-10)
    ax1.annotate('', xy=(location_tx[0] + dir_to_scatter[0]*arrow_scale, location_tx[1] + dir_to_scatter[1]*arrow_scale),
                 xytext=(location_tx[0], location_tx[1]),
                 arrowprops=dict(arrowstyle='->', color=scatter_colors[s], lw=2))

ax1.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax1.set_title('2D Scene (XY Plane, Elevation = 0)', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# Build scatter azimuth text for info box
if num_scatterers_vis:
    scatter_az_text = '\n'.join([f'Scatter {s+1} Az: {np.degrees(scatter_azimuths[s]):.1f}°' for s in range(num_scatterers_vis)])
else:
    scatter_az_text = 'No scatterers'
ax1.text(0.02, 0.98, f'BD Az: {np.degrees(bd_azimuth):.1f}°\n{scatter_az_text}', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Utilities for comparison plots.
def pattern_to_db(pattern):
    return 10 * np.log10(np.asarray(pattern) + 1e-10)

def sample_pattern_at_angles(azimuth_grid, pattern, angles):
    sample_angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    return np.interp(sample_angles, azimuth_grid, pattern)

# ===== Row 1, Col 2: Polar Plot - Learned Shared Beam =====
ax2 = fig.add_subplot(2, 3, 2, projection='polar')

ax2.plot(az_shared_learned, pattern_shared_learned, color='navy', linewidth=2.5, label='Learned beam')
ax2.fill(az_shared_learned, pattern_shared_learned, color='steelblue', alpha=0.2)

ax2.axvline(bd_azimuth, color='red', linestyle='--', linewidth=2.5, label=f'BD: {np.degrees(bd_azimuth):.1f}°')
scatter_colors_polar = plt.cm.Oranges(np.linspace(0.5, 0.9, num_scatterers_vis))
for s in range(num_scatterers_vis):
    ax2.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0, 
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax2.set_title('Learned Shared Beamformer (w = v)', fontsize=12, fontweight='bold', pad=15)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax2)

# ===== Row 1, Col 3: Polar Plot - Optimal Shared Beam =====
ax3 = fig.add_subplot(2, 3, 3, projection='polar')

ax3.plot(az_shared_optimal, pattern_shared_optimal, color='black', linewidth=2.5, label='Optimal beam')
ax3.fill(az_shared_optimal, pattern_shared_optimal, color='gray', alpha=0.2)

ax3.axvline(bd_azimuth, color='red', linestyle='--', linewidth=2.5, label=f'BD: {np.degrees(bd_azimuth):.1f}°')
for s in range(num_scatterers_vis):
    ax3.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0, 
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax3.set_title('Optimal Shared Beamformer', fontsize=12, fontweight='bold', pad=15)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax3)

# ===== Row 2, Col 1: Cartesian Comparison Plot =====
ax4 = fig.add_subplot(2, 3, 4)

azimuth_deg = np.degrees(az_shared_learned)

ax4.plot(azimuth_deg, pattern_to_db(pattern_shared_learned), 'b-', linewidth=2.2, label='Learned', alpha=0.85)
ax4.plot(azimuth_deg, pattern_to_db(pattern_shared_optimal), 'k--', linewidth=2.2, label='Optimal', alpha=0.85)

bd_az_deg = np.degrees(bd_azimuth)
ax4.axvline(bd_az_deg, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'BD: {bd_az_deg:.1f}°')
scatter_colors_cart = plt.cm.Oranges(np.linspace(0.5, 0.9, num_scatterers_vis))
for s in range(num_scatterers_vis):
    scatter_az_deg = np.degrees(scatter_azimuths[s])
    ax4.axvline(scatter_az_deg, color=scatter_colors_cart[s], linestyle=':', linewidth=2, alpha=0.7, 
                label=f'S{s+1}: {scatter_az_deg:.1f}°')

ax4.set_xlabel('Azimuth Angle (degrees)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Normalized Gain (dB)', fontsize=11, fontweight='bold')
ax4.set_title('Shared Beam Pattern Comparison (Elevation = 0°)', fontsize=12, fontweight='bold')
ax4.set_xlim([-90, 90])
ax4.set_ylim([-30, 5])
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax4.text(0.02, 0.02, f'N_tx = {N_tx}, N_rx = {N_rx}\nτ = {tau}, SNR = {snr_display} dB', 
         transform=ax4.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ===== Row 2, Col 2: Polar Overlay - Learned vs Optimal =====
ax5 = fig.add_subplot(2, 3, 5, projection='polar')

ax5.plot(az_shared_learned, pattern_shared_learned, color='navy', linewidth=2.5, label='Learned')
ax5.fill(az_shared_learned, pattern_shared_learned, color='steelblue', alpha=0.12)
ax5.plot(az_shared_optimal, pattern_shared_optimal, color='black', linestyle='--', linewidth=2.5, label='Optimal')

ax5.axvline(bd_azimuth, color='red', linestyle='--', linewidth=2.5, label=f'BD: {np.degrees(bd_azimuth):.1f}°')
for s in range(num_scatterers_vis):
    ax5.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0, 
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax5.set_title('Learned vs Optimal Shared Beam', fontsize=12, fontweight='bold', pad=15)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax5)

# ===== Row 2, Col 3: Gain at BD / Scatter Directions =====
ax6 = fig.add_subplot(2, 3, 6)

direction_labels = ['BD'] + [f'S{s+1}' for s in range(num_scatterers_vis)]
direction_angles = np.concatenate(([bd_azimuth], scatter_azimuths)) if num_scatterers_vis else np.array([bd_azimuth])
learned_direction_gains_db = pattern_to_db(sample_pattern_at_angles(az_shared_learned, pattern_shared_learned, direction_angles))
optimal_direction_gains_db = pattern_to_db(sample_pattern_at_angles(az_shared_optimal, pattern_shared_optimal, direction_angles))

x = np.arange(len(direction_labels))
width = 0.36
ax6.bar(x - width / 2, learned_direction_gains_db, width=width, color='steelblue', alpha=0.85, label='Learned')
ax6.bar(x + width / 2, optimal_direction_gains_db, width=width, color='gray', alpha=0.85, label='Optimal')

ax6.set_xticks(x)
ax6.set_xticklabels(direction_labels)
ax6.set_ylabel('Normalized Gain (dB)', fontsize=11, fontweight='bold')
ax6.set_title('Gain at BD and Scatter Directions', fontsize=12, fontweight='bold')
ax6.grid(True, axis='y', alpha=0.3, linestyle='--')
ax6.legend(loc='upper right', fontsize=9, framealpha=0.9)

gain_floor = min(-30.0, np.min(np.concatenate([learned_direction_gains_db, optimal_direction_gains_db])) - 3.0)
ax6.set_ylim([gain_floor, 2.0])
ax6.text(0.02, 0.98, 'Angles use ULA-folded azimuth sector',
         transform=ax6.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()

# Save figure
# fig_filename = os.path.join(drive_save_path, f'beam_pattern_2d_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}.png')
# plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
# print(f"Beam pattern figure saved to {fig_filename}")

plt.show()


# %%
