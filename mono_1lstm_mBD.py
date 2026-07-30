"""
MONO-Static MIMO with BD SINR Maximization Objective

There exist other static scatters and multiple BDs.
BDs apply orthogonal Hadamard codes.
BD alters its states to distinguish itself from other reflectors.

Objective: identify BD and maximize the wake-up BD SINR.

Architecture: Tx and Rx are co-located.
- Tx only trains transmit beamformer w
- Rx only trains receive beamformer v

"""

import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# TensorFlow/CUDA environment variables must be set before importing TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
from scipy.linalg import eigh
from tensorflow.keras.layers import BatchNormalization, Dense
from parse_args import parse_args
from channel_functions import *

args = parse_args()

# GPU configuration
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
    """LSTM cell with one fused projection for all four gates.

    Concatenating the observation and previous hidden state allows all input
    and recurrent gate projections to be evaluated by one matrix
    multiplication instead of the eight Dense calls used previously.
    """

    def __init__(self, hidden_size, name):
        super(LSTM_Cell, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.gate_projection = Dense(
            units=4 * hidden_size,
            activation='linear',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name='gate_projection'
        )

    def call(self, inputs, **kwargs):
        input_x, h_old, c_old = inputs

        # Shape: (batch, input_size + hidden_size).
        recurrent_input = tf.concat(
            [input_x, h_old], axis=1, name='recurrent_input'
        )

        # Gate order is input, forget, output, candidate. This is one Dense
        # operation and therefore one matrix multiplication for all gates.
        gate_values = self.gate_projection(recurrent_input)
        input_gate_raw, forget_gate_raw, output_gate_raw, candidate_raw = \
            tf.split(gate_values, num_or_size_splits=4, axis=1)

        input_gate = tf.sigmoid(input_gate_raw)
        forget_gate = tf.sigmoid(forget_gate_raw)
        output_gate = tf.sigmoid(output_gate_raw)
        candidate = tf.tanh(candidate_raw)

        c_new = input_gate * candidate + forget_gate * c_old
        h_new = output_gate * tf.tanh(c_new)
        return h_new, c_new


#####################################################
# System Configuration
#####################################################

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

num_users = 3#args.num_users  # Number of independently modulating BD
if num_users < 1:
    raise ValueError("--num_users must be at least 1")

drive_save_path = f'Mo_1LSTM_{num_users}BD_results'
os.makedirs(drive_save_path, exist_ok=True)

num_scatters = args.N_scatterers  # excluding BD
Rician_factor = args.rician_factor
location_bd = None

# Sensing parameters
tau = args.tau  # Pilot length (also number of BD interactions)
K = getattr(args, "N_symbols", 5)  # Number of OFDM symbols per BD state
snr_const = args.snr
snr_const = np.array([snr_const])
ref_dis = 15*Wavelength
Pvec = 10 ** (snr_const / 10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4)  / N_tx / N_rx

# An orthogonal Hadamard preamble is used at every sensing step. Column 0 is
# the common/static-channel code and columns 1..num_users are the zero-mean BD
# codes. Exactly one randomly selected BD transmits its code; the other BDs
# transmit zeros.
if tau < 1:
    raise ValueError("tau must be at least 1")

def make_bd_preamble(num_devices):
    """Return enough orthogonal Hadamard columns for any positive BD count."""
    from scipy.linalg import hadamard
    num_codes = num_devices + 1
    preamble_length = 1
    while preamble_length < num_codes:
        preamble_length *= 2
    return hadamard(preamble_length)[:, :num_codes].astype(np.float32)

BD_preamble = make_bd_preamble(num_users)
preamble_length = BD_preamble.shape[0]
print(f"Hadamard preamble (common code followed by {num_users} BD codes):\n{BD_preamble}")

#####################################################
# Optimized Data Generation Helpers (Performance Optimization)
#####################################################

def _generate_single_sample(loc_tx, loc_rx, n_scatters, n_tx, n_rx, rician):
    """Generate a single channel sample"""
    # Generate random locations
    bd_loc = generate_location_mimo(num_users, 'u')
    scatter_loc = generate_location_mimo(n_scatters, 's')
    
    # Generate the shared direct/scatter channels once and all BD channels
    # together. H_b has shape (num_users, N_rx, N_tx).
    _, H_d, H_r, H_b = generate_mimo_channel(
        loc_tx, loc_rx, scatter_loc, bd_loc,
        N_tx_h=n_tx, N_tx_v=1,
        N_rx_h=n_rx, N_rx_v=1,
        Rician_factor=rician
    )
    
    # Compute location info
    d_bd_rx = np.linalg.norm(bd_loc - loc_rx, axis=1)
    d_bd_tx = np.linalg.norm(bd_loc - loc_tx, axis=1)
    azimuth_bd = np.arctan2(bd_loc[:, 1] - loc_rx[1], bd_loc[:, 0] - loc_rx[0])
    loc_info = np.stack([azimuth_bd, d_bd_rx, d_bd_tx], axis=0)
    
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
    """Alternating generalized-eigenvector solution for sum-SINR over all BDs."""
    batch_size = H_b_batch.shape[0]
    n_rx, n_tx = H_b_batch.shape[2:4]
    v_opt_batch = np.zeros((batch_size, n_rx, 1), dtype=np.complex64)
    w_opt_batch = np.zeros((batch_size, n_tx, 1), dtype=np.complex64)
    
    for i in range(batch_size):
        H_b_i = H_b_batch[i]
        H_int_i = H_int_batch[i]
        best_sinr = -np.inf
        for _ in range(num_restarts):
            try:
                w = np.random.randn(n_tx) + 1j * np.random.randn(n_tx)
                w /= np.linalg.norm(w)
                for _ in range(20):
                    desired_rx = np.stack([H @ w for H in H_b_i])
                    R_sig_rx = P_val * desired_rx.T @ desired_rx.conj()
                    int_rx = H_int_i @ w
                    R_int_rx = P_val * np.outer(int_rx, int_rx.conj()) + noise_var_val * np.eye(n_rx)
                    vals, vecs = eigh(
                        R_sig_rx, R_int_rx, check_finite=False
                    )
                    v = vecs[:, np.argmax(np.real(vals))]
                    v /= np.linalg.norm(v)
                    desired_tx = np.stack([H.conj().T @ v for H in H_b_i])
                    R_sig_tx = P_val * desired_tx.T @ desired_tx.conj()
                    int_tx = H_int_i.conj().T @ v
                    R_int_tx = P_val * np.outer(int_tx, int_tx.conj()) + noise_var_val * np.eye(n_tx)
                    vals, vecs = eigh(
                        R_sig_tx, R_int_tx, check_finite=False
                    )
                    w = vecs[:, np.argmax(np.real(vals))]
                    w /= np.linalg.norm(w)
            except Exception:
                continue
            desired = sum(abs(np.vdot(v, H @ w)) ** 2 for H in H_b_i) * P_val
            interference = abs(np.vdot(v, H_int_i @ w)) ** 2 * P_val + noise_var_val
            if desired / interference > best_sinr:
                best_sinr = desired / interference
                v_opt_batch[i, :, 0] = v
                w_opt_batch[i, :, 0] = w
        if not np.isfinite(best_sinr):
            U, _, Vh = np.linalg.svd(np.sum(H_b_i, axis=0))
            v_opt_batch[i, :, 0] = U[:, 0]
            w_opt_batch[i, :, 0] = Vh[0, :].conj()
    
    return v_opt_batch.astype(np.complex64), w_opt_batch.astype(np.complex64)

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

def hadamard_codebook(n, m):
    """
    Create a unit-norm Hadamard-based codebook of shape (n, m).
    
    Hadamard matrices only exist for sizes 1, 2, or multiples of 4.
    We find the smallest valid Hadamard size >= max(n, m), construct it,
    then take the first n rows and m columns with unit-norm normalization.
    
    Properties:
    - All columns are orthogonal: S^H S = I (when m <= hadamard_size)
    - SS^H has better conditioning than truncated DFT for arbitrary tau
    """
    from scipy.linalg import hadamard
    
    # Find smallest Hadamard size >= max(n, m)
    target_size = max(n, m)
    if target_size <= 1:
        had_size = 1
    elif target_size <= 2:
        had_size = 2
    else:
        # Hadamard exists for multiples of 4
        had_size = 4
        while had_size < target_size:
            had_size *= 2
    
    # Generate Hadamard matrix and normalize
    H = hadamard(had_size).astype(np.float64) / np.sqrt(n)
    
    # Take first n rows and m columns, convert to complex
    codebook = H[:n, :m].astype(np.complex64)
    
    return codebook

pilot_codebook = hadamard_codebook(N_tx, max(N_tx, tau))
s_pilot = pilot_codebook[:, :tau].copy()

# Beam-sweeping codebook: ensure exactly 2*tau beams are available.
sweep_codebook = hadamard_codebook(N_tx, max(N_tx, 2 * tau))[:, : (2 * tau)].copy()
    
#####################################################
# Learning parameters and Computation Graph
#####################################################
#  
initial_run = 1 if args.n_epochs > 0 else 0
n_epochs = args.n_epochs
learning_rate = args.learning_rate
batch_per_epoch = args.batch_per_epoch
batch_size_order = args.batch_size_order
val_size_order = 20
test_size = 800

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Placeholders
loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_users), name="loc_input")
H_d_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_d")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, num_users, N_ris, N_tx), name="H_b")
H_r_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_r")

##################### ACTIVE SENSING NETWORK #####################

with tf.name_scope("system_parameters"):
    lay = {}
    lay['P'] = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(), name="tx_power")
    hadamard_preamble = tf.constant(BD_preamble, dtype=tf.float32)
    common_seq = hadamard_preamble[:, 0]
    bd_seq = hadamard_preamble[:, 1:num_users + 1]

with tf.name_scope("active_sensing_agent"):
    hidden_size = args.hidden_size  # Shared hidden size for both nodes
    
    # One fused recurrent state for both Y1 (BD-differential) and Y2 (common term).
    LSTM1 = LSTM_Cell(hidden_size, name='LSTM_fused')
    
    # RIS only trains transmit beamformer w
    mlp_ris_tx = MLPBlock(3, [hidden_size * 2, hidden_size * 2, 2 * N_tx], name='RIS_transmitter')
    # Rx only trains receive beamformer v  
    mlp_rx_rx = MLPBlock(3, [hidden_size * 2, hidden_size * 2, 2 * N_rx], name='Receiver_receiver')
    
    # SNR feature
    snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
    snr_dB = 10 * tf.log(snr) / np.log(10)
    snr_normal = snr_dB
    
    # Storage
    v_list = []
    w_list = []
    Y1 = tf.zeros([tf.shape(loc_input)[0], num_users, N_rx], dtype=tf.complex64)
    Y2 = tf.zeros([tf.shape(loc_input)[0], N_rx], dtype=tf.complex64)

    batch_size = tf.shape(loc_input)[0]

    # Select one BD per sample and keep it active for the complete tau-step
    # sensing episode. This index is also the identification target.
    active_user = tf.random_uniform(
        [batch_size], minval=0, maxval=num_users, dtype=tf.int32,
        name="active_user"
    )
    active_mask = tf.one_hot(
        active_user, depth=num_users, dtype=tf.float32,
        name="active_user_mask"
    )
    
    for t in range(tau):
        'Initialization at t=0'
        if t == 0:
            # Initialize shared LSTM state.
            h_old = tf.zeros([batch_size, hidden_size])
            c_old = tf.zeros([batch_size, hidden_size])
            
            # Initialize first RIS transmit beamformer w 
            w_init_real = tf.get_variable("w_init_real", shape=(1, N_tx, 1), trainable=True)
            w_init_imag = tf.get_variable("w_init_imag", shape=(1, N_tx, 1), trainable=True)
            w_complex_init = tf.complex(w_init_real, w_init_imag)
            w1 = w_complex_init / (tf.cast(tf.norm(w_complex_init, axis=1, keepdims=True), tf.complex64) + 1e-8)
            w1 = tf.tile(w1, [batch_size, 1, 1])
            
            # Initialize first Rx receive beamformer v
            v_init_real = tf.get_variable("v_init_real", shape=(1, N_rx, 1), trainable=True)
            v_init_imag = tf.get_variable("v_init_imag", shape=(1, N_rx, 1), trainable=True)
            v_complex_init = tf.complex(v_init_real, v_init_imag)
            v1 = v_complex_init / (tf.cast(tf.norm(v_complex_init, axis=1, keepdims=True), tf.complex64) + 1e-8)
            v1 = tf.tile(v1, [batch_size, 1, 1])
        
        # The selected BD transmits its Hadamard sequence; all others transmit zeros.
        bd_chip_states = tf.expand_dims(active_mask, 1) * tf.expand_dims(bd_seq, 0)
        bd_chip_states = tf.cast(bd_chip_states, tf.complex64)

        # Keep the preamble-chip dimension explicit so this graph works for
        # every num_users/preamble_length without per-chip tensor definitions.
        H_common = H_d_placeholder + H_r_placeholder
        H_bd_chips = tf.einsum('bpu,burc->bprc', bd_chip_states, H_b_placeholder)
        H_eff = tf.expand_dims(H_common, axis=1) + H_bd_chips

        sqrt_p = tf.complex(tf.sqrt(lay['P']), 0.0)
        y_noiseless = tf.einsum(
            'bprc,bci->bpri', H_eff, w1
        )  # (batch, preamble_length, N_rx, 1)
        y_noiseless = tf.tile(
            sqrt_p * y_noiseless, [1, 1, 1, K]
        )

        # A single tensor draw still gives independent noise for every sample,
        # preamble chip, receive antenna, and repeated observation.
        noise_shape = [batch_size, preamble_length, N_rx, K]
        noise = tf.complex(
            tf.random_normal(
                noise_shape, mean=0.0, stddev=noiseSTD_per_dim,
                name="noise_real_t{}".format(t)
            ),
            tf.random_normal(
                noise_shape, mean=0.0, stddev=noiseSTD_per_dim,
                name="noise_imag_t{}".format(t)
            )
        )
        y_preamble = y_noiseless + noise
        # Received preamble sequence: (batch, preamble_length, N_rx, K)
        y_bd_despread = tf.einsum(
            'pu,bprk->burk', tf.cast(bd_seq, tf.complex64), y_preamble
        ) / tf.cast(preamble_length, tf.complex64)
        y_common_despread = tf.einsum(
            'p,bprk->brk', tf.cast(common_seq, tf.complex64), y_preamble
        ) / tf.cast(preamble_length, tf.complex64)

        # Average the K repeated observations while retaining the full receive-
        # antenna vectors before applying v1^H.
        y_bd_before_beamforming = tf.reduce_mean(
            y_bd_despread, axis=3
        )  # (batch, num_users, N_rx)
        y_common_before_beamforming = tf.reduce_mean(
            y_common_despread, axis=2
        )  # (batch, N_rx)

        Y1 =  y_bd_before_beamforming
        Y2 =  y_common_before_beamforming

        # Apply the receive beamformer: Y1_after = v1^H Y1 and
        # Y2_after = v1^H Y2.
        Y1_after = tf.squeeze(
            tf.matmul(tf.linalg.adjoint(v1), tf.transpose(Y1, [0, 2, 1])),
            axis=1
        )  # (batch, num_users)
        Y2_after = tf.reshape(
            tf.matmul(tf.linalg.adjoint(v1), tf.expand_dims(Y2, axis=2)),
            [-1, 1]
        )  # (batch, 1)

        # Include both the despread receive vectors before beamforming and the
        # scalar observations after applying v1^H.
        y_real = tf.concat([
            tf.reshape(
                tf.cast(tf.real(y_bd_before_beamforming), tf.float32),
                [batch_size, num_users * N_rx]
            ),
            tf.reshape(
                tf.cast(tf.imag(y_bd_before_beamforming), tf.float32),
                [batch_size, num_users * N_rx]
            ),
            tf.cast(tf.real(Y1_after), tf.float32),
            tf.cast(tf.imag(Y1_after), tf.float32),
        ], axis=1)  # (batch, 2*num_users*N_rx + 2*num_users)

        y_real2 = tf.concat([
            tf.cast(tf.real(y_common_before_beamforming), tf.float32),
            tf.cast(tf.imag(y_common_before_beamforming), tf.float32),
            tf.cast(tf.real(Y2_after), tf.float32),
            tf.cast(tf.imag(Y2_after), tf.float32),
        ], axis=1)  # (batch, 2*N_rx + 2)
        
        'Update one fused LSTM state from concatenated Y1/Y2 statistics'
        fused_obs = tf.concat([y_real, y_real2, snr_normal], axis=1)
        h_old, c_old = LSTM1((fused_obs, h_old, c_old))
        
        'RIS designs transmit beamformer w based on the fused hidden state'
        w_her = mlp_ris_tx(tf.concat([h_old, c_old], axis=1))
        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm + 1e-8)
        w1 = tf.complex(w_her[:, 0:N_tx], w_her[:, N_tx:2 * N_tx])
        w1 = tf.reshape(w1, [-1, N_tx, 1])
        
        'Rx designs receive beamformer v based on the fused hidden state'
        v_her = mlp_rx_rx(tf.concat([h_old, c_old], axis=1))
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm + 1e-8)
        v1 = tf.complex(v_her[:, 0:N_rx], v_her[:, N_rx:2 * N_rx])
        v1 = tf.reshape(v1, [-1, N_rx, 1])
        
        # Store beamformers for analysis
        v_list.append(v1)
        w_list.append(w1)

    #####################################################
    # Output Final Beamformers after tau interactions
    # Both RIS and Rx use the shared LSTM state c_old
    #####################################################
    MLP_bf_w = MLPBlock(3, [2 * hidden_size, 2 * hidden_size, 2 * N_tx], name='MLP_bf_w')
    MLP_bf_v = MLPBlock(3, [2 * hidden_size, 2 * hidden_size, 2 * N_rx], name='MLP_bf_v')
    
    # Final RIS transmit beamformer w (from shared state)
    w_tmp = MLP_bf_w(tf.concat([h_old, c_old], axis=1))
    w_norm = tf.reshape(tf.norm(w_tmp, axis=1), (-1, 1))
    w_tmp = tf.divide(w_tmp, w_norm + 1e-8)
    w_complex = tf.complex(w_tmp[:, 0:N_tx], w_tmp[:, N_tx:2 * N_tx])
    w_complex = tf.reshape(w_complex, [-1, N_tx, 1])
    
    # Final Rx receive beamformer v (from shared state)
    v_tmp = MLP_bf_v(tf.concat([h_old, c_old], axis=1))
    v_norm = tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1))
    v_tmp = tf.divide(v_tmp, v_norm + 1e-8)
    v_complex = tf.complex(v_tmp[:, 0:N_rx], v_tmp[:, N_rx:2 * N_rx])
    v_complex = tf.reshape(v_complex, [-1, N_rx, 1])

    v_list.append(v_complex)
    w_list.append(w_complex)

    # Identify the active BD from the final recurrent state.
    bd_classifier = Dense(
        units=num_users, activation='linear', name='BD_classifier'
    )
    classifier_input = tf.concat([h_old, c_old], axis=1)
    id_logits = bd_classifier(classifier_input)
    id_probabilities = tf.nn.softmax(id_logits, axis=1, name='BD_probabilities')
    predicted_user = tf.argmax(
        id_logits, axis=1, output_type=tf.int32, name='predicted_BD'
    )
    identification_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=active_user, logits=id_logits
        ),
        name='identification_loss'
    )
    identification_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted_user, active_user), tf.float32),
        name='identification_accuracy'
    )
#####################################################
# SINR: optimize the randomly selected active BD
#####################################################

with tf.name_scope("sinr_computation"):
    
    # BD signal power: P * |v^H H_b w|^2
    # v^H: (batch, 1, N_rx), H_b: (batch, N_rx, N_tx), w: (batch, N_tx, 1)
    H_b_w = tf.einsum('burc,bci->buri', H_b_placeholder, w_complex)
    bd_amplitudes = tf.einsum('bri,buri->bu', tf.math.conj(v_complex), H_b_w)
    sig_BD_per_device = tf.abs(bd_amplitudes) ** 2 * lay['P']
    sig_BD = tf.reduce_sum(sig_BD_per_device, axis=1)
    
    # Interference channel: H_d + H_r
    H_interference = H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
    
    # Interference power: P * |v^H (H_d + H_r) w|^2
    sig_int = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_interference, w_complex))  # (batch, 1, 1)
    sig_int = tf.squeeze(tf.abs(sig_int) ** 2) * lay['P']  # (batch,)
    
    sinr_BD = sig_BD / (sig_int + noise_var)
    sinr_BD_per_device = sig_BD_per_device / tf.expand_dims(sig_int + noise_var, 1)
    sinr_active = tf.reduce_sum(
        sinr_BD_per_device * active_mask, axis=1, name='active_BD_sinr'
    )
    sinr_BD_clipped = tf.clip_by_value(sinr_BD, 1e-5, 1e2)
    sinr_active_clipped = tf.clip_by_value(sinr_active, 1e-5, 1e2)
    
    # Log SINR with per-sample clipping to prevent outliers from dominating
    log_sinr_BD_raw = tf.log(sinr_BD_clipped)
    log_sinr_BD = tf.clip_by_value(log_sinr_BD_raw, -10.0, 10.0)  # ~+-35 dB range
    log_sinr_active_raw = tf.log(sinr_active_clipped)
    log_sinr_active = tf.clip_by_value(
        log_sinr_active_raw, -10.0, 10.0
    )
    
    # For backward compatibility, define sig_ref as interference
    sig_ref = sig_int

    # Calculate the SINR towards the scatters
    sig_scatter = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_r_placeholder, w_complex))  # (batch, 1, 1)
    H_b_sum = tf.reduce_sum(H_b_placeholder, axis=1)
    int_scatter = tf.matmul(tf.linalg.adjoint(v_complex), \
                            tf.matmul(H_d_placeholder+H_b_sum, w_complex))  # (batch, 1, 1)
    sinr_scatter = tf.squeeze(tf.abs(sig_scatter) ** 2) * lay['P'] / \
                    (tf.squeeze(tf.abs(int_scatter) ** 2) * lay['P'] + noise_var)


#####################################################
# Optimal Beamformer (SVD-based solution for MIMO)
#####################################################

with tf.name_scope("optimal_beamformer"):
    batch_size_opt = tf.shape(H_b_placeholder)[0]

    # Keep only the woke-up BD channel for each sample. The resulting shape is
    # (batch, 1, N_rx, N_tx), as expected by the batch optimizer.
    active_channel_mask = tf.reshape(
        tf.cast(active_mask, tf.complex64), [-1, num_users, 1, 1]
    )
    H_b_active = tf.reduce_sum(
        active_channel_mask * H_b_placeholder,
        axis=1,
        keepdims=True,
        name="active_BD_channel"
    )
    
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
    v_opt, w_opt = tf.py_func(
        lambda H_b, H_int: compute_optimal_beamformers_np(H_b, H_int, noise_var, Pvec[0]),
        [H_b_active, H_interference],
        [tf.complex64, tf.complex64]
    )
    
    # Set shapes explicitly
    v_opt = tf.reshape(v_opt, [-1, N_rx, 1])
    w_opt = tf.reshape(w_opt, [-1, N_tx, 1])
    # Optimal SINR for the woke-up BD only.
    opt_bd_amp = tf.einsum(
        'bri,buri->bu',
        tf.math.conj(v_opt),
        tf.einsum('burc,bci->buri', H_b_active, w_opt)
    )
    sig_BD_opt_per_device = tf.abs(opt_bd_amp) ** 2 * lay['P']
    sig_BD_opt = tf.squeeze(sig_BD_opt_per_device, axis=1)
    
    sig_int_opt = tf.matmul(tf.linalg.adjoint(v_opt), tf.matmul(H_interference, w_opt))
    sig_int_opt = tf.squeeze(tf.abs(sig_int_opt) ** 2) * lay['P']
    
    sinr_BD_opt = sig_BD_opt / (sig_int_opt + noise_var )

    scatter_sig_opt = tf.matmul(tf.linalg.adjoint(v_opt), tf.matmul(H_r_placeholder, w_opt))  # (batch, 1, 1)
    scatter_sig_opt = tf.squeeze(tf.abs(scatter_sig_opt) ** 2) * lay['P']
    scatter_int_opt = tf.matmul(tf.linalg.adjoint(v_opt), \
                            tf.matmul(H_d_placeholder+H_b_sum, w_opt))  # (batch, 1, 1)
    scatter_int_opt = tf.squeeze(tf.abs(scatter_int_opt) ** 2) * lay['P']
    sinr_scatter_opt = scatter_sig_opt / (scatter_int_opt + noise_var )
    
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

    H_int_expanded = tf.expand_dims(H_interference, 1)
    H_eff0 = -H_b_placeholder + H_int_expanded
    Y_0_clean = tf.matmul(H_eff0, s_pilot_tf)  # (batch, N_rx, tau)
    noise_0 = tf.complex(
        tf.random_normal(tf.shape(Y_0_clean), mean=0.0, stddev=noiseSTD_per_dim),
        tf.random_normal(tf.shape(Y_0_clean), mean=0.0, stddev=noiseSTD_per_dim)
    )
    Y_0 = sqrt_p * Y_0_clean + noise_0
    H_eff_hat0 = tf.matmul(Y_0, s_pilot_pinv) / (sqrt_p + 1e-10)

    H_eff1 = H_b_placeholder + H_int_expanded
    Y_1_clean = tf.matmul(H_eff1, s_pilot_tf)  # (batch, N_rx, tau)
    noise_1 = tf.complex(
        tf.random_normal(tf.shape(Y_1_clean), mean=0.0, stddev=noiseSTD_per_dim),
        tf.random_normal(tf.shape(Y_1_clean), mean=0.0, stddev=noiseSTD_per_dim)
    )
    Y_1 = sqrt_p * Y_1_clean + noise_1
    H_eff_hat1 = tf.matmul(Y_1, s_pilot_pinv) / (sqrt_p + 1e-10)  # (batch, N_rx, N_tx)

    H_int_estimated = tf.reduce_mean(
        (H_eff_hat0 + H_eff_hat1) / tf.cast(2, tf.complex64), axis=1
    )
    H_bd_estimated = (H_eff_hat1 - H_eff_hat0) / tf.cast(2, tf.complex64)

    v_sp, w_sp = tf.py_func(
        lambda H_b, H_int: compute_optimal_beamformers_np(H_b, H_int, noise_var, Pvec[0]),
        [H_bd_estimated, H_int_estimated],
        [tf.complex64, tf.complex64]
    )
    
    # Set shapes explicitly
    v_sp = tf.reshape(v_sp, [-1, N_rx, 1])
    w_sp = tf.reshape(w_sp, [-1, N_tx, 1])
    
    # # Compute SINR with signal processing beamformers
    sp_bd_amp = tf.einsum('bri,buri->bu', tf.math.conj(v_sp), tf.einsum('burc,bci->buri', H_b_placeholder, w_sp))
    sig_BD_sp = tf.reduce_sum(tf.abs(sp_bd_amp) ** 2, axis=1) * lay['P']
    
    sig_int_sp = tf.matmul(tf.linalg.adjoint(v_sp), tf.matmul(H_interference, w_sp))
    sig_int_sp = tf.squeeze(tf.abs(sig_int_sp) ** 2) * lay['P']
    
    sinr_BD_sp = sig_BD_sp / (sig_int_sp + noise_var )

    #### Beam sweeping baseline (selection from estimated channels; evaluation on true channels).
    sweep_codebook_tf = tf.constant(sweep_codebook, dtype=tf.complex64)
    
    # Beam sweeping from estimated channels
    Hbd_est_W = tf.einsum('burc,ck->burk', H_bd_estimated, sweep_codebook_tf)
    codebook_H = tf.linalg.adjoint(sweep_codebook_tf)         # (num_beams, N_tx), N_tx == N_rx
    codebook_H_expanded = tf.expand_dims(codebook_H, 0)       # (1, num_beams, N_rx)
    bi_matrix = tf.abs(tf.einsum('kr,burm->bukm', codebook_H, Hbd_est_W)) ** 2
    sweep_metric = tf.reduce_sum(tf.linalg.diag_part(bi_matrix), axis=1)
    best_w_idx = tf.argmax(sweep_metric, axis=1, output_type=tf.int32)

    # Gather selected Tx beam
    codebook_T = tf.transpose(sweep_codebook_tf)  # (num_beams, N_tx)
    w_sweep = tf.gather(codebook_T, best_w_idx)  # (batch, N_tx)
    w_sweep = tf.reshape(w_sweep, [-1, N_tx, 1])
    w_sweep = w_sweep / tf.cast(tf.norm(w_sweep, axis=1, keepdims=True) + 1e-10, tf.complex64)

    # Nulling Rx beam: v = (I - xx^H/||x||^2) * seed, where x = H_int_est * w
    x_int_est = tf.matmul(H_int_estimated, w_sweep)  # (batch, N_rx, 1)
    x_norm_sq = tf.reduce_sum(tf.abs(x_int_est) ** 2, axis=[1, 2], keepdims=True)  # (batch,1,1)
    proj_x = tf.matmul(x_int_est, tf.linalg.adjoint(x_int_est)) / tf.cast(x_norm_sq + 1e-10, tf.complex64)

    batch_size_sweep = tf.shape(w_sweep)[0]
    eye_rx = tf.tile(tf.expand_dims(tf.eye(N_rx, dtype=tf.complex64), 0), [batch_size_sweep, 1, 1])
    null_proj = eye_rx - proj_x

    if N_rx == N_tx:
        v_seed = w_sweep
    else:
        v_seed = tf.reduce_sum(tf.einsum('burc,bci->buri', H_bd_estimated, w_sweep), axis=1)

    v_sweep_raw = tf.matmul(null_proj, v_seed)
    # Use real-valued power norm for compatibility in TF versions that keep complex dtype in tf.norm.
    v_raw_norm_sq = tf.reduce_sum(tf.abs(v_sweep_raw) ** 2, axis=[1, 2], keepdims=True)  # (batch,1,1), real
    v_valid_mask = tf.tile(v_raw_norm_sq > 1e-8, [1, N_rx, 1])  # (batch,N_rx,1)
    v_sweep_safe = tf.where(v_valid_mask, v_sweep_raw, v_seed)
    v_sweep = v_sweep_safe / tf.cast(tf.norm(v_sweep_safe, axis=1, keepdims=True) + 1e-10, tf.complex64)
    
    # Compute final SINR using selected beams on TRUE channels
    sweep_bd_amp = tf.einsum('bri,buri->bu', tf.math.conj(v_sweep), tf.einsum('burc,bci->buri', H_b_placeholder, w_sweep))
    sig_BD_sweep = tf.reduce_sum(tf.abs(sweep_bd_amp) ** 2, axis=1) * lay['P']
    
    sig_int_sweep = tf.matmul(tf.linalg.adjoint(v_sweep), tf.matmul(H_interference, w_sweep))
    sig_int_sweep = tf.squeeze(tf.abs(sig_int_sweep) ** 2) * lay['P']
    
    sinr_BD_sweep = sig_BD_sweep / (sig_int_sweep + noise_var + 1e-10)


#####################################################
# Loss and Optimizer
#####################################################

# Multi-task loss: maximize the active BD SINR and identify that BD.
sinr_loss = -tf.reduce_mean(log_sinr_active)
lambda_id = 1.0
loss = sinr_loss + lambda_id * identification_loss

# Regularization
global_step = tf.train.get_or_create_global_step()
l2 = args.l2
reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss_reg = loss + l2 * reg_term


# Add warmup and slower decay
warmup_steps = args.warmup_steps
global_step_float = tf.cast(global_step, tf.float32)
warmup_lr = learning_rate * tf.minimum(1.0, global_step_float / warmup_steps)

decayed_lr = tf.train.exponential_decay(
    learning_rate, 
    global_step, 
    decay_steps=args.decay_steps,  # Slower decay
    decay_rate=args.decay_rate,    # Gentler decay
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

clipped_grads, global_norm = tf.clip_by_global_norm([g for g in safe_grads if g is not None], args.clip_norm)

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
    tf.check_numerics(log_sinr_active, 'log_sinr_active NaN'),
    tf.check_numerics(identification_loss, 'identification_loss NaN')
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
print("Active-BD SINR Maximization and Identification (One LSTM)")
print("=" * 60)
print(f"N_tx: {N_tx}, N_rx: {N_rx}, tau: {tau}, K: {K}, SNR: {snr_const[0]} dB")
print(f"Backscatter devices: {num_users} (active-BD SINR + identification objective)")
print(f"Identification-loss weight: {lambda_id:g}")
print(f"LR: {learning_rate:g}, clip_norm: {args.clip_norm:g}, warmup: {warmup_steps}, "
    f"decay_steps: {args.decay_steps}, decay_rate: {args.decay_rate:g}, hidden: {hidden_size}")
print("=" * 60 + "\n")

model_ckpt = (
    f'{drive_save_path}/params_joint_sinr_id_fused_lstm_'
    f'N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}_K_{K}_'
    f'Nsc_{num_scatters}_Nbd_{num_users}'
)

with tf.Session() as sess:
    if initial_run == 1:
        init.run()
        best_val = np.inf
    else:
        saver.restore(sess, model_ckpt)
        best_val = float(sess.run(loss, feed_dict=feed_dict_val))
        print(f"Restored validation loss baseline: {best_val:.6f}")
        n_epochs = 5 # If continue training
    
    # Early stopping
    wait = 0
    PATIENCE = args.patience if args.patience > 0 else max(20, 2 * tau)
    
    for epoch in range(n_epochs):
        batch_iter = 0
        epoch_train_losses = []
        epoch_sinr_values = []
        epoch_id_accuracies = []
        
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
            
            _, train_loss, sinr_values, id_accuracy_value = sess.run(
                [training_op, loss, sinr_active, identification_accuracy],
                feed_dict=feed_dict_batch
            )
            
            epoch_train_losses.append(train_loss)
            epoch_sinr_values.append(np.mean(sinr_values))
            epoch_id_accuracies.append(id_accuracy_value)
            batch_iter += 1
        
        # Validation
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_sinr = np.mean(epoch_sinr_values)
        avg_train_id_accuracy = np.mean(epoch_id_accuracies)
        
        loss_val, sinr_loss_val, id_loss_val, sinr_val, id_accuracy_val, \
            sinr_scatter_val, sig_bd_val, sig_ref_val = sess.run(
            [loss, sinr_loss, identification_loss, sinr_active,
             identification_accuracy, sinr_scatter, sig_BD, sig_ref],
            feed_dict=feed_dict_val
        )

        print(f'Epoch {epoch:3d} | '
              f'Train Loss: {avg_train_loss:8.4f} | '
              f'Val Loss: {loss_val:8.4f} | '
              f'Best: {best_val:8.4f}')
        print(f'         | '
              f'ID Accuracy train/val: {100 * avg_train_id_accuracy:6.2f}% / '
              f'{100 * id_accuracy_val:6.2f}% | '
              f'Val SINR Loss: {sinr_loss_val:8.4f} | '
              f'Val ID Loss: {id_loss_val:8.4f}')
        print(f'         | '
              f'SINR_active (learned): '
              f'{10 * np.log10(np.mean(sinr_val) + 1e-10):6.2f} dB')
        print(f'         | '
              f'Sig_BD: {np.mean(sig_bd_val):8.4f} | '
              f'Sig_int: {np.mean(sig_ref_val):8.4f} | ')
        print(f'         | '
              f'SINR_scatter: '
              f'{10 * np.log10(np.mean(sinr_scatter_val) + 1e-10):6.2f} dB')
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
    # Testing and optimal-beamformer benchmark
    #
    # The expensive NumPy/SciPy optimal solver is deliberately evaluated only
    # after all gradient updates have finished. It is not needed for training.
    #####################################################
    print("\n" + "=" * 60)
    print("Testing (including post-training optimal benchmark)")
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
    
    sinr_test, sinr_sum_test, sinr_per_device_test, sinr_opt_test, \
        sinr_scatter_test, v_learned, w_learned, v_optimal, w_optimal, \
        v_list_test, w_list_test, active_user_test, predicted_user_test, \
        id_probabilities_test, id_accuracy_test, id_loss_test = sess.run(
            [sinr_active, sinr_BD, sinr_BD_per_device, sinr_BD_opt,
             sinr_scatter, v_complex, w_complex, v_opt, w_opt, v_list,
             w_list, active_user, predicted_user, id_probabilities,
             identification_accuracy, identification_loss],
            feed_dict=feed_dict_test
    )

    sinr_sp_test, sinr_sweep_test = sess.run([sinr_BD_sp, sinr_BD_sweep], feed_dict=feed_dict_test)
    
    print(f"Test active-BD identification accuracy: {100 * id_accuracy_test:6.2f}%")
    print(f"Test identification loss: {id_loss_test:8.4f}")
    print(f"Test SINR_active (learned): {10 * np.log10(np.mean(sinr_test) + 1e-10):6.2f} dB")
    print(f"Test sum-SINR (learned):    {10 * np.log10(np.mean(sinr_sum_test) + 1e-10):6.2f} dB")
    for device_idx in range(num_users):
        print(f"  BD {device_idx + 1}: {10 * np.log10(np.mean(sinr_per_device_test[:, device_idx]) + 1e-10):6.2f} dB")
    print(f"Test SINR_active (optimal): {10 * np.log10(np.mean(sinr_opt_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_scatter (learned):       {10 * np.log10(np.mean(sinr_scatter_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_BD (signalprocessing): {10 * np.log10(np.mean(sinr_sp_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_BD (sweeping): {10 * np.log10(np.mean(sinr_sweep_test) + 1e-10):6.2f} dB")
    print(f"Active-SINR gap to optimal: {10 * np.log10((np.mean(sinr_opt_test) + 1e-10) / (np.mean(sinr_test) + 1e-10)):6.2f} dB")
    
    # Save results
    model_filename = os.path.join(drive_save_path, \
            f'TEST_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}_K_{K}_Nsca_{num_scatters}.mat')
    sio.savemat(model_filename, dict(
        snr_const=snr_const,
        N_tx=N_tx,
        N_rx=N_rx,
        tau=tau,
        K=K,
        num_users=num_users,
        num_scatters=num_scatters,
        fc=fc,
        wavelength=Wavelength,
        rician_factor=Rician_factor,
        location_tx=location_tx,
        location_rx=location_rx,
        BD_preamble=BD_preamble,
        BD_location=BD_loc,
        Scatter_location=Scatter_loc if Scatter_loc[0] is not None else [],
        sinr_learned=sinr_test,
        sinr_sum_learned=sinr_sum_test,
        sinr_per_device=sinr_per_device_test,
        active_user=active_user_test,
        predicted_user=predicted_user_test,
        identification_probabilities=id_probabilities_test,
        identification_accuracy=id_accuracy_test,
        identification_loss=id_loss_test,
        sinr_optimal=sinr_opt_test,
        sinr_sp=sinr_sp_test,
        sinr_sweep=sinr_sweep_test,
        sinr_scatter_learned=sinr_scatter_test,
        v_learned=v_learned,
        w_learned=w_learned,
        v_optimal=v_optimal,
        w_optimal=w_optimal,
        v_list_test=v_list_test, 
        w_list_test=w_list_test
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
if 'num_users' not in globals():
    num_users = 1

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
    parameter_names = (
        'N_tx', 'N_rx', 'tau', 'snr_const', 'K',
        'num_scatters', 'num_users'
    )
    if all(name in globals() for name in parameter_names):
        snr_arr = np.atleast_1d(np.squeeze(np.asarray(snr_const)))
        snr0 = int(snr_arr[0])
        candidate = os.path.join(
            result_dir,
            f"TEST_sinr_N_{N_tx}_{N_rx}_tau_{tau}_snr_{snr0}_K_{K}_Nsca_{num_scatters}.mat"
        )

    if candidate is None or not os.path.isfile(candidate):
        candidate = [
            os.path.join(result_dir, f) for f in os.listdir(result_dir)
            if f.startswith(('TEST_joint_sinr_id_', 'TEST_sinr_'))
            and f.endswith('.mat')
        ]
        if not candidate:
            raise FileNotFoundError(
                f"No saved joint SINR/identification result files found in {result_dir}. "
                "Run testing once or set visualization_result_file."
            )
        candidate.sort(key=os.path.getmtime)
        candidate = candidate[-1]
    return candidate

required_vis_vars = [
    'w_learned', 'v_learned', 'w_optimal', 'v_optimal',
    'sinr_test', 'sinr_opt_test', 'sinr_sp_test', 'sinr_sweep_test',
    'BD_loc', 'Scatter_loc', 'active_user_test', 'predicted_user_test'
]
force_reload = bool(globals().get('force_load_visualization_data', False))
if force_reload or any(name not in globals() for name in required_vis_vars):
    result_dir = globals().get(
        'drive_save_path', f'Mo_1LSTM_{num_users}BD_results'
    )
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
    if 'K' in loaded:
        K = int(np.squeeze(loaded['K']))
    if 'num_users' in loaded:
        num_users = int(np.squeeze(loaded['num_users']))
    if 'num_scatters' in loaded:
        num_scatters = int(np.squeeze(loaded['num_scatters']))
    if 'fc' in loaded:
        fc = float(np.squeeze(loaded['fc']))
    if 'wavelength' in loaded:
        Wavelength = float(np.squeeze(loaded['wavelength']))
    if 'rician_factor' in loaded:
        Rician_factor = float(np.squeeze(loaded['rician_factor']))
    if 'location_tx' in loaded:
        location_tx = np.squeeze(loaded['location_tx']).astype(np.float64)
    if 'location_rx' in loaded:
        location_rx = np.squeeze(loaded['location_rx']).astype(np.float64)
    if 'snr_const' in loaded:
        snr_const = np.atleast_1d(np.squeeze(loaded['snr_const']))

    w_learned = loaded['w_learned']
    v_learned = loaded['v_learned']
    w_optimal = loaded['w_optimal']
    v_optimal = loaded['v_optimal']

    sinr_test = np.squeeze(loaded.get('sinr_learned', np.array([np.nan])))
    sinr_opt_test = np.squeeze(loaded.get('sinr_optimal', np.array([np.nan])))
    sinr_sp_test = np.squeeze(loaded.get('sinr_sp', np.full_like(np.atleast_1d(sinr_test), np.nan, dtype=np.float64)))
    sinr_sweep_test = np.squeeze(loaded.get('sinr_sweep', np.full_like(np.atleast_1d(sinr_test), np.nan, dtype=np.float64)))

    active_user_test = np.squeeze(loaded.get('active_user', np.array([], dtype=np.int64)))
    predicted_user_test = np.squeeze(loaded.get('predicted_user', np.array([], dtype=np.int64)))
    id_probabilities_test = np.asarray(
        loaded.get('identification_probabilities', np.array([]))
    )
    BD_loc = _decode_saved_locations(loaded.get('BD_location', np.array([])))
    Scatter_loc = _decode_saved_locations(loaded.get('Scatter_location', np.array([])))

if 'location_tx' not in globals():
    location_tx = np.array([0, 0, 0])
if 'snr_const' not in globals():
    snr_const = np.array([np.nan], dtype=np.float64)
snr_display = np.atleast_1d(np.squeeze(np.asarray(snr_const)))[0]

w_learned = np.asarray(w_learned)
v_learned = np.asarray(v_learned)
w_optimal = np.asarray(w_optimal)
v_optimal = np.asarray(v_optimal)

if w_learned.ndim == 2:
    w_learned = w_learned[np.newaxis, ...]
if v_learned.ndim == 2:
    v_learned = v_learned[np.newaxis, ...]
if w_optimal.ndim == 2:
    w_optimal = w_optimal[np.newaxis, ...]
if v_optimal.ndim == 2:
    v_optimal = v_optimal[np.newaxis, ...]

num_realizations = int(w_learned.shape[0])
if len(BD_loc) == 0:
    raise ValueError("No BD locations are available for the 2D scene.")
if len(Scatter_loc) == 0:
    Scatter_loc = [None for _ in range(num_realizations)]
if len(BD_loc) < num_realizations:
    raise ValueError(
        f"Only {len(BD_loc)} BD-location entries are available for "
        f"{num_realizations} beamformer realizations."
    )
if len(Scatter_loc) < num_realizations:
    Scatter_loc += [None for _ in range(num_realizations - len(Scatter_loc))]

active_user_test = np.asarray(active_user_test, dtype=np.int64).reshape(-1)
predicted_user_test = np.asarray(predicted_user_test, dtype=np.int64).reshape(-1)
if active_user_test.size < num_realizations:
    raise ValueError("Saved active-BD labels do not cover all realizations.")
if predicted_user_test.size < num_realizations:
    raise ValueError("Saved identified-BD labels do not cover all realizations.")

# Select one test instance for visualization (random unless beam_pattern_idx is set)
idx_default = np.random.randint(num_realizations)
idx = int(globals().get('beam_pattern_idx', idx_default))
idx = max(0, min(idx, num_realizations - 1))
bd_loc_vis = np.squeeze(np.asarray(BD_loc[idx]))
if bd_loc_vis.ndim == 1:
    bd_loc_vis = bd_loc_vis[np.newaxis, :]
if bd_loc_vis.ndim != 2 or bd_loc_vis.shape[-1] != 3:
    bd_loc_vis = np.asarray(bd_loc_vis).reshape(-1, 3)
bd_locations_vis = bd_loc_vis
num_bds_vis = bd_locations_vis.shape[0]
active_bd_idx = int(active_user_test[idx])
identified_bd_idx = int(predicted_user_test[idx])
if not 0 <= active_bd_idx < num_bds_vis:
    raise ValueError(
        f"Active BD index {active_bd_idx} is outside the {num_bds_vis} saved BDs."
    )
if not 0 <= identified_bd_idx < num_bds_vis:
    raise ValueError(
        f"Identified BD index {identified_bd_idx} is outside the {num_bds_vis} saved BDs."
    )
active_bd_loc = bd_locations_vis[active_bd_idx]
identified_bd_loc = bd_locations_vis[identified_bd_idx]
identification_correct = active_bd_idx == identified_bd_idx

# Handle multiple scatterers - ensure 2D array shape (num_scatterers, 3)
scatter_entry = Scatter_loc[idx] if idx < len(Scatter_loc) else None
if scatter_entry is None:
    scatter_loc_vis = np.array([[0, 0, 0]], dtype=np.float64)
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
        scatter_loc_vis = np.array([[0, 0, 0]], dtype=np.float64)
num_scatterers_vis = scatter_loc_vis.shape[0]

# Array dimensions (for ULA, N_h = N, N_v = 1; for UPA, N_h = N_v = sqrt(N))
# With elevation = 0, effectively treating as ULA in azimuth
N_tx_h = N_tx  # Treat as horizontal array for azimuth-only pattern
N_rx_h = N_rx

def fold_ula_azimuth(angle_rad, sector_start=-np.pi / 2):
    """Fold azimuth angle(s) into ULA-equivalent sector [sector_start, sector_start + pi)."""
    return np.mod(angle_rad - sector_start, np.pi) + sector_start

# Function to compute 2D beam pattern (azimuth only, elevation = 0)
def compute_beam_pattern_2d(beamformer, N_h, array_role, num_points=360):
    """Compute a Tx or Rx response versus the physical scene bearing.

    ``generate_mimo_channel`` builds a monostatic path as
    ``a_rx(-az) @ a_tx(az).T``. Consequently, the transmit response is
    ``|a_tx(az).T w|^2`` and the receive response is
    ``|v.H a_rx(-az)|^2``. Keeping those conventions explicit prevents the
    plotted lobes from being mirrored relative to the XY scene.
    """
    if array_role not in ('tx', 'rx'):
        raise ValueError("array_role must be 'tx' or 'rx'")

    azimuth = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    pattern = np.zeros(num_points, dtype=np.float64)
    bf = beamformer.flatten()
    if bf.size != N_h:
        raise ValueError(
            f"{array_role.upper()} beamformer has {bf.size} entries; expected {N_h}."
        )
    indices = np.arange(N_h)

    for i_az in range(num_points):
        az = azimuth[i_az]
        a_tx = np.exp(1j * np.pi * indices * np.sin(az))
        if array_role == 'tx':
            response = np.dot(a_tx, bf)
        else:
            a_rx = np.conj(a_tx)
            response = np.vdot(bf, a_rx)
        pattern[i_az] = np.abs(response) ** 2

    # Normalize
    pattern = pattern / (np.max(pattern) + 1e-10)
    return azimuth, pattern

def compute_joint_beam_pattern_2d(v_beamformer, w_beamformer, num_points=360):
    """Compute ``|v.H a_rx(-az) a_tx(az).T w|^2`` over scene bearing."""
    azimuth = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    pattern = np.zeros(num_points, dtype=np.float64)

    v_vec = np.asarray(v_beamformer).reshape(-1)
    w_vec = np.asarray(w_beamformer).reshape(-1)
    rx_indices = np.arange(v_vec.size)
    tx_indices = np.arange(w_vec.size)

    for i_az, az in enumerate(azimuth):
        sin_azimuth = np.sin(az)
        a_rx = np.exp(-1j * np.pi * rx_indices * sin_azimuth)
        a_tx = np.exp(1j * np.pi * tx_indices * sin_azimuth)
        pattern[i_az] = np.abs(np.vdot(v_vec, a_rx) * np.dot(a_tx, w_vec)) ** 2

    pattern = pattern / (np.max(pattern) + 1e-10)
    return azimuth, pattern

# Get learned and optimal beamformers for this instance
w_learned_vis = w_learned[idx]  # (N_tx, 1)
v_learned_vis = v_learned[idx]  # (N_rx, 1)
w_optimal_vis = w_optimal[idx]  # (N_tx, 1)
v_optimal_vis = v_optimal[idx]  # (N_rx, 1)

# Compute 2D beam patterns
az_tx_learned, pattern_tx_learned = compute_beam_pattern_2d(w_learned_vis, N_tx_h, 'tx', num_points=360)
az_rx_learned, pattern_rx_learned = compute_beam_pattern_2d(v_learned_vis, N_rx_h, 'rx', num_points=360)
az_tx_optimal, pattern_tx_optimal = compute_beam_pattern_2d(w_optimal_vis, N_tx_h, 'tx', num_points=360)
az_rx_optimal, pattern_rx_optimal = compute_beam_pattern_2d(v_optimal_vis, N_rx_h, 'rx', num_points=360)
az_joint_learned, pattern_joint_learned = compute_joint_beam_pattern_2d(v_learned_vis, w_learned_vis, num_points=360)
az_joint_optimal, pattern_joint_optimal = compute_joint_beam_pattern_2d(v_optimal_vis, w_optimal_vis, num_points=360)

# Calculate all BD directions and fold them to the ULA-equivalent azimuth
# sector [-pi/2, pi/2). The active and identified entries index this array.
bd_azimuths = fold_ula_azimuth(np.arctan2(
    bd_locations_vis[:, 1] - location_tx[1],
    bd_locations_vis[:, 0] - location_tx[0]
))
active_bd_azimuth = bd_azimuths[active_bd_idx]
identified_bd_azimuth = bd_azimuths[identified_bd_idx]

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

def mark_bd_directions(ax, cartesian_degrees=False):
    """Mark every BD, emphasizing the active and identified devices."""
    direction_scale = 180.0 / np.pi if cartesian_degrees else 1.0
    for device_idx, direction in enumerate(bd_azimuths):
        if device_idx not in (active_bd_idx, identified_bd_idx):
            ax.axvline(
                direction * direction_scale, color='0.55', linestyle=':',
                linewidth=1.3, alpha=0.7, label=f'BD {device_idx + 1}'
            )

    active_direction = active_bd_azimuth * direction_scale
    if identification_correct:
        ax.axvline(
            active_direction, color='crimson', linestyle='--', linewidth=2.5,
            label=f'Active & identified: BD {active_bd_idx + 1}'
        )
    else:
        ax.axvline(
            active_direction, color='crimson', linestyle='--', linewidth=2.5,
            label=f'Active: BD {active_bd_idx + 1}'
        )
        ax.axvline(
            identified_bd_azimuth * direction_scale, color='deepskyblue',
            linestyle='-.', linewidth=2.5,
            label=f'Identified: BD {identified_bd_idx + 1}'
        )

# Create figure with 3 rows, 3 columns
fig = plt.figure(figsize=(18, 16))
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
    f"Instance {idx} | Active BD: {active_bd_idx + 1} | "
    f"Identified BD: {identified_bd_idx + 1} "
    f"({'correct' if identification_correct else 'incorrect'})\n"
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
ax1 = fig.add_subplot(3, 3, 1)

# Plot Tx/Rx location (co-located)
ax1.scatter(location_tx[0], location_tx[1], c='blue', marker='s', s=200, label='Tx/Rx Array', zorder=10, edgecolors='black')

# Plot all BD locations, then overlay the active and identified markers.
ax1.scatter(
    bd_locations_vis[:, 0], bd_locations_vis[:, 1], c='0.75', marker='o',
    s=150, label='Backscatter devices', zorder=7, edgecolors='0.25'
)
for device_idx, bd_location in enumerate(bd_locations_vis):
    ax1.annotate(
        f'BD {device_idx + 1}', xy=bd_location[:2], xytext=(6, 6),
        textcoords='offset points', fontsize=9
    )
ax1.scatter(
    active_bd_loc[0], active_bd_loc[1], c='crimson', marker='*', s=430,
    label=f'Active BD {active_bd_idx + 1}', zorder=12,
    edgecolors='black', linewidth=1.2
)
ax1.scatter(
    identified_bd_loc[0], identified_bd_loc[1], facecolors='none',
    edgecolors='deepskyblue', marker='o', s=580, linewidth=3,
    label=f'Identified BD {identified_bd_idx + 1}', zorder=13
)

# Plot Scatter locations (multiple scatterers)
scatter_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, num_scatterers_vis))
for s in range(num_scatterers_vis):
    label = 'Scatterers' if s == 0 else None
    ax1.scatter(scatter_loc_vis[s, 0], scatter_loc_vis[s, 1], c=[scatter_colors[s]], marker='o', s=250, 
                label=label, zorder=10, edgecolors='black', linewidth=1.5)

# Draw lines showing BD and scatter paths.
for device_idx, bd_location in enumerate(bd_locations_vis):
    if device_idx == active_bd_idx:
        color, linestyle, linewidth, label = (
            'crimson', '--', 2.4, 'Active-BD path'
        )
    elif device_idx == identified_bd_idx:
        color, linestyle, linewidth, label = (
            'deepskyblue', '-.', 2.4, 'Identified-BD path'
        )
    else:
        color, linestyle, linewidth, label = ('0.6', ':', 1.3, None)
    ax1.plot(
        [location_tx[0], bd_location[0]],
        [location_tx[1], bd_location[1]],
        color=color, linestyle=linestyle, alpha=0.75,
        linewidth=linewidth, label=label
    )
for s in range(num_scatterers_vis):
    label = 'Scatter paths' if s == 0 else None
    ax1.plot([location_tx[0], scatter_loc_vis[s, 0]], [location_tx[1], scatter_loc_vis[s, 1]], 
             color=scatter_colors[s], linestyle=':', alpha=0.6, linewidth=2, label=label)

# Add direction arrows, scaled to the generated scene dimensions.
scene_points = np.vstack([bd_locations_vis[:, :2], scatter_loc_vis[:, :2]])
scene_ranges = np.ptp(np.vstack([location_tx[:2], scene_points]), axis=0)
arrow_scale = 0.18 * max(float(np.max(scene_ranges)), np.finfo(float).eps)
dir_to_active = (
    (active_bd_loc[:2] - location_tx[:2])
    / (np.linalg.norm(active_bd_loc[:2] - location_tx[:2]) + 1e-10)
)
ax1.annotate('', xy=(location_tx[0] + dir_to_active[0]*arrow_scale, location_tx[1] + dir_to_active[1]*arrow_scale),
             xytext=(location_tx[0], location_tx[1]),
             arrowprops=dict(arrowstyle='->', color='crimson', lw=2.5))
if not identification_correct:
    dir_to_identified = (
        (identified_bd_loc[:2] - location_tx[:2])
        / (np.linalg.norm(identified_bd_loc[:2] - location_tx[:2]) + 1e-10)
    )
    ax1.annotate(
        '', xy=(
            location_tx[0] + dir_to_identified[0] * arrow_scale,
            location_tx[1] + dir_to_identified[1] * arrow_scale
        ), xytext=(location_tx[0], location_tx[1]),
        arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2.5)
    )
for s in range(num_scatterers_vis):
    dir_to_scatter = (scatter_loc_vis[s, :2] - location_tx[:2]) / (np.linalg.norm(scatter_loc_vis[s, :2] - location_tx[:2]) + 1e-10)
    ax1.annotate('', xy=(location_tx[0] + dir_to_scatter[0]*arrow_scale, location_tx[1] + dir_to_scatter[1]*arrow_scale),
                 xytext=(location_tx[0], location_tx[1]),
                 arrowprops=dict(arrowstyle='->', color=scatter_colors[s], lw=2))

ax1.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax1.set_title(
    f'2D Scene — Instance {idx} '
    f'({"correct ID" if identification_correct else "incorrect ID"})',
    fontsize=12, fontweight='bold'
)
ax1.legend(loc='best', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# Build BD/scatter azimuth text for the scene information box.
bd_az_text = '\n'.join([
    f'BD {device_idx + 1} Az: {np.degrees(direction):.1f}°'
    for device_idx, direction in enumerate(bd_azimuths)
])
scatter_az_text = '\n'.join([f'Scatter {s+1} Az: {np.degrees(scatter_azimuths[s]):.1f}°' for s in range(num_scatterers_vis)])
ax1.text(0.02, 0.98, f'{bd_az_text}\n{scatter_az_text}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ===== Row 1, Col 2: Polar Plot - Learned Tx Beam =====
ax2 = fig.add_subplot(3, 3, 2, projection='polar')

# Plot beam pattern
ax2.plot(az_tx_learned, pattern_tx_learned, 'b-', linewidth=2.5, label='Tx Beam')
ax2.fill(az_tx_learned, pattern_tx_learned, 'blue', alpha=0.2)

# Mark BD directions
mark_bd_directions(ax2)
# Mark all scatter directions
scatter_colors_polar = plt.cm.Oranges(np.linspace(0.5, 0.9, num_scatterers_vis))
for s in range(num_scatterers_vis):
    ax2.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0, 
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax2.set_title('Learned Tx Beamformer', fontsize=12, fontweight='bold', pad=15)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax2)

# ===== Row 1, Col 3: Polar Plot - Learned Rx Beam =====
ax3 = fig.add_subplot(3, 3, 3, projection='polar')

ax3.plot(az_rx_learned, pattern_rx_learned, 'g-', linewidth=2.5, label='Rx Beam')
ax3.fill(az_rx_learned, pattern_rx_learned, 'green', alpha=0.2)

mark_bd_directions(ax3)
for s in range(num_scatterers_vis):
    ax3.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0, 
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax3.set_title('Learned Rx Beamformer', fontsize=12, fontweight='bold', pad=15)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax3)

# ===== Row 2, Col 1: Learned Joint Pattern =====
ax4 = fig.add_subplot(3, 3, 4, projection='polar')

ax4.plot(az_joint_learned, pattern_joint_learned, color='purple', linewidth=2.5, label=r'$|v^H a_{\rm rx}a_{\rm tx}^T w|^2$')
ax4.fill(az_joint_learned, pattern_joint_learned, color='purple', alpha=0.2)

mark_bd_directions(ax4)
for s in range(num_scatterers_vis):
    ax4.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0,
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax4.set_title(r'Learned Joint Pattern $|v^H a_{\rm rx}a_{\rm tx}^T w|^2$', fontsize=12, fontweight='bold', pad=15)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax4)

# ===== Row 2, Col 2: Cartesian Comparison Plot =====
ax5 = fig.add_subplot(3, 3, 5)

azimuth_deg = np.degrees(az_tx_learned)

# Plot all patterns in dB
ax5.plot(azimuth_deg, 10*np.log10(pattern_tx_learned + 1e-10), 'b-', linewidth=2, label='Learned Tx', alpha=0.8)
ax5.plot(azimuth_deg, 10*np.log10(pattern_rx_learned + 1e-10), 'g-', linewidth=2, label='Learned Rx', alpha=0.8)
ax5.plot(azimuth_deg, 10*np.log10(pattern_joint_learned + 1e-10), color='purple', linewidth=2, label='Learned Joint', alpha=0.8)
ax5.plot(azimuth_deg, 10*np.log10(pattern_tx_optimal + 1e-10), 'b--', linewidth=2, label='Optimal Tx', alpha=0.8)
ax5.plot(azimuth_deg, 10*np.log10(pattern_rx_optimal + 1e-10), 'g--', linewidth=2, label='Optimal Rx', alpha=0.8)
ax5.plot(azimuth_deg, 10*np.log10(pattern_joint_optimal + 1e-10), color='purple', linestyle='--', linewidth=2, label='Optimal Joint', alpha=0.8)

# Mark BD and scatter directions
mark_bd_directions(ax5, cartesian_degrees=True)
scatter_colors_cart = plt.cm.Oranges(np.linspace(0.5, 0.9, num_scatterers_vis))
for s in range(num_scatterers_vis):
    scatter_az_deg = np.degrees(scatter_azimuths[s])
    ax5.axvline(scatter_az_deg, color=scatter_colors_cart[s], linestyle=':', linewidth=2, alpha=0.7, 
                label=f'S{s+1}: {scatter_az_deg:.1f}°')

ax5.set_xlabel('Azimuth Angle (degrees)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Normalized Gain (dB)', fontsize=11, fontweight='bold')
ax5.set_title('Beam Pattern Comparison (Elevation = 0°)', fontsize=12, fontweight='bold')
ax5.set_xlim([-90, 90])
ax5.set_ylim([-30, 5])
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
ax5.text(0.02, 0.02, f'N_tx = {N_tx}, N_rx = {N_rx}\nτ = {tau}, SNR = {snr_display} dB', 
         transform=ax5.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ===== Row 2, Col 3: Optimal Joint Pattern =====
ax6 = fig.add_subplot(3, 3, 6, projection='polar')

ax6.plot(az_joint_optimal, pattern_joint_optimal, color='purple', linewidth=2.5, label=r'$|v^H a_{\rm rx}a_{\rm tx}^T w|^2$')
ax6.fill(az_joint_optimal, pattern_joint_optimal, color='purple', alpha=0.2)

mark_bd_directions(ax6)
for s in range(num_scatterers_vis):
    ax6.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0,
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax6.set_title(r'Optimal Joint Pattern $|v^H a_{\rm rx}a_{\rm tx}^T w|^2$', fontsize=12, fontweight='bold', pad=15)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax6)

# ===== Row 3, Col 2: Polar Plot - Optimal Tx Beam =====
ax7 = fig.add_subplot(3, 3, 8, projection='polar')

ax7.plot(az_tx_optimal, pattern_tx_optimal, 'b-', linewidth=2.5, label='Tx Beam')
ax7.fill(az_tx_optimal, pattern_tx_optimal, 'blue', alpha=0.2)

mark_bd_directions(ax7)
for s in range(num_scatterers_vis):
    ax7.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0, 
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax7.set_title('Optimal Tx Beamformer', fontsize=12, fontweight='bold', pad=15)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax7)

# ===== Row 3, Col 3: Polar Plot - Optimal Rx Beam =====
ax8 = fig.add_subplot(3, 3, 9, projection='polar')

ax8.plot(az_rx_optimal, pattern_rx_optimal, 'g-', linewidth=2.5, label='Rx Beam')
ax8.fill(az_rx_optimal, pattern_rx_optimal, 'green', alpha=0.2)

mark_bd_directions(ax8)
for s in range(num_scatterers_vis):
    ax8.axvline(scatter_azimuths[s], color=scatter_colors_polar[s], linestyle=':', linewidth=2.0, 
                label=f'S{s+1}: {np.degrees(scatter_azimuths[s]):.1f}°')

ax8.set_title('Optimal Rx Beamformer', fontsize=12, fontweight='bold', pad=15)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
configure_upper_half_polar_axis(ax8)

# Keep the center-bottom subplot empty to avoid over-crowding the figure.
ax9 = fig.add_subplot(3, 3, 7)
ax9.axis('off')

plt.tight_layout()

# Save figure
# fig_filename = os.path.join(drive_save_path, f'beam_pattern_2d_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}.png')
# plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
# print(f"Beam pattern figure saved to {fig_filename}")

plt.show()


# %%
