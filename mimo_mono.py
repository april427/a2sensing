"""
Bi-Static MIMO with BD SINR Maximization Objective

There exist other static scatters. 
BD alters its states to distinguish itself from other reflectors.

Objective: max_v SINR_BD = P|w^H H_b v|^2 / (P|w^H (H_d+H_r) v|^2 + noise_var)

Architecture: RIS and Rx are separate nodes but share observations via a single LSTM.
- RIS only trains transmit beamformer w
- Rx only trains receive beamformer v
- Both nodes can see each other's results through shared hidden state
  (similar to 2b.py where a single node trains both beamformers)

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
location_rx = np.array([0, 0, 0])                    # Tx and Rx are co-located
    # Antenna array, square array 
N_tx = args.N_ris 
N_rx = args.N_ris
N_ris = N_rx       # Numbers of antennas are all the same
num_users = 1
num_scatters = 1  # excluding BD
Rician_factor = args.rician_factor
location_bd = None

# Sensing parameters
tau = 32  # Pilot length (also number of BD interactions)
K = 5  # Number of OFDM symbols per BD state
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


#####################################################
# Learning parameters and Computation Graph
#####################################################
#  
initial_run = 1
n_epochs = 50
learning_rate = 5e-4
batch_per_epoch = 5
batch_size_order = 8
val_size_order = 20
test_size = 200

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Placeholders
loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_users), name="loc_input")
scatter_loc_input = tf.placeholder(tf.float32, shape=(None, 3, num_scatters), name="scatter_loc_input")
H_d_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_d")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_b")
H_r_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_tx), name="H_r")
s_placeholder = tf.placeholder(tf.complex64, shape=(None, 1, num_subcarriers), name="ambient_signal")

##################### ACTIVE SENSING NETWORK #####################

with tf.name_scope("system_parameters"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    bd_seq = tf.constant(BD_modulation.astype(np.float32), dtype=tf.float32)

with tf.name_scope("active_sensing_agent"):
    hidden_size = 256  # Shared hidden size for both nodes
    
    # Single shared LSTM cell - both RIS and Rx can see the results
    # Similar to 2b.py where a single node trains both beamformers
    # LSTM_shared = LSTM_Cell(hidden_size, name='LSTM_shared')
    LSTM1 = LSTM_Cell(hidden_size, name='LSTM_1')
    LSTM2 = LSTM_Cell(hidden_size, name='LSTM_2')
    
    
    # RIS only trains transmit beamformer w
    mlp_ris_tx = MLPBlock(3, [hidden_size * 2, hidden_size * 2, 2 * N_tx], name='RIS_transmitter')
    # Rx only trains receive beamformer v  
    mlp_rx_rx = MLPBlock(3, [hidden_size * 2, hidden_size * 2, 2 * N_rx], name='Receiver_receiver')
    
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
        
        'Construct effective channel H(t) = x_BD[t] * H_b + H_d + H_r'
        # H_d, H_r: (batch, N_rx, N_tx), H_b: (batch, N_rx, N_tx)
        # x_BD[t] is scalar (+1 or -1)
        x_bd_t = tf.reshape(tf.cast(x_BD[t], tf.complex64), [-1, 1, 1])  # (batch, 1, 1)
        H_eff = x_bd_t * H_b_placeholder + H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
        
        'Observe received signal: y = sqrt(P) * v^H * H * w + noise'
        y_noiseless1 = tf.complex(tf.sqrt(lay['P']), 0.0) *  tf.matmul(H_eff, w1)  # (batch, N_rx, 1)
        # Broadcast to K samples
        y_noiseless1 = tf.tile(y_noiseless1, [1, 1, K])  # (batch, N_rx, K)
        # Add noise
        noise = tf.complex(
            tf.random_normal([batch_size, N_rx, K], mean=0.0, stddev=noiseSTD_per_dim),
            tf.random_normal([batch_size, N_rx, K], mean=0.0, stddev=noiseSTD_per_dim)
        )

        y_complex1 = tf.add(y_noiseless1, noise)  # (batch, N_rx, K) Before beamforming

        # y_complex2 = tf.matmul(tf.conj(tf.transpose(v1, perm=[0, 2, 1])), y_complex1)  # (batch, 1, K)
        # y_complex2 = tf.reduce_mean(tf.reshape(y_complex2, [batch_size, K]), axis=1, keepdims=True)  # (batch, 1)
        
        # y_complex1_flat = tf.reshape(tf.reduce_mean(y_complex1, axis=2), [batch_size, N_rx])  # (batch, N_rx)
        # 'Prepare features for shared LSTM'
        # # Combine features: [Re(y1), Im(y1), Re(y2), Im(y2),x_BD[t], snr]
        # y_real = tf.concat([
        #     tf.cast(tf.real(y_complex1_flat), tf.float32),
        #     tf.cast(tf.imag(y_complex1_flat), tf.float32),
        #     tf.cast(tf.real(y_complex2), tf.float32),
        #     tf.cast(tf.imag(y_complex2), tf.float32)
        # ], axis=1)  # (batch, 2*N_rx + 2)

        Y1 = Y1 + tf.reduce_mean(x_bd_t * y_complex1, axis=2, keepdims=False)
        Y2 = Y2 + tf.reduce_mean(y_complex1, axis=2, keepdims=False)  # Accumulate over time steps
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
        
        'RIS designs transmit beamformer w based on shared hidden state'
        w_her = mlp_ris_tx(tf.concat([h_old, h_old2], axis=1))
        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm + 1e-8)
        w1 = tf.complex(w_her[:, 0:N_tx], w_her[:, N_tx:2 * N_tx])
        w1 = tf.reshape(w1, [-1, N_tx, 1])
        
        'Rx designs receive beamformer v based on shared hidden state'
        v_her = mlp_rx_rx(tf.concat([h_old, h_old2], axis=1))
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
    MLP_bf_w = MLPBlock(3, [2 * hidden_size, 2 * hidden_size, 2 * N_tx], name='MLP_bf_w')
    MLP_bf_v = MLPBlock(3, [2 * hidden_size, 2 * hidden_size, 2 * N_rx], name='MLP_bf_v')
    
    # Final RIS transmit beamformer w (from shared state)
    w_tmp = MLP_bf_w(tf.concat([c_old, c_old2], axis=1))
    w_norm = tf.reshape(tf.norm(w_tmp, axis=1), (-1, 1))
    w_tmp = tf.divide(w_tmp, w_norm + 1e-8)
    w_complex = tf.complex(w_tmp[:, 0:N_tx], w_tmp[:, N_tx:2 * N_tx])
    w_complex = tf.reshape(w_complex, [-1, N_tx, 1])
    
    # Final Rx receive beamformer v (from shared state)
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
    sig_BD = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_b_placeholder, w_complex))  # (batch, 1, 1)
    sig_BD = tf.squeeze(tf.abs(sig_BD) ** 2) * lay['P']  # (batch,)
    
    # Interference channel: H_d + H_r
    H_interference = H_d_placeholder + H_r_placeholder  # (batch, N_rx, N_tx)
    
    # Interference power: P * |v^H (H_d + H_r) w|^2
    sig_int = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_interference, w_complex))  # (batch, 1, 1)
    sig_int = tf.squeeze(tf.abs(sig_int) ** 2) * lay['P']  # (batch,)
    
    sinr_BD = sig_BD / (sig_int + noise_var + 1e-10)
    sinr_BD_clipped = tf.clip_by_value(sinr_BD, 1e-4, 1e4)
    
    # Log SINR with per-sample clipping to prevent outliers from dominating
    log_sinr_BD_raw = tf.log(sinr_BD_clipped + 1e-9)
    log_sinr_BD = tf.clip_by_value(log_sinr_BD_raw, -8.0, 8.0)  # ~±35 dB range
    
    # For backward compatibility, define sig_ref as interference
    sig_ref = sig_int

    # Calculate the SINR towards the scatters
    sig_scatter = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_r_placeholder, w_complex))  # (batch, 1, 1)
    int_scatter = tf.matmul(tf.linalg.adjoint(v_complex), \
                            tf.matmul(H_d_placeholder+H_b_placeholder, w_complex))  # (batch, 1, 1)
    sinr_scatter = tf.squeeze(tf.abs(sig_scatter) ** 2) * lay['P'] / \
                    (tf.squeeze(tf.abs(int_scatter) ** 2) * lay['P'] + noise_var + 1e-10)
    
    # Compute sigma1_sq and sigma2_sq for BD state detection metrics
    # g1 = -H_b + H_d + H_r (BD = -1), g2 = H_b + H_d + H_r (BD = +1)
    # H_g1 = -H_b_placeholder + H_d_placeholder + H_r_placeholder
    # H_g2 = H_b_placeholder + H_d_placeholder + H_r_placeholder
    
    # sigma1_sq = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_g1, w_complex))
    # sigma2_sq = tf.matmul(tf.linalg.adjoint(v_complex), tf.matmul(H_g2, w_complex))
    # sigma1_sq = tf.squeeze(tf.abs(sigma1_sq) ** 2) * lay['P'] + noise_var
    # sigma2_sq = tf.squeeze(tf.abs(sigma2_sq) ** 2) * lay['P'] + noise_var
    
    # # Threshold and error probability (for evaluation)
    # Th = sigma1_sq * sigma2_sq / (sigma1_sq - sigma2_sq + 1e-10) * tf.log(sigma1_sq / (sigma2_sq + 1e-10) + 1e-10)
    # pe = tf.where(
    #     sigma1_sq > sigma2_sq,
    #     0.5 + 0.5 * tf.exp(-Th / (sigma2_sq + 1e-10)) - 0.5 * tf.exp(-Th / (sigma1_sq + 1e-10)),
    #     0.5 + 0.5 * tf.exp(-Th / (sigma1_sq + 1e-10)) - 0.5 * tf.exp(-Th / (sigma2_sq + 1e-10))
    # )


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


#####################################################
# Data Generation
#####################################################

# Validation set
num_val_samples = val_size_order * 32

# Generate channels for validation
H_d_val_list = []
H_b_val_list = []
H_r_val_list = []
set_location_user_val = []

for ii in range(num_val_samples):
    # Generate random locations
    bd_loc = generate_location_mimo(1)[0]
    scatter_loc = generate_location_mimo(1)[0]
    
    # Generate MIMO channels using generate_mimo_channel
    # Returns: H_total, H_direct, H_scatter, H_bd
    _, H_d, H_r, H_b = generate_mimo_channel(
        location_tx, location_rx, scatter_loc, bd_loc,
        N_tx_h=int(np.sqrt(N_tx)), N_tx_v=int(np.sqrt(N_tx)),
        N_rx_h=int(np.sqrt(N_rx)), N_rx_v=int(np.sqrt(N_rx)),
        Rician_factor=Rician_factor
    )
    
    H_d_val_list.append(H_d)
    H_b_val_list.append(H_b)
    H_r_val_list.append(H_r)
    
    # Store location info (azimuth, distance info)
    d_bd_rx = np.linalg.norm(bd_loc - location_rx)
    d_bd_tx = np.linalg.norm(bd_loc - location_tx)
    azimuth_bd = np.arctan2(bd_loc[1] - location_rx[1], bd_loc[0] - location_rx[0])
    set_location_user_val.append(np.array([azimuth_bd, d_bd_rx, d_bd_tx])[:, np.newaxis])

H_b_val = np.array(H_b_val_list)
H_d_val = np.array(H_d_val_list)  # Use ALL per-sample direct channels
H_r_val = np.array(H_r_val_list)  # Use ALL per-sample scatter channels

# QPSK ambient signal
qpsk_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
s_signal_val = qpsk_symbols[np.random.randint(0, 4, size=(val_size_order * 32, 1, num_subcarriers))]

# Prepare batch-sized channel matrices
H_d_val_batch = H_d_val  # Already (num_val_samples, N_rx, N_tx)
H_r_val_batch = H_r_val  # Already (num_val_samples, N_rx, N_tx)

feed_dict_val = {
    loc_input: np.array(set_location_user_val),
    lay['P']: Pvec[0],
    H_d_placeholder: H_d_val_batch,
    H_b_placeholder: H_b_val,
    H_r_placeholder: H_r_val_batch,
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
        
        # QPSK ambient signal
        s_signal = qpsk_symbols[np.random.randint(0, 4, size=(batch_size_order * 32, 1, num_subcarriers))]
        
        for rnd_indices in range(batch_per_epoch):
            # Generate training batch
            num_train_samples = batch_size_order * 32
            
            H_d_train_list = []
            H_b_train_list = []
            H_r_train_list = []
            set_location_user_train = []
            
            for ii in range(num_train_samples):
                # Generate random locations
                bd_loc = generate_location_mimo(1)[0]
                scatter_loc = generate_location_mimo(1)[0]
                
                # Generate MIMO channels
                _, H_d, H_r, H_b = generate_mimo_channel(
                    location_tx, location_rx, scatter_loc, bd_loc,
                    N_tx_h=int(np.sqrt(N_tx)), N_tx_v=int(np.sqrt(N_tx)),
                    N_rx_h=int(np.sqrt(N_rx)), N_rx_v=int(np.sqrt(N_rx)),
                    Rician_factor=Rician_factor
                )
                
                H_d_train_list.append(H_d)
                H_b_train_list.append(H_b)
                H_r_train_list.append(H_r)
                
                # Store location info
                d_bd_rx = np.linalg.norm(bd_loc - location_rx)
                d_bd_tx = np.linalg.norm(bd_loc - location_tx)
                azimuth_bd = np.arctan2(bd_loc[1] - location_rx[1], bd_loc[0] - location_rx[0])
                set_location_user_train.append(np.array([azimuth_bd, d_bd_rx, d_bd_tx])[:, np.newaxis])
            
            H_b_train = np.array(H_b_train_list)
            H_d_train = np.array(H_d_train_list)  # Per-sample
            H_r_train = np.array(H_r_train_list)  # Per-sample

            H_d_train_batch = H_d_train  # Already correct shape
            H_r_train_batch = H_r_train  # Already correct shape

            feed_dict_batch = {
                loc_input: np.array(set_location_user_train),
                lay['P']: Pvec[0],
                H_d_placeholder: H_d_train_batch,
                H_b_placeholder: H_b_train,
                H_r_placeholder: H_r_train_batch,
                s_placeholder: s_signal
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
        
        loss_val, sinr_val, sinr_opt_val, sinr_scatter_val, \
            sig_bd_val, sig_ref_val, sig_bd_opt_val, sig_int_opt_val = sess.run(
            [loss, sinr_BD, sinr_BD_opt, sinr_scatter, \
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
              f'SINR_scatter: {10 * np.log10(np.mean(sinr_scatter_val) + 1e-10):6.2f} dB')
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
    H_d_test_list = []
    H_b_test_list = []
    H_r_test_list = []
    set_location_user_test = []
    BD_loc = []
    Scatter_loc = []
    
    for ii in range(test_size):
        # Generate random locations
        bd_loc = generate_location_mimo(1)[0]
        scatter_loc = generate_location_mimo(1)[0]

        BD_loc.append(bd_loc)
        Scatter_loc.append(scatter_loc)
        
        # Generate MIMO channels
        _, H_d, H_r, H_b = generate_mimo_channel(
            location_tx, location_rx, scatter_loc, bd_loc,
            N_tx_h=int(np.sqrt(N_tx)), N_tx_v=int(np.sqrt(N_tx)),
            N_rx_h=int(np.sqrt(N_rx)), N_rx_v=int(np.sqrt(N_rx)),
            Rician_factor=Rician_factor
        )
        
        H_d_test_list.append(H_d)
        H_b_test_list.append(H_b)
        H_r_test_list.append(H_r)
        
        # Store location info
        d_bd_rx = np.linalg.norm(bd_loc - location_rx)
        d_bd_tx = np.linalg.norm(bd_loc - location_tx)
        azimuth_bd = np.arctan2(bd_loc[1] - location_rx[1], bd_loc[0] - location_rx[0])
        set_location_user_test.append(np.array([azimuth_bd, d_bd_rx, d_bd_tx])[:, np.newaxis])
    
    H_b_test = np.array(H_b_test_list)
    H_d_test = np.array(H_d_test_list)  # Per-sample
    H_r_test = np.array(H_r_test_list)  # Per-sample

    H_d_test_batch = H_d_test
    H_r_test_batch = H_r_test
    
    s_signal_test = qpsk_symbols[np.random.randint(0, 4, size=(test_size, 1, num_subcarriers))]
    
    feed_dict_test = {
        loc_input: np.array(set_location_user_test),
        lay['P']: Pvec[0],
        H_d_placeholder: H_d_test_batch,
        H_b_placeholder: H_b_test,
        H_r_placeholder: H_r_test_batch,
        s_placeholder: s_signal_test
    }
    
    sinr_test, sinr_opt_test, sinr_scatter_test, v_learned, w_learned, v_optimal, w_optimal = sess.run(
        [sinr_BD, sinr_BD_opt, sinr_scatter, v_complex, w_complex, v_opt, w_opt], feed_dict=feed_dict_test
    )
    
    print(f"Test SINR_BD (learned):  {10 * np.log10(np.mean(sinr_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_BD (optimal):  {10 * np.log10(np.mean(sinr_opt_test) + 1e-10):6.2f} dB")
    print(f"Test SINR_scatter (learned):       {10 * np.log10(np.mean(sinr_scatter_test) + 1e-10):6.2f} dB")
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
        Scatter_location=Scatter_loc if Scatter_loc[0] is not None else [],
        sinr_learned=sinr_test,
        sinr_optimal=sinr_opt_test,
        sinr_scatter_learned=sinr_scatter_test,
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
scatter_loc_vis = Scatter_loc[idx] if Scatter_loc[idx] is not None else np.array([0, 0, 0])

# Reshape beamformers to 2D arrays for UPA
N_tx_h = int(np.sqrt(N_tx))
N_tx_v = int(np.sqrt(N_tx))
N_rx_h = int(np.sqrt(N_rx))
N_rx_v = int(np.sqrt(N_rx))

# Function to compute beam pattern for UPA
def compute_beam_pattern(beamformer, N_h, N_v, wavelength, num_points=100):
    """Compute 3D beam pattern for a UPA beamformer
    
    The steering vector convention must match generate_upa_steering_vector in channel_functions.py:
    - i1 = index % N_h (horizontal indices)
    - i2 = index // N_h (vertical indices)
    - steering_vector = exp(1j * pi * (i1 * sin(az) * cos(el) + i2 * sin(el)))
    """
    azimuth = np.linspace(-np.pi, np.pi, num_points)
    elevation = np.linspace(-np.pi/2, np.pi/2, num_points)
    AZ, EL = np.meshgrid(azimuth, elevation)
    
    pattern = np.zeros_like(AZ, dtype=np.float64)  # Use float, not complex
    bf = beamformer.flatten()
    
    N_total = N_h * N_v
    # Match the indexing convention from channel_functions.py
    i1 = np.mod(np.arange(N_total), N_h)  # Horizontal indices
    i2 = np.floor(np.arange(N_total) / N_h)  # Vertical indices
    
    for i_el in range(num_points):
        for i_az in range(num_points):
            az = azimuth[i_az]
            el = elevation[i_el]
            
            sin_azimuth = np.sin(az)
            cos_elevation = np.cos(el)
            sin_elevation = np.sin(el)
            
            # Steering vector matching channel_functions.py convention
            a = np.exp(1j * np.pi * (i1 * sin_azimuth * cos_elevation + i2 * sin_elevation))
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

# Create figure with 2 rows, 4 columns (3D Tx beam, 3D Rx beam, azimuth cut, elevation cut)
fig = plt.figure(figsize=(32, 12))

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
# ===== Row 1, Col 1: 3D Scene with Learned Tx Beam Pattern =====
ax1 = fig.add_subplot(2, 4, 1, projection='3d')

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

# ===== Add Learned Tx Beam Pattern to Scene =====
beam_scale = 5  # Scale factor for beam visualization
X_tx_learned, Y_tx_learned, Z_tx_learned = spherical_to_cartesian(AZ_tx_learned, EL_tx_learned, pattern_tx_learned * beam_scale)
X_tx_learned += location_tx[0]
Y_tx_learned += location_tx[1]
Z_tx_learned += location_tx[2]

# Plot Tx beam pattern with semi-transparent surface
surf_tx = ax1.plot_surface(X_tx_learned, Y_tx_learned, Z_tx_learned, cmap='Blues', alpha=0.4, 
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# Mark direction to BD from Tx
dir_to_bd = (bd_loc_vis - location_tx) / np.linalg.norm(bd_loc_vis - location_tx)
ax1.quiver(location_tx[0], location_tx[1], location_tx[2], 
            dir_to_bd[0]*4, dir_to_bd[1]*4, dir_to_bd[2]*4, 
            color='darkblue', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

ax1.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
ax1.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
ax1.set_title('Learned Tx Beamformer - 3D Beam Pattern', fontsize=11, fontweight='bold')
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

# ===== Row 1, Col 2: 3D Scene with Learned Rx Beam Pattern =====
ax1b = fig.add_subplot(2, 4, 2, projection='3d')

# Plot Tx array
ax1b.scatter(tx_antenna_pos[:, 0], tx_antenna_pos[:, 1], tx_antenna_pos[:, 2], 
            c='blue', marker='s', s=100, label='Tx Array', alpha=0.8, zorder=10)

# Plot Rx array
ax1b.scatter(rx_antenna_pos[:, 0], rx_antenna_pos[:, 1], rx_antenna_pos[:, 2], 
            c='green', marker='^', s=100, label='Rx Array', alpha=0.8, zorder=10)

# Plot BD location
ax1b.scatter(bd_loc_vis[0], bd_loc_vis[1], bd_loc_vis[2], 
            c='red', marker='*', s=400, label='BD', edgecolors='black', linewidth=2, zorder=10)

# Plot Scatter location
ax1b.scatter(scatter_loc_vis[0], scatter_loc_vis[1], scatter_loc_vis[2], 
            c='orange', marker='o', s=250, label='Scatterer', edgecolors='black', linewidth=1.5, zorder=10)

# ===== Add Learned Rx Beam Pattern to Scene =====
X_rx_learned, Y_rx_learned, Z_rx_learned = spherical_to_cartesian(AZ_rx_learned, EL_rx_learned, pattern_rx_learned * beam_scale)
X_rx_learned += location_rx[0]
Y_rx_learned += location_rx[1]
Z_rx_learned += location_rx[2]

# Plot Rx beam pattern with semi-transparent surface
surf_rx = ax1b.plot_surface(X_rx_learned, Y_rx_learned, Z_rx_learned, cmap='Greens', alpha=0.4,
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# Mark direction to BD from Rx
dir_to_bd_rx = (bd_loc_vis - location_rx) / np.linalg.norm(bd_loc_vis - location_rx)
ax1b.quiver(location_rx[0], location_rx[1], location_rx[2], 
            dir_to_bd_rx[0]*4, dir_to_bd_rx[1]*4, dir_to_bd_rx[2]*4, 
            color='darkgreen', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

ax1b.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax1b.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
ax1b.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
ax1b.set_title('Learned Rx Beamformer - 3D Beam Pattern', fontsize=11, fontweight='bold')
ax1b.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Add coordinate axis arrows at origin for reference
ax1b.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax1b.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax1b.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax1b.text(axis_length*1.1, 0, 0, 'X', color='red', fontsize=10, fontweight='bold')
ax1b.text(0, axis_length*1.1, 0, 'Y', color='green', fontsize=10, fontweight='bold')
ax1b.text(0, 0, axis_length*1.1, 'Z', color='blue', fontsize=10, fontweight='bold')

# Set better viewing angle
ax1b.view_init(elev=20, azim=45)

# ===== Row 1, Col 3: 2D Beam Pattern for Learned Beamformers (Azimuth Cut) =====
ax2 = fig.add_subplot(2, 4, 3)

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
# sc_azimuth_rx = np.arctan2(scatter_loc_vis[1] - location_rx[1], scatter_loc_vis[0] - location_rx[0]) * 180/np.pi
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

# ===== Row 1, Col 4: 2D Beam Pattern for Learned Beamformers (Elevation Cut) =====
ax2b = fig.add_subplot(2, 4, 4)

# Take a vertical cut at the BD azimuth direction
# Find the azimuth index closest to BD direction
bd_az_idx = int((bd_azimuth_tx + 180) / 360 * pattern_tx_learned.shape[1])
bd_az_idx = np.clip(bd_az_idx, 0, pattern_tx_learned.shape[1] - 1)

elevation_deg = np.linspace(-90, 90, pattern_tx_learned.shape[0])

ax2b.plot(elevation_deg, 10*np.log10(pattern_tx_learned[:, bd_az_idx] + 1e-10), 'b-', 
            linewidth=2.5, label='Tx Beam', alpha=0.8)
ax2b.plot(elevation_deg, 10*np.log10(pattern_rx_learned[:, bd_az_idx] + 1e-10), 'g-', 
            linewidth=2.5, label='Rx Beam', alpha=0.8)

# Mark BD elevation direction
bd_dist_xy_tx = np.sqrt((bd_loc_vis[0] - location_tx[0])**2 + (bd_loc_vis[1] - location_tx[1])**2)
bd_elevation_tx = np.arctan2(bd_loc_vis[2] - location_tx[2], bd_dist_xy_tx) * 180/np.pi
bd_dist_xy_rx = np.sqrt((bd_loc_vis[0] - location_rx[0])**2 + (bd_loc_vis[1] - location_rx[1])**2)
bd_elevation_rx = np.arctan2(bd_loc_vis[2] - location_rx[2], bd_dist_xy_rx) * 180/np.pi

ax2b.axvline(bd_elevation_tx, color='darkblue', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Tx: {bd_elevation_tx:.1f}°')
ax2b.axvline(bd_elevation_rx, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Rx: {bd_elevation_rx:.1f}°')

# Mark scatter elevation direction
sc_dist_xy_tx = np.sqrt((scatter_loc_vis[0] - location_tx[0])**2 + (scatter_loc_vis[1] - location_tx[1])**2)
sc_elevation_tx = np.arctan2(scatter_loc_vis[2] - location_tx[2], sc_dist_xy_tx) * 180/np.pi
ax2b.axvline(sc_elevation_tx, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Scatter from Tx: {sc_elevation_tx:.1f}°')

ax2b.set_xlabel('Elevation Angle (degrees)', fontsize=10, fontweight='bold')
ax2b.set_ylabel('Normalized Gain (dB)', fontsize=10, fontweight='bold')
ax2b.set_title(f'Learned Beamformers - Elevation Cut (Az = {bd_azimuth_tx:.1f}°)', fontsize=11, fontweight='bold')
ax2b.set_xlim([-90, 90])
ax2b.set_ylim([-30, 5])
ax2b.grid(True, alpha=0.3, linestyle='--')
ax2b.legend(loc='upper right', fontsize=8, framealpha=0.9)

# ===== ROW 2: OPTIMAL BEAMFORMERS =====
# ===== Row 2, Col 1: 3D Scene with Optimal Tx Beam Pattern =====
ax3 = fig.add_subplot(2, 4, 5, projection='3d')

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

# ===== Add Optimal Tx Beam Pattern to Scene =====
X_tx_optimal, Y_tx_optimal, Z_tx_optimal = spherical_to_cartesian(AZ_tx_optimal, EL_tx_optimal, pattern_tx_optimal * beam_scale)
X_tx_optimal += location_tx[0]
Y_tx_optimal += location_tx[1]
Z_tx_optimal += location_tx[2]

# Plot Tx beam pattern with semi-transparent surface
surf_tx_opt = ax3.plot_surface(X_tx_optimal, Y_tx_optimal, Z_tx_optimal, cmap='Blues', alpha=0.4, 
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# Mark direction to BD from Tx
dir_to_bd = (bd_loc_vis - location_tx) / np.linalg.norm(bd_loc_vis - location_tx)
ax3.quiver(location_tx[0], location_tx[1], location_tx[2], 
            dir_to_bd[0]*4, dir_to_bd[1]*4, dir_to_bd[2]*4, 
            color='darkblue', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

ax3.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
ax3.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
ax3.set_title('Optimal Tx Beamformer - 3D Beam Pattern', fontsize=11, fontweight='bold')
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

# ===== Row 2, Col 2: 3D Scene with Optimal Rx Beam Pattern =====
ax3b = fig.add_subplot(2, 4, 6, projection='3d')

# Plot Tx array
ax3b.scatter(tx_antenna_pos[:, 0], tx_antenna_pos[:, 1], tx_antenna_pos[:, 2], 
            c='blue', marker='s', s=100, label='Tx Array', alpha=0.8, zorder=10)

# Plot Rx array
ax3b.scatter(rx_antenna_pos[:, 0], rx_antenna_pos[:, 1], rx_antenna_pos[:, 2], 
            c='green', marker='^', s=100, label='Rx Array', alpha=0.8, zorder=10)

# Plot BD location
ax3b.scatter(bd_loc_vis[0], bd_loc_vis[1], bd_loc_vis[2], 
            c='red', marker='*', s=400, label='BD', edgecolors='black', linewidth=2, zorder=10)

# Plot Scatter location
ax3b.scatter(scatter_loc_vis[0], scatter_loc_vis[1], scatter_loc_vis[2], 
            c='orange', marker='o', s=250, label='Scatterer', edgecolors='black', linewidth=1.5, zorder=10)

# Draw lines showing signal paths
ax3b.plot([location_tx[0], bd_loc_vis[0]], [location_tx[1], bd_loc_vis[1]], 
            [location_tx[2], bd_loc_vis[2]], 'r--', alpha=0.6, linewidth=2, label='Tx-BD path', zorder=5)
ax3b.plot([bd_loc_vis[0], location_rx[0]], [bd_loc_vis[1], location_rx[1]], 
            [bd_loc_vis[2], location_rx[2]], 'r--', alpha=0.6, linewidth=2, zorder=5)
ax3b.plot([location_tx[0], scatter_loc_vis[0]], [location_tx[1], scatter_loc_vis[1]], 
            [location_tx[2], scatter_loc_vis[2]], 'orange', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)
ax3b.plot([scatter_loc_vis[0], location_rx[0]], [scatter_loc_vis[1], location_rx[1]], 
            [scatter_loc_vis[2], location_rx[2]], 'orange', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)

# ===== Add Optimal Rx Beam Pattern to Scene =====
X_rx_optimal, Y_rx_optimal, Z_rx_optimal = spherical_to_cartesian(AZ_rx_optimal, EL_rx_optimal, pattern_rx_optimal * beam_scale)
X_rx_optimal += location_rx[0]
Y_rx_optimal += location_rx[1]
Z_rx_optimal += location_rx[2]

# Plot Rx beam pattern with semi-transparent surface
surf_rx_opt = ax3b.plot_surface(X_rx_optimal, Y_rx_optimal, Z_rx_optimal, cmap='Greens', alpha=0.4,
                            rstride=2, cstride=2, linewidth=0, zorder=1)

# Mark direction to BD from Rx
dir_to_bd_rx = (bd_loc_vis - location_rx) / np.linalg.norm(bd_loc_vis - location_rx)
ax3b.quiver(location_rx[0], location_rx[1], location_rx[2], 
            dir_to_bd_rx[0]*4, dir_to_bd_rx[1]*4, dir_to_bd_rx[2]*4, 
            color='darkgreen', arrow_length_ratio=0.15, linewidth=3, alpha=0.8, zorder=8)

ax3b.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax3b.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
ax3b.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
ax3b.set_title('Optimal Rx Beamformer - 3D Beam Pattern', fontsize=11, fontweight='bold')
ax3b.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Add coordinate axis arrows at origin for reference
ax3b.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax3b.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax3b.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
ax3b.text(axis_length*1.1, 0, 0, 'X', color='red', fontsize=10, fontweight='bold')
ax3b.text(0, axis_length*1.1, 0, 'Y', color='green', fontsize=10, fontweight='bold')
ax3b.text(0, 0, axis_length*1.1, 'Z', color='blue', fontsize=10, fontweight='bold')

# Set better viewing angle
ax3b.view_init(elev=20, azim=45)

# ===== Row 2, Col 3: 2D Beam Pattern for Optimal Beamformers (Azimuth Cut) =====
ax4 = fig.add_subplot(2, 4, 7)

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
# sc_azimuth_rx = np.arctan2(scatter_loc_vis[1] - location_rx[1], scatter_loc_vis[0] - location_rx[0]) * 180/np.pi
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

# ===== Row 2, Col 4: 2D Beam Pattern for Optimal Beamformers (Elevation Cut) =====
ax4b = fig.add_subplot(2, 4, 8)

# Take a vertical cut at the BD azimuth direction
bd_az_idx_opt = int((bd_azimuth_tx + 180) / 360 * pattern_tx_optimal.shape[1])
bd_az_idx_opt = np.clip(bd_az_idx_opt, 0, pattern_tx_optimal.shape[1] - 1)

elevation_deg = np.linspace(-90, 90, pattern_tx_optimal.shape[0])

ax4b.plot(elevation_deg, 10*np.log10(pattern_tx_optimal[:, bd_az_idx_opt] + 1e-10), 'b-', 
            linewidth=2.5, label='Tx Beam', alpha=0.8)
ax4b.plot(elevation_deg, 10*np.log10(pattern_rx_optimal[:, bd_az_idx_opt] + 1e-10), 'g-', 
            linewidth=2.5, label='Rx Beam', alpha=0.8)

# Mark BD elevation direction
ax4b.axvline(bd_elevation_tx, color='darkblue', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Tx: {bd_elevation_tx:.1f}°')
ax4b.axvline(bd_elevation_rx, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'BD from Rx: {bd_elevation_rx:.1f}°')

# Mark scatter elevation direction
ax4b.axvline(sc_elevation_tx, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Scatter from Tx: {sc_elevation_tx:.1f}°')

ax4b.set_xlabel('Elevation Angle (degrees)', fontsize=10, fontweight='bold')
ax4b.set_ylabel('Normalized Gain (dB)', fontsize=10, fontweight='bold')
ax4b.set_title(f'Optimal Beamformers - Elevation Cut (Az = {bd_azimuth_tx:.1f}°)', fontsize=11, fontweight='bold')
ax4b.set_xlim([-90, 90])
ax4b.set_ylim([-30, 5])
ax4b.grid(True, alpha=0.3, linestyle='--')
ax4b.legend(loc='upper right', fontsize=8, framealpha=0.9)

plt.tight_layout()

# Save figure
# fig_filename = os.path.join(drive_save_path, f'beam_pattern_N_{N_tx}_{N_rx}_tau_{tau}_snr_{int(snr_const[0])}.png')
# plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
# print(f"Beam pattern figure saved to {fig_filename}")

plt.show()

# %%
