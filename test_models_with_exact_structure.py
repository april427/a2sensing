#!/usr/bin/env python3
"""
Test script that recreates the exact network structures for model restoration
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Dense
from manifold_optimization import solve_x_equals_y_fast
from parse_args import parse_args
from channel_functions import *

# Configure GPU for M4 Mac
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU memory growth enabled for: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Parse arguments
args = parse_args()

# Constants from original code
Wavelength = 3e8 / 10e9  # wavelength
N_ris = args.N_ris
N = N_ris  # Number of antennas same as N_ris 
Rician_factor = args.rician_factor
drive_save_path = 'BD_beamInit'  # Will be changed per model

phi_min = -60*(np.pi/180)
phi_max = 60*(np.pi/180)
y_wall = 55.5

location_ris_1 = np.array([0, 0, -20])
num_ris = 1

mean_true_alpha = 0.0 + 0.0j
std_per_dim_alpha = np.sqrt(0.5) 
noiseSTD_per_dim = np.sqrt(0.5)

N_ris = args.N_ris
num_users = 1
params_system = (N_ris, N_ris, num_users)
Rician_factor = args.rician_factor
location_user = None

tau = args.tau
snr_const = args.snr
snr_const = np.array([snr_const]) 
ref_dis = 5
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2

initial_run = 1
learning_rate = 1e-3
batch_per_epoch = 256
batch_size_order = 16
val_size_order = 10
scale_factor = 1
test_size = 1000

USE_FFT = False
LOS_weight = 1
NLOS_weight = 0


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


class RNN_2b(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name):
        super(RNN_2b, self).__init__()
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


def build_rnn_server_wall_network():
    """Build the exact network structure from rnn_server_wall.py"""
    
    print("Building RNN Server Wall network structure...")
    
    tf.reset_default_graph()
    he_init = tf.variance_scaling_initializer()

    # Place Holders - exact from rnn_server_wall.py
    loc_input = tf.placeholder(tf.float32, shape=(None, 2, num_users), name="loc_input")
    channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_ris, num_users), name="channel_bs_irs_user")
    H_SI_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_SI")
    H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_b")

    # Network architecture (exactly from rnn_server_wall.py)
    with tf.name_scope("array_response_construction"):
        lay = {}
        lay['P'] = tf.constant(1.0)
        from0toN = tf.cast(tf.range(0, N, 1), tf.float32)

    with tf.name_scope("channel_sensing"):
        hidden_size = 256
        layer_neuron_size = 256
        A1 = tf.get_variable("A1", shape=[hidden_size, layer_neuron_size], dtype=tf.float32, initializer=he_init)
        A2 = tf.get_variable("A2", shape=[layer_neuron_size, layer_neuron_size], dtype=tf.float32, initializer=he_init)
        A3 = tf.get_variable("A3", shape=[layer_neuron_size, layer_neuron_size], dtype=tf.float32, initializer=he_init)
        A4 = tf.get_variable("A4", shape=[layer_neuron_size, 2*N_ris], dtype=tf.float32, initializer=he_init)

        b1 = tf.get_variable("b1", shape=[layer_neuron_size], dtype=tf.float32, initializer=he_init)
        b2 = tf.get_variable("b2", shape=[layer_neuron_size], dtype=tf.float32, initializer=he_init)
        b3 = tf.get_variable("b3", shape=[layer_neuron_size], dtype=tf.float32, initializer=he_init)
        b4 = tf.get_variable("b4", shape=[2*N_ris], dtype=tf.float32, initializer=he_init)
        
        layer_Ui = Dense(units=hidden_size, activation='linear')
        layer_Wi = Dense(units=hidden_size, activation='linear')
        layer_Uf = Dense(units=hidden_size, activation='linear')
        layer_Wf = Dense(units=hidden_size, activation='linear')
        layer_Uo = Dense(units=hidden_size, activation='linear')
        layer_Wo = Dense(units=hidden_size, activation='linear')
        layer_Uc = Dense(units=hidden_size, activation='linear')
        layer_Wc = Dense(units=hidden_size, activation='linear')
        
        def RNN(input_x, h_old, c_old):
            i_t = tf.sigmoid(layer_Ui(input_x) + layer_Wi(h_old))
            f_t = tf.sigmoid(layer_Uf(input_x) + layer_Wf(h_old))
            o_t = tf.sigmoid(layer_Uo(input_x) + layer_Wo(h_old))
            c_t = tf.tanh(layer_Uc(input_x) + layer_Wc(h_old))
            c = i_t * c_t + f_t * c_old
            h_new = o_t * tf.tanh(c)
            return h_new, c
        
        snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
        snr_dB = 10* tf.log(snr) / np.log(10)
        snr_normal = snr_dB
        
        theta_list = []
        
        for t in range(tau):
            if t == 0:
                y_real = tf.ones([tf.shape(loc_input)[0], 32])
                h_old = tf.zeros([tf.shape(loc_input)[0], hidden_size])
                c_old = tf.zeros([tf.shape(loc_input)[0], hidden_size])
            h_old, c_old = RNN(tf.concat([y_real, snr_normal], axis=1), h_old, c_old)

            x1 = tf.nn.relu(h_old @ A1 + b1)
            x1 = BatchNormalization()(x1)
            x2 = tf.nn.relu(x1 @ A2 + b2)
            x2 = BatchNormalization()(x2)
            x3 = tf.nn.relu(x2 @ A3 + b3)
            x3 = BatchNormalization()(x3)

            ris_her_unnorm = x3 @ A4 + b4
            ris_her_r = ris_her_unnorm[:, 0:N_ris]
            ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris]
            theta_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True))
            theta_real = tf.divide(ris_her_r , theta_tmp)
            theta_imag = tf.divide(ris_her_i , theta_tmp)
            theta = tf.concat([theta_real, theta_imag], axis=1)   
            theta_T = tf.reshape(theta, [-1, 1, 2 * N_ris])

            theta_list.append(theta_T[:, 0, :])
            
            A_T_k1 = channel_bs_irs_user[:, :, :, 0]
            A_T_k = (A_T_k1)
            
            theta_A_k_T = tf.matmul(A_T_k, tf.transpose(theta_T, perm=[0,2,1]))
            
            h_d_plus_h_cas = theta_A_k_T
            h_d_plus_h_cas_re = h_d_plus_h_cas[:, 0:N_ris, :]
            h_d_plus_h_cas_im = h_d_plus_h_cas[:, N_ris:2*N_ris, :]
            noise = tf.complex(tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean=0.0, stddev=noiseSTD_per_dim), 
                               tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean=0.0, stddev=noiseSTD_per_dim))
            y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.complex(h_d_plus_h_cas_re, h_d_plus_h_cas_im) + noise
            
            if USE_FFT:
                y_fft = tf.signal.fft(y_complex)
                y_real = tf.concat([tf.real(y_fft), tf.imag(y_fft)], axis=1)
            else:
                y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1)
            y_real = tf.reshape(y_real, [-1, 32])
                
        h_old, c_old = RNN(tf.concat([y_real, snr_normal], axis=1), h_old, c_old)
        c_old = Dense(units=200, activation='linear')(c_old)
        c_old = Dense(units=200, activation='linear')(c_old)
        
        x1 = tf.nn.relu(h_old @ A1 + b1)
        x1 = BatchNormalization()(x1)
        x2 = tf.nn.relu(x1 @ A2 + b2)
        x2 = BatchNormalization()(x2)
        x3 = tf.nn.relu(x2 @ A3 + b3)
        x3 = BatchNormalization()(x3)
        ris_her_unnorm = x3 @ A4 + b4
        ris_her_r = ris_her_unnorm[:, 0:N_ris]
        ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris]
        theta_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True))
        theta_real = ris_her_r / theta_tmp
        theta_imag = ris_her_i / theta_tmp
        
        loc_hat = Dense(units=2, activation='relu')(c_old)  
         
        theta_T_complex = tf.complex(theta_real, theta_imag)
        theta_T_complex = tf.reshape(theta_T_complex, [-1, N_ris, 1])

        H_SI_tf = H_SI_placeholder

        batch_size = tf.shape(loc_input)[0]
        def compute_G_for_batch(H_SI_batch):
            s, u, v = tf.linalg.svd(H_SI_batch)
            s_inv = tf.where(s > 1e-14, 1.0 / s, tf.zeros_like(s))
            s_inv_complex = tf.cast(s_inv, tf.complex64)
            H_SI_pinv = tf.matmul(v, tf.matmul(tf.linalg.diag(s_inv_complex), tf.linalg.adjoint(u)))
            middle = tf.matmul(H_SI_batch, H_SI_pinv)
            I = tf.eye(N_ris, dtype=tf.complex64)
            return I - middle

        G = tf.map_fn(compute_G_for_batch, H_SI_tf, dtype=tf.complex64)

        H_b_hat = H_b_placeholder

        G_Hb = tf.matmul(G, H_b_hat)
        G_Hb_theta = tf.matmul(G_Hb, theta_T_complex)
        power = tf.reduce_mean(lay['P'] *tf.abs(G_Hb_theta) ** 2, axis=[1, 2])
        snr_eff = power / (N_ris - 1 )/2/ (noiseSTD_per_dim**2)

        # sig_pow = lay['P'] * tf.reduce_sum( tf.abs( tf.matmul(H_b_hat, theta_T_complex) )**2, axis=1 )
        # interference_pow = lay['P'] * tf.reduce_sum(tf.abs( tf.matmul(H_SI_tf, theta_T_complex))**2, axis=1)
        # sinr = sig_pow / (interference_pow + 2*(noiseSTD_per_dim**2)*N_ris)

        sp = lay['P'] * tf.abs(tf.linalg.adjoint(theta_T_complex) @ tf.matmul(H_b_hat, theta_T_complex))**2
        intp = lay['P'] * (tf.abs(tf.linalg.adjoint(theta_T_complex) @ tf.matmul(H_SI_tf, theta_T_complex))**2) 
        sinr_tr =  sp / ( intp + 2*(noiseSTD_per_dim**2))

    loss = - tf.log(tf.reduce_mean(sinr_tr))
    
    return {
        'placeholders': {
            'loc_input': loc_input,
            'channel_bs_irs_user': channel_bs_irs_user,
            'H_SI_placeholder': H_SI_placeholder,
            'H_b_placeholder': H_b_placeholder
        },
        'outputs': {
            'theta_list': theta_list,
            'theta_T_complex': theta_T_complex,
            'sinr_tr': sinr_tr,
            'sp': sp,
            'intp': intp,
            'loss': loss
        },
        'layers': {
            'lay': lay
        }
    }


def build_2b_network():
    """Build the exact network structure from 2b.py"""
    
    print("Building 2b dual beamforming network structure...")
    
    tf.reset_default_graph()
    
    # Create placeholders exactly as in 2b.py
    loc_input = tf.placeholder(tf.float32, shape=(None, 2, num_users), name="loc_input")
    channel_bs_irs_user = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris, num_users), name="channel_bs_irs_user")
    H_SI_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_SI")
    H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_b")
    
    batch_size = tf.shape(loc_input)[0]
    
    with tf.name_scope("array_response_construction"):
        lay = {}
        lay['P'] = tf.constant(1.0)
    
    with tf.name_scope("channel_sensing"):
        hidden_size1 = 256
        
        # Import Keras layers exactly as in 2b.py
        from keras.layers import BatchNormalization, Dense
        
        # Create MLPBlock class exactly as in 2b.py
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

        # Create RNN class exactly as in 2b.py
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

        # Create the exact network instances
        RNN1 = RNN(hidden_size1, 'rnn_1')
        MLP_user1_transmit = MLPBlock(3, [256, 256, 2 * N_ris], name='mlp_user1_transmit')
        MLP_user1_receive = MLPBlock(3, [256, 256, 2 * N_ris], name='mlp_user1_receive')

        A_T_k1 = channel_bs_irs_user[:,:,:,0]  # channel within coherence time
        theta_list = []  # list of complex transmit beamforming
        v_list = []  # list of complex receive beamforming
        
        # Create the EXACT loop structure from 2b.py
        for t in range(tau):
            if t == 0:
                h_old1 = tf.zeros([batch_size, hidden_size1])
                c_old1 = tf.zeros([batch_size, hidden_size1])

                # RIS receive beamformer - EXACT variable names from checkpoint
                v_uplink_real = tf.get_variable("v_uplink_real1" + str(t), shape=(1, N_ris, 1), trainable=True)
                v_uplink_imag = tf.get_variable("v_uplink_imag1" + str(t), shape=(1, N_ris, 1), trainable=True)
                v_complex = tf.complex(v_uplink_real, v_uplink_imag)
                v1 = v_complex / tf.norm(v_complex, axis=1, keepdims=True)

                # RIS transmit beamformer - EXACT variable names from checkpoint
                v_uplink_real = tf.get_variable("v_uplink_real2" + str(t), shape=(1, N_ris, 1), trainable=True)
                v_uplink_imag = tf.get_variable("v_uplink_imag2" + str(t), shape=(1, N_ris, 1), trainable=True)
                v_complex = tf.complex(v_uplink_real, v_uplink_imag)
                v2 = v_complex / tf.norm(v_complex, axis=1, keepdims=True)

            # BS observes the next measurement
            y_noiseless1 = A_T_k1 @ v2
            noise1 = tf.complex(tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim),
                                tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim))
            y_complex1 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless1 + noise1      
            y_complex2 = tf.transpose(tf.conj(v1), perm=[0, 2, 1]) @ y_complex1     

            # Flatten both to compatible dimensions for concatenation
            y_complex1_flat = tf.reshape(y_complex1, [tf.shape(y_complex1)[0], N_ris])  
            y_complex2_flat = tf.reshape(y_complex2, [tf.shape(y_complex2)[0], 1])      
            y_real = tf.concat([tf.real(y_complex1_flat), tf.imag(y_complex1_flat), 
                               tf.real(y_complex2_flat), tf.imag(y_complex2_flat)], axis=1)

            # BS design next receive beamformer based on h_old1
            h_old1, c_old1 = RNN1((y_real, h_old1, c_old1))
            v_her = MLP_user1_transmit(h_old1)  
            v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
            v_her = tf.divide(v_her, v_norm)
            vr = tf.complex(v_her[:, 0:N_ris], v_her[:, N_ris:2 * N_ris])
            vr = tf.reshape(vr, [-1, N_ris, 1])
            v_list.append(vr)

            # BS design next transmit beamformer based on h_old1
            v_her = MLP_user1_receive(h_old1)  
            v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
            v_her = tf.divide(v_her, v_norm)
            theta_T_complex = tf.complex(v_her[:, 0:N_ris], v_her[:, N_ris:2 * N_ris])
            theta_T_complex = tf.reshape(theta_T_complex, [-1, N_ris, 1])
            theta_list.append(theta_T_complex)

        # Calculate SINR
        MLP_bf1 = MLPBlock(3, [512, 512, 2 * N_ris], name='mlp_bf1')

        # Final receive beamformer
        v_tmp = MLP_bf1(c_old1)
        v_norm = tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1))
        v_tmp = tf.divide(v_tmp, v_norm)
        v_complex_final = tf.complex(v_tmp[:, 0:N_ris], v_tmp[:, N_ris:2 * N_ris])
        v_complex_final = tf.reshape(v_complex_final, [-1, N_ris, 1])

        H_SI_tf = H_SI_placeholder
        H_b_hat = H_b_placeholder

        def compute_G_for_batch(H_SI_batch):   
            s, u, v = tf.linalg.svd(H_SI_batch)
            s_inv = tf.where(s > 1e-8, 1.0 / s, tf.zeros_like(s))
            s_inv_complex = tf.cast(s_inv, tf.complex64)
            H_SI_pinv = tf.matmul(v, tf.matmul(tf.linalg.diag(s_inv_complex), tf.linalg.adjoint(u)))
            middle = tf.matmul(H_SI_batch, H_SI_pinv)
            I = tf.eye(N_ris, dtype=tf.complex64)
            return I - middle

        G = tf.map_fn(compute_G_for_batch, H_SI_tf, dtype=tf.complex64)  
        G_Hb = tf.matmul(G, H_b_hat)  
        G_Hb_theta = tf.matmul(G_Hb, theta_T_complex)  
        power = lay['P'] * tf.abs( tf.transpose(tf.conj(v_complex_final), perm=[0,2,1]) @ G_Hb_theta) ** 2   

        sig_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex_final), perm=[0,2,1]) @ tf.matmul(H_b_hat, theta_T_complex))**2
        interference_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex_final), perm=[0,2,1]) @ tf.matmul(H_SI_tf, theta_T_complex))**2
        sinr = sig_pow / (interference_pow + 2*(noiseSTD_per_dim**2))

    # Loss function
    loss = - tf.log(tf.reduce_mean(sinr))
    
    return {
        'placeholders': {
            'loc_input': loc_input,
            'channel_bs_irs_user': channel_bs_irs_user,
            'H_SI_placeholder': H_SI_placeholder,
            'H_b_placeholder': H_b_placeholder
        },
        'outputs': {
            'theta_list': theta_list,
            'v_list': v_list,
            'theta_T_complex': theta_T_complex,
            'v_complex_final': v_complex_final,
            'sinr': sinr,
            'sig_pow': sig_pow,
            'interference_pow': interference_pow,
            'loss': loss
        },
        'layers': {
            'lay': lay
        }
    }


def test_model_with_exact_structure(model_path, test_file_path, network_type='rnn_server_wall', num_samples=5):
    """Test model using exact network structure"""
    
    print(f"\\n{'='*60}")
    print(f"TESTING MODEL: {os.path.basename(model_path)} ({network_type})")
    print(f"{'='*60}")
    
    # Load test data
    try:
        test_data = scipy.io.loadmat(test_file_path)
        loc_true = test_data['loc_true']  
        print(f"Loaded {loc_true.shape[0]} test locations from {os.path.basename(test_file_path)}")
    except Exception as e:
        print(f"Error loading test file: {e}")
        return None
    
    # Build the appropriate network structure
    if network_type == 'rnn_server_wall':
        network = build_rnn_server_wall_network()
        outputs_to_fetch = ['sinr_tr', 'sp', 'intp', 'theta_list']
    elif network_type == '2b':
        loc_true = loc_true[0]
        network = build_2b_network()
        outputs_to_fetch = ['sinr', 'sig_pow', 'interference_pow', 'theta_list', 'v_list']
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Test with saved model
    results = {
        'sinr_predicted': [],
        'sig_pow_predicted': [],
        'interference_pow_predicted': [],
        'theta_predicted': [],
        'locations': [],
        'theta_list_predicted': []
    }
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        try:
            print(f"Restoring model from: {model_path}")
            saver.restore(sess, model_path)
            print("✅ Model restored successfully!")
            
            num_samples = min(num_samples, loc_true.shape[0])
            
            for i in range(num_samples):
                if i % 5 == 0:
                    print(f"Processing sample {i+1}/{num_samples}")
                
                # Get location from saved data
                saved_location = loc_true[i, 0, :, 0]  # [angle, distance]
                angle, distance = saved_location[0], saved_location[1]
                
                # Convert to Cartesian coordinates
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                z = -20.0
                location_user_test = np.array([[x, y, z]])
                
                # Generate channel for this location
                channel_true_test, set_location_user_test = generate_irs_user_channel(
                    location_user_test, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
                
                # Prepare feed dictionary based on network type
                if network_type == 'rnn_server_wall':
                    A_T_1_real_test, _ = channel_complex2real(channel_true_test)
                    feed_dict_test = {
                        network['placeholders']['loc_input']: set_location_user_test,
                        network['placeholders']['channel_bs_irs_user']: A_T_1_real_test,
                        network['layers']['lay']['P']: Pvec[0],
                        network['placeholders']['H_SI_placeholder']: np.tile(channel_true_test[0][np.newaxis, :, :], (1, 1, 1)),
                        network['placeholders']['H_b_placeholder']: channel_true_test[2]
                    }
                else:  # 2b network
                    feed_dict_test = {
                        network['placeholders']['loc_input']: np.array(set_location_user_test),
                        network['placeholders']['channel_bs_irs_user']: channel_true_test[1],  # Use complex channel directly
                        network['layers']['lay']['P']: Pvec[0],
                        network['placeholders']['H_SI_placeholder']: np.tile(channel_true_test[0][np.newaxis, :, :], (1, 1, 1)),
                        network['placeholders']['H_b_placeholder']: channel_true_test[2]
                    }
                
                # Run inference
                outputs = [network['outputs'][key] for key in outputs_to_fetch]
                results_run = sess.run(outputs, feed_dict=feed_dict_test)
                
                # Unpack results
                sinr_pred, sp_pred, intp_pred, theta_list_pred, theta_pred = results_run
                
                # Store results
                results['sinr_predicted'].append(float(sinr_pred[0]))
                results['sig_pow_predicted'].append(float(sp_pred[0]))
                results['interference_pow_predicted'].append(float(intp_pred[0]))
                results['theta_predicted'].append(theta_pred[0].squeeze())
                results['locations'].append(saved_location)
                
                # Process theta_list for beam pattern
                theta_test = np.array(theta_list_pred)
                if network_type == 'rnn_server_wall':
                    theta_test_cplx = theta_test[:, :, 0:N_ris] + 1j * theta_test[:, :, N_ris:2*N_ris]
                else:  # 2b network - theta_list is already complex
                    theta_test_cplx = theta_test.squeeze()
                results['theta_list_predicted'].append(theta_test_cplx)
            
            print(f"✅ Successfully processed {len(results['sinr_predicted'])} samples")
            
        except Exception as e:
            print(f"❌ Error during model testing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return results


def test_single_location(model_path, user_location, network_type):
    """Test a single model with a specific user location"""
    
    print(f"Testing model: {os.path.basename(model_path)} ({network_type})")
    snr_const = 10#args.snr
    ref_dis = 5
    Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2
    
    # Build the appropriate network structure
    if network_type == 'rnn_server_wall':
        network = build_rnn_server_wall_network()
        outputs_to_fetch = ['sinr_tr', 'sp', 'intp', 'theta_list']
    elif network_type == '2b':
        network = build_2b_network()
        outputs_to_fetch = ['sinr', 'sig_pow', 'interference_pow', 'theta_list', 'v_list']
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        try:
            print(f"Restoring model from: {model_path}")
            saver.restore(sess, model_path)
            print("✅ Model restored successfully!")
            
            # Generate channels for the specific user location
            channel_true_test, set_location_user_test = generate_irs_user_channel(
                user_location.reshape(1, -1), location_ris_1, num_samples=1, Rician_factor=Rician_factor)
            
            # Prepare feed dict based on network type
            if network_type == 'rnn_server_wall':
                # Convert complex channels to real for rnn_server_wall
                A_T_1_real, _ = channel_complex2real(channel_true_test)
                
                H_b_test = np.array(channel_true_test[2])
                if H_b_test.ndim == 4 and H_b_test.shape[-1] == 1:
                    H_b_test = H_b_test.squeeze(axis=-1)
                
                feed_dict = {
                    network['placeholders']['loc_input']: np.array(set_location_user_test),
                    network['placeholders']['channel_bs_irs_user']: A_T_1_real[:, :, :, 0:1],  # Take first user, keep 4D
                    network['layers']['lay']['P']: Pvec,
                    network['placeholders']['H_SI_placeholder']: np.tile(channel_true_test[0][np.newaxis, :, :], (1, 1, 1)),
                    network['placeholders']['H_b_placeholder']: H_b_test
                }
                
                # Run inference
                results = sess.run({
                    'sinr_tr': network['outputs']['sinr_tr'],
                    'sp': network['outputs']['sp'],
                    'intp': network['outputs']['intp'],
                    'theta_list': network['outputs']['theta_list']
                }, feed_dict=feed_dict)
                
                sinr_db = 10 * np.log10(np.maximum(results['sinr_tr'], 1e-12))
                sig_pow = np.asarray(results['sp']).item()
                interference_pow = np.asarray(results['intp']).item()
                theta_list = np.asarray(results['theta_list']).squeeze()
                print(f"  SINR: {np.asarray(sinr_db).item():.2f} dB")
                print(f"  Signal Power: {sig_pow:.6f}")
                print(f"  Interference Power: {interference_pow:.6f}")
                
                return {
                    'sinr_db': np.asarray(sinr_db).item(),
                    'sig_pow': sig_pow,
                    'interference_pow': interference_pow,
                    'theta_list': theta_list,
                    'user_location': user_location
                }
                
            elif network_type == '2b':
                # Use complex channels directly for 2b
                H_b_test = np.array(channel_true_test[2])
                if H_b_test.ndim == 4 and H_b_test.shape[-1] == 1:
                    H_b_test = H_b_test.squeeze(axis=-1)
                
                feed_dict = {
                    network['placeholders']['loc_input']: np.array(set_location_user_test),
                    network['placeholders']['channel_bs_irs_user']: channel_true_test[1],
                    network['layers']['lay']['P']: Pvec,
                    network['placeholders']['H_SI_placeholder']: np.tile(channel_true_test[0][np.newaxis, :, :], (1, 1, 1)),
                    network['placeholders']['H_b_placeholder']: H_b_test
                }
                
                # Run inference
                results = sess.run({
                    'sinr': network['outputs']['sinr'],
                    'sig_pow': network['outputs']['sig_pow'],
                    'interference_pow': network['outputs']['interference_pow'],
                    'theta_list': network['outputs']['theta_list'],
                    'v_list': network['outputs']['v_list']
                }, feed_dict=feed_dict)
                
                sinr_db = 10 * np.log10(np.maximum(results['sinr'], 1e-12))
                sig_pow = np.asarray(results['sig_pow']).item()
                interference_pow = np.asarray(results['interference_pow']).item()
                theta_list = np.asarray(results['theta_list']).squeeze()
                v_list = np.asarray(results['v_list']).squeeze()

                print(f"  SINR: {np.asarray(sinr_db).item():.2f} dB")
                print(f"  Signal Power: {sig_pow:.6f}")
                print(f"  Interference Power: {interference_pow:.6f}")
                
                return {
                    'sinr_db': np.asarray(sinr_db).item(),
                    'sig_pow': sig_pow,
                    'interference_pow': interference_pow,
                    'theta_list': theta_list,
                    'v_list': v_list,
                    'user_location': user_location
                }
            
        except Exception as e:
            print(f"❌ Error during model testing: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None


# Main execution
if __name__ == "__main__":
#     # Test both networks with a fixed user location
#     results = test_both_networks_with_fixed_location()

# def test_both_networks_with_fixed_location():
    """Test both networks with a specific user location"""
    
    print("="*60)
    print("TESTING BOTH NETWORKS WITH FIXED USER LOCATION")
    print("="*60)

    # Set a specific test location
    test_angle = 1.00988728#np.random.uniform(-np.pi/2, np.pi/2)  # Random angle between -90 and 90 degrees
    test_distance = 5.0  # 5 meters

    # Convert to Cartesian coordinates
    x = test_distance * np.cos(test_angle)
    y = test_distance * np.sin(test_angle)
    z = -20  # Ground level
    
    user_location = np.array([x, y, z])
    
    print(f"Test User Location:")
    print(f"  Angle: {test_angle * 180/np.pi:.1f}°")
    print(f"  Distance: {test_distance:.1f} m")
    print(f"  Cartesian: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # Test both networks
    models_to_test = [
        {
            'name': 'BD_beamInit',
            'model_path': 'BD_beamInit/params_RiK10_mono_N_16_tau_10_snr_10',
            'network_type': 'rnn_server_wall'
        },
        {
            'name': 'BD_2beamsInit', 
            'model_path': 'BD_2beamsInit/params_RiK10_mono_N_16_tau_10_snr_10',
            'network_type': '2b'
        }
    ]
    
    results_all = {}
    
    for model_config in models_to_test:
        print(f"\\n{'='*40}")
        print(f"TESTING {model_config['name']}")
        print(f"{'='*40}")
        
        if not os.path.exists(model_config['model_path'] + '.meta'):
            print(f"❌ Model not found: {model_config['model_path']}")
            continue
        
        # Test the model with fixed location
        results = test_single_location(
            model_config['model_path'], 
            user_location,
            model_config['network_type']
        )
        
        if results is not None:
            results_all[model_config['name']] = results
            
            # Plot beam patterns
            if 'v_list' not in results:
                theta_test_cplx = results['theta_list'][ :, 0:N_ris] + 1j * results['theta_list'][ :, N_ris:2*N_ris]
                plot_beam_patterns(
                    theta_test_cplx, 
                    [[test_angle, test_distance]], 
                    save_path=None
                )
            else:
                plot_beam_patterns(
                    results['theta_list'][:,np.newaxis, :], 
                    [[test_angle, test_distance]], 
                    results['v_list'][:,np.newaxis, :], 
                    save_path=None
                )
    
    # Compare results
    print(f"\\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if len(results_all) == 2:
        bd_beam = results_all.get('BD_beamInit', {})
        bd_2beam = results_all.get('BD_2beamsInit', {})
        
        print(f"Performance Comparison:")
        if 'sinr_db' in bd_beam and 'sinr_db' in bd_2beam:
            print(f"  BD_beamInit SINR:     {bd_beam['sinr_db']:.2f} dB")
            print(f"  BD_2beamsInit SINR:   {bd_2beam['sinr_db']:.2f} dB")
            print(f"  Improvement:          {bd_2beam['sinr_db'] - bd_beam['sinr_db']:.2f} dB")
        
        if 'sig_pow' in bd_beam and 'sig_pow' in bd_2beam:
            print(f"\\nSignal Power:")
            print(f"  BD_beamInit:          {bd_beam['sig_pow']:.6f}")
            print(f"  BD_2beamsInit:        {bd_2beam['sig_pow']:.6f}")
        
        if 'interference_pow' in bd_beam and 'interference_pow' in bd_2beam:
            print(f"\\nInterference Power:")
            print(f"  BD_beamInit:          {bd_beam['interference_pow']:.6f}")
            print(f"  BD_2beamsInit:        {bd_2beam['interference_pow']:.6f}")
    
    print(f"\\n{'='*60}")
    print("TESTING COMPLETED")
    print(f"{'='*60}")
    print(f"Successfully tested {len(results_all)} model(s) with fixed location:")
    for name in results_all.keys():
        print(f"  ✅ {name}")
        

