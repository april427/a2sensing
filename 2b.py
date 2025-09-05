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
    # os.kill(os.getpid())

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# Additional M4 optimization
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from keras.layers import BatchNormalization, Dense
import random
from manifold_optimization import solve_with_random_restarts
from parse_args import parse_args
from channel_functions import *

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
drive_save_path = 'BD_2beamsInit'
os.makedirs(drive_save_path, exist_ok=True)

'System Information'
N = 1   # Number of BS's antennas
delta_inv = 32 #number of epoch 
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
Rician_factor = 10
location_user = None

# Sensing parameters
tau = 16# args.tau  # Pilot length
snr_const = args.snr
snr_const = np.array([snr_const])
ref_dis = 5
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2
            # BD at ref_dis has received SNR of snr_const. ||v||^2 = N_ris 

'Learning Parameters'
initial_run = 1   # 0: Continue training; 1: Starts from scratch
n_epochs = 200#args.n_epochs
learning_rate = 1e-3
batch_per_epoch = 128
batch_size_order = 4
batch_size_val = 10000
scale_factor = 1
test_size = 2000

model_path = f'{drive_save_path}/params_RiK10_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}'

USE_FFT = False

# Loss weights
LOS_weight = 1
NLOS_weight = 0

#####################################################
'Build the graph'
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method

# Place Holders
loc_input = tf.placeholder(tf.float32, shape=(None, 2, num_users), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.complex64, shape=(None,  N_ris, N_ris, num_users), name="channel_bs_irs_user")
H_SI_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_SI")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_b")

channel_true, set_location_user_train = generate_irs_user_channel(
                location_user, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
channel_complex2real(channel_true)
batch_size = tf.shape(loc_input)[0]
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    

with tf.name_scope("channel_sensing"):
    hidden_size1 = 512
    RNN1 = RNN(hidden_size1, 'rnn_1')
    
    MLP_user1_transmit = MLPBlock(3, [512, 512, 2 * N_ris], name='mlp_user1_transmit')
    MLP_user1_receive = MLPBlock(3, [512, 512, 2 * N_ris], name='mlp_user1_receive')

    A_T_k1 = channel_bs_irs_user[:,:,:,0] # channel within coherence time
    theta_list =[] # list of complex transmit beamforming
    v_list = [] # list of complex receive beamforming
    
    for t in range(tau):
        'initailization'
        if t == 0:
            h_old1 = tf.zeros([batch_size, hidden_size1])
            c_old1 = tf.zeros([batch_size, hidden_size1])

            'RIS receive beamformer'
            v_uplink_real = tf.get_variable("v_uplink_real1" + str(t), shape=(1, N_ris, 1), trainable=True)
            v_uplink_imag = tf.get_variable("v_uplink_imag1" + str(t), shape=(1, N_ris, 1), trainable=True)
            v_complex = tf.complex(v_uplink_real, v_uplink_imag)
            v1 = v_complex / tf.norm(v_complex, axis=1, keepdims=True)

            'RIS transmit beamformer'
            v_uplink_real = tf.get_variable("v_uplink_real2" + str(t), shape=(1, N_ris, 1), trainable=True)
            v_uplink_imag = tf.get_variable("v_uplink_imag2" + str(t), shape=(1, N_ris, 1), trainable=True)
            v_complex = tf.complex(v_uplink_real, v_uplink_imag)
            v2 = v_complex / tf.norm(v_complex, axis=1, keepdims=True)

        'BS observes the next measurement'
        y_noiseless1 = A_T_k1 @ v2
        noise1 = tf.complex(tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex1 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless1 + noise1      # without receive beamforming
        y_complex2 = tf.transpose(tf.conj(v1), perm=[0, 2, 1]) @ y_complex1     # after receive beamforming
        
        # Flatten both to compatible dimensions for concatenation
        y_complex1_flat = tf.reshape(y_complex1, [tf.shape(y_complex1)[0], N_ris])  # (batch_size, N_ris)
        y_complex2_flat = tf.reshape(y_complex2, [tf.shape(y_complex2)[0], 1])      # (batch_size, 1)
        y_real = tf.concat([tf.real(y_complex1_flat), tf.imag(y_complex1_flat), tf.real(y_complex2_flat), tf.imag(y_complex2_flat)], axis=1) #/ tf.sqrt(lay['P'])
        # y_real shape is now (batch_size, 2*N_ris + 2)

        'BS design next receive beamformer based on  h_old1'
        h_old1, c_old1 = RNN1((y_real, h_old1, c_old1))
        v_her = MLP_user1_transmit(h_old1)  # this is actually the MLP for receive beamformer
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        vr = tf.complex(v_her[:, 0:N_ris], v_her[:, N_ris:2 * N_ris])
        vr = tf.reshape(vr, [-1, N_ris, 1])
        v_list.append(vr)

        'BS design next transimit beamformer based on  h_old1'
        v_her = MLP_user1_receive(h_old1)  # this is actually the MLP for transimit beamformer
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        theta_T_complex = tf.complex(v_her[:, 0:N_ris], v_her[:, N_ris:2 * N_ris])
        theta_T_complex = tf.reshape(theta_T_complex, [-1, N_ris, 1])
        theta_list.append(theta_T_complex)

    'calculate SINR'
    MLP_bf1 = MLPBlock(3, [1024, 1024, 2 * N_ris], name='mlp_bf1')

    'receive beamformer'
    v_tmp = MLP_bf1(c_old1)
    v_norm = tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1))
    v_tmp = tf.divide(v_tmp, v_norm)
    v_complex = tf.complex(v_tmp[:, 0:N_ris], v_tmp[:, N_ris:2 * N_ris])
    v_complex = tf.reshape(v_complex, [-1, N_ris, 1])

    H_SI_tf = H_SI_placeholder
    H_b_hat = H_b_placeholder

    batch_size = tf.shape(loc_input)[0]
    def compute_G_for_batch(H_SI_batch):   # G = I - H_SI H_SI^{\dagger}
        s, u, v = tf.linalg.svd(H_SI_batch)
        s_inv = tf.where(s > 1e-8, 1.0 / s, tf.zeros_like(s))
        s_inv_complex = tf.cast(s_inv, tf.complex64)
        H_SI_pinv = tf.matmul(v, tf.matmul(tf.linalg.diag(s_inv_complex), tf.linalg.adjoint(u)))
        middle = tf.matmul(H_SI_batch, H_SI_pinv)
        I = tf.eye(N_ris, dtype=tf.complex64)
        return I - middle

    G = tf.map_fn(compute_G_for_batch, H_SI_tf, dtype=tf.complex64)  # (batch, N_ris, N_ris)
    G_Hb = tf.matmul(G, H_b_hat)  # (batch, N_ris, N_ris)
    G_Hb_theta = tf.matmul(G_Hb, theta_T_complex)  # (batch, N_ris, 1)
    power = lay['P'] * tf.abs( tf.transpose(tf.conj(v_complex), perm=[0,2,1]) @ G_Hb_theta) ** 2   # (batch,)
    snr_eff = power / ( 2*(noiseSTD_per_dim**2))

    sig_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex), perm=[0,2,1]) @ tf.matmul(H_b_hat, theta_T_complex))**2
    interference_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex), perm=[0,2,1]) @ tf.matmul(H_SI_tf, theta_T_complex))**2
    sinr = sig_pow / (interference_pow + 2*(noiseSTD_per_dim**2))
    

####### Loss Function
loss = - tf.log(tf.reduce_mean(sinr))
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

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

#%%
###########  Training ##########

with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, model_path)
    
    # Early stop
    best_val = 1e9
    wait = 0
    PATIENCE = 40
    print(tf.test.is_gpu_available())  

    no_increase = 0

    for epoch in range(n_epochs):
        batch_iter = 0
        epoch_train_losses = []
        epoch_u1_losses = []
        epoch_u2_losses = []

        for rnd_indices in range(batch_per_epoch):
            channel_true_train, set_location_user_train = generate_irs_user_channel(
                None, location_ris_1, num_samples=batch_size_order*delta_inv, Rician_factor=Rician_factor)
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

            _, train_loss_val, sinr_val = sess.run(
                [training_op, loss, sinr], feed_dict=feed_dict_batch
            )
            
            epoch_train_losses.append(train_loss_val)
            batch_iter += 1
        
        avg_train_loss = np.mean(epoch_train_losses)
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        
        print('epoch', epoch,
              '  train_loss:%2.7f' % avg_train_loss,
              '  val_loss:%2.7f' % loss_val,
              '  best_val:%2.7f' % best_val)
        
        if epoch % 4 == 3:  # Every 4 iterations it checks if the validation performace is improved, then saves parameters
            if loss_val < best_val:
                save_path = saver.save(sess, model_path)
                best_val = loss_val
                no_increase = 0
            else:
                no_increase = no_increase + 10
        
        if epoch % 5 == 0:
            print(f" SINR[dB]: {10*np.log10(np.mean(sinr_val)):.6f}")

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

    # Validation samples from training set
    num_test_samples = 10
    sample_indices = random.sample(range(len(set_location_user_val)), num_test_samples)
    
    train_losses = []
    theta_test_list = []
    location_list = []
    
    for sample_index in sample_indices:
        location_user_target = set_location_user_val[sample_index]
        A_T_1_real_test = A_T_1_real_val[sample_index]
        
        feed_dict_test = {
            loc_input: np.expand_dims(location_user_target, axis=0),
            channel_bs_irs_user: channel_true_val[1],
            lay['P']: Pvec[0],
            H_SI_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
            H_b_placeholder: H_b_val
        }
        
        mse_loss, theta_test, v_test = sess.run([loss, theta_list, v_list], feed_dict=feed_dict_test)
        train_losses.append(mse_loss)
        theta_test_list.append(theta_test)
        location_list.append(location_user_target)
    
    ########################## TEST and Save ####################### 
    
    sinr_test_set = []
    interference_pow_set = []
    test_loc = []
    sig_pow_set = []
    theta_test_set = []
    v_test_set = []

    # optimal beamforming using Riemannian optimization
    rieman_opti_theta_set = []
    rieman_opti_v_set = []
    rieman_opti_sig_pow_set = []
    rieman_opti_int_pow_set = []
    rieman_opti_sinr_set = []

        
    channel_true_test, set_location_user_test = generate_irs_user_channel(
        None, location_ris_1, num_samples=test_size, Rician_factor=Rician_factor)

    feed_dict_test = {
        loc_input: np.array(set_location_user_test),
        channel_bs_irs_user: channel_true_test[1],  # Use complex channel directly
        lay['P']: Pvec[0],
        H_SI_placeholder: np.tile(channel_true_test[0][np.newaxis, :, :], (len(set_location_user_test), 1, 1)),
        H_b_placeholder: channel_true_test[2]
    }

    _, sinr_test, theta_test, v_test, sig_pow_test, interference_pow_test = \
            sess.run([loss, sinr, theta_list, v_list, sig_pow, interference_pow], 
                            feed_dict=feed_dict_test)

    theta_test = np.array(theta_test)
    v_test = np.array(v_test)

    sinr_test_set = sinr_test[:,np.newaxis]
    interference_pow_set = interference_pow_test[:,np.newaxis]
    sig_pow_set = sig_pow_test[:,np.newaxis]
    test_loc = np.expand_dims(set_location_user_test, axis = 1)
    theta_test_set = np.expand_dims(theta_test.transpose(1,0,2,3), axis=2 ) 
    v_test_set = np.expand_dims( v_test.transpose(1,0,2,3), axis = 2)

    for j in range(test_size):
        ## Generate channel coherent optimal beamformer results
        H_b_test = channel_true_test[2][j]
        H_SI_test = channel_true_test[0]
        c = 2*noiseSTD_per_dim**2
        theta_star, v_star, _ = solve_with_random_restarts(np.sqrt(Pvec[0])*H_b_test, np.sqrt(Pvec[0])*H_SI_test, c, restarts=20)
        sig_pow_opti_rieman = Pvec[0] * np.abs(np.conj(v_star).T @ H_b_test @ theta_star)**2
        int_pow_opti_rieman = Pvec[0] * np.abs(np.conj(v_star).T @ H_SI_test @ theta_star)**2

        rieman_opti_theta_set.append(theta_star)
        rieman_opti_v_set.append(v_star)
        rieman_opti_sig_pow_set.append(sig_pow_opti_rieman)
        rieman_opti_int_pow_set.append(int_pow_opti_rieman)
        rieman_opti_sinr_set.append(10 * np.log10(sig_pow_opti_rieman / (int_pow_opti_rieman + 2 * noiseSTD_per_dim**2)))
    
# Save the final results
model_filename = os.path.join(drive_save_path, f'TEST_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}.mat')
sio.savemat(model_filename, dict(
    snr_const = snr_const,
    N = N, N_ris = N_ris, tau = tau,
    epoch = n_epochs, 
    theta_test = theta_test_set,
    v_test = v_test_set,
    loc_true = test_loc,
    sinr_test = sinr_test_set,
    interference_pow = interference_pow_set,
    sig_pow = sig_pow_set,
    rieman_opti_theta = rieman_opti_theta_set,
    rieman_opti_v = rieman_opti_v_set,
    rieman_opti_sig_pow = rieman_opti_sig_pow_set,
    rieman_opti_int_pow = rieman_opti_int_pow_set,
    rieman_opti_sinr = rieman_opti_sinr_set
))

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
        location_user_test = generate_location(num_users)
        channel_true_test, set_location_user_test = generate_irs_user_channel(
            location_user_test, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
        print(set_location_user_test)
        H_SI = channel_true_test[0][np.newaxis, :, :]  # Self-interference channel

        H_b_test = np.array(channel_true_test[2])
        if H_b_test.ndim == 4 and H_b_test.shape[-1] == 1:
            H_b_test = H_b_test.squeeze(axis=-1)

        feed_dict_test = {
            loc_input: np.array(set_location_user_test),
            channel_bs_irs_user: channel_true_test[1],
            lay['P']: Pvec[0],
            H_SI_placeholder: H_SI,  # Add batch dimension
            H_b_placeholder: H_b_test    # Add batch dimension
        }

        mse_loss,  theta_test, v_test,  sinr_test, sig_pow_test, interference_pow_test = sess.run(
                [loss,  theta_list, v_list, sinr, sig_pow, interference_pow], 
                feed_dict=feed_dict_test)
        test_losses.append(mse_loss)
        theta_test_cplx = np.array(theta_test)
        v_cplx = np.array(v_test) 
        test_location_list.append(location_user_test)

        plot_beam_patterns(
            theta_test_cplx, set_location_user_test, v_cplx, 
            save_path=None
        ) 

        print(f" SINR: {10*np.log10(np.mean(sinr_test)):.6f}")
        print(f" Signal Power: {np.mean(sig_pow_test):.6f}")
        print(f" Interference Power: {np.mean(interference_pow_test):.6f}")



        ### Optimum coherent beamformer 
        i1 = np.mod(np.arange(16), 16)
        
        H_b_hat = channel_true_test[2][0]
        if np.all(H_SI) == 0:
            theta_opti = 1/np.sqrt(N_ris)*np.exp(1j * np.pi * (i1 * np.sin(set_location_user_test[0][0][0]))) 
            H_B = np.matmul(np.conj(H_b_hat).transpose(), H_b_hat)
        else:
            # Rayleigh quotient with numerical stability
            H_B = np.matmul(np.conj(H_b_hat).transpose(), H_b_hat)
            H_A = np.matmul(np.conj(H_SI.squeeze()).transpose(), H_SI.squeeze()) + 2*(noiseSTD_per_dim**2)/Pvec[0] * np.eye(N_ris)
            
            # Add numerical stability
            H_A_reg = H_A + 1e-8 * np.eye(N_ris)  # Regularization
            H_B_reg = H_B + 1e-8 * np.eye(N_ris)  # Regularization
            
            try:
                eigv, s, _ = np.linalg.svd(np.matmul(np.linalg.pinv(H_A_reg), H_B_reg))
                theta_opti = eigv[:,0]
            except np.linalg.LinAlgError:
                # Fallback to simple steering vector if SVD fails
                angle = set_location_user_test[0][0][0]
                theta_opti = 1/np.sqrt(N_ris)*np.exp(1j * np.pi * (i1 * np.sin(angle)))

        # Numerical stability for G matrix calculation
        H_SI_pinv = np.linalg.pinv(H_SI.squeeze())
        midterm = np.matmul(H_SI.squeeze(), H_SI_pinv)
        G = np.eye(N_ris) - midterm 

        sig_pow_opti = Pvec[0]*np.abs( np.transpose(np.conj(theta_opti))@ H_b_hat @ theta_opti)**2
        interference_pow_opti = Pvec[0]*np.abs(np.transpose(np.conj(theta_opti))@ H_SI.squeeze() @ theta_opti)**2
        sinr_opti = sig_pow_opti / (interference_pow_opti + 2*(noiseSTD_per_dim**2) + 1e-12)

        ### Using manifold optimization
        c = 2*noiseSTD_per_dim**2
        theta_star, v_star, val = solve_with_random_restarts(np.sqrt(Pvec[0])*H_b_hat, np.sqrt(Pvec[0])*H_SI.squeeze(), c, restarts=20)
        sig_pow_opti_mani = Pvec[0]*np.abs( np.transpose(np.conj(v_star))@ H_b_hat @ theta_star)**2
        interference_pow_opti_mani = Pvec[0]*np.abs(np.transpose(np.conj(v_star))@ H_SI.squeeze() @ theta_star)**2
        sinr_opti_mani = sig_pow_opti_mani / (interference_pow_opti_mani + 2*(noiseSTD_per_dim**2) + 1e-12)

        print(f"SINR of optimal beam: {10*np.log10(np.maximum(sinr_opti, 1e-12)):.3f} ")
        print(f"Signal Power: {sig_pow_opti:.3f} ")
        print(f"Interference Power: {interference_pow_opti:.3f} ")

        print(f"SINR of optimal beam (manifold): {10*np.log10(np.maximum(sinr_opti_mani, 1e-12)):.3f} ")
        print(f"Signal Power (manifold): {sig_pow_opti_mani:.3f} ")
        print(f"Interference Power (manifold): {interference_pow_opti_mani:.3f} ")

        plot_beam_patterns(
            theta_star[np.newaxis,np.newaxis,:], set_location_user_test, v_star[np.newaxis, np.newaxis, :], 
            save_path=None
        )    
# %%
