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

drive_save_path = 'BD_beamInit'
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
y_wall = 55.5

# Positions
location_ris_1 = np.array([0, 0, -20])       # This RIS is our BS
num_ris = 1

# Channel parameters
mean_true_alpha = 0.0 + 0.0j
std_per_dim_alpha = np.sqrt(0.5) 
noiseSTD_per_dim = np.sqrt(0.5)

# RIS configuration
N_ris = 16  # Number of RIS elements, treat RIS as BS for the mono-static case
num_users = 1  
params_system = (N_ris, N_ris, num_users)
Rician_factor = 10
location_user = None

# Sensing parameters
tau = 10  # Pilot length
snr_const = 5
snr_const = np.array([snr_const]) 
ref_dis = 5
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)
            # BD at ref_dis has received SNR of snr_const. ||v||^2 = N_ris 

'Learning Parameters'
initial_run = 1   # 0: Continue training; 1: Starts from scratch
n_epochs = 3
learning_rate = 1e-3
batch_per_epoch = 256
batch_size_order = 16
val_size_order = 10
scale_factor = 1
test_size = 782

USE_FFT = False

# Loss weights
LOS_weight = 1
NLOS_weight = 0

tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Place Holders
# loc_input = tf.placeholder(tf.float32, shape=(None, num_users, 3), name="loc_input")
loc_input = tf.placeholder(tf.float32, shape=(None, 2, num_users), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_ris, num_users), name="channel_bs_irs_user")
H_SI_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_SI")
H_b_placeholder = tf.placeholder(tf.complex64, shape=(None, N_ris, N_ris), name="H_b")

def path_loss_r(d, wavelength):   # return |beta|^2 in dB
    """Keyhole channel: pathloss of backscatter signal for Rician fading in dB. (4 pi d / wavelength)^4 ! """
    # wavelength = 3e8/fc
    loss = 40*np.log10(4*np.pi/wavelength) + 40.0 * np.log10(d + 1e-8)  
    return loss

def generate_location(num_users):
    """
    generate user position:  
    user is on a circle 
    """
    location_user = np.empty([num_users, 3])

    angle = np.random.uniform(-np.pi, np.pi)
    dis = 5
    x1 = dis * np.cos(angle)
    y1 = dis * np.sin(angle)

    # x1 = np.random.uniform(-40, -10)
    # y1 = np.random.uniform(20, 60)
    z1 = -20.0
    location_user[0, :] = np.array([x1, y1, z1])

    if num_users >= 2:
        y2 = 2 * y_wall - y1
        location_user[1, :] = np.array([x1, y2, z1])

    return location_user


def generate_irs_user_channel(user_locations, location_irs, num_samples=1, Rician_factor=10, scale_factor=0, irs_Nh=16, x_BD = 1):

    num_elements_irs = N_ris
    if user_locations is None:
        num_user = num_users  # 使用全局变量
    else:
        num_user = user_locations.shape[0] if user_locations.ndim == 2 else user_locations.shape[1]
    
    channel_irs_user = []
    set_location_user = []
    H_b = []

    wavelength = Wavelength

    # Self-interference channel
        # The BS antennas Nr = Nt
    tx = np.arange(0, N_ris, 1) * (wavelength/2)
    rx = np.arange(N_ris, N_ris*2, 1) * (wavelength/2)
    H_SI = np.zeros((N_ris, N_ris), dtype=complex)
    for i in range(N_ris):
        for j in range(N_ris):
            H_SI[i, j] = 1/(rx[j] - tx[i]) * np.exp(- 1j * 2 * np.pi * (rx[j] - tx[i]) / wavelength)
    H_SI = 5e-6 * H_SI * N_ris/ np.linalg.norm(H_SI,'fro')   # Normalization
    
    for ii in range(num_samples):
        # 获取用户位置
        if user_locations is None:
            location_user = generate_location(num_user)
        elif user_locations.ndim >= 3:  # For multiple samples
            location_user = user_locations[ii, :, :]
        else:
            location_user = user_locations
            
        # set_location_user.append(location_user)
        
        # Pathloss and AoA calculation
        pathloss_irs_user = []
        aoa_irs_y = []
        aoa_irs_z = []
        aoa_irs_cos_z = []
        
        for k in range(num_user):
            d_k = np.linalg.norm(location_user[k] - location_irs) # user-IRS distance
            d_k_xy = np.linalg.norm(location_user[k][0:2] - location_irs[0:2])  # horizontal distance
            pathloss_irs_user.append(path_loss_r(d_k,wavelength))
            aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / (d_k_xy +1e-8)     # Sine of azimuth angle
            aoa_irs_z_cos_k = d_k_xy / (d_k + 1e-8)  # Cosine of Elevation angle
            aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / (d_k +1e-8)     # Sine of Elevation angle
            aoa_irs_y.append(aoa_irs_y_k)
            aoa_irs_z.append(aoa_irs_z_k)
            aoa_irs_cos_z.append(aoa_irs_z_cos_k)
        
        aoa_irs_y = np.array(aoa_irs_y)
        aoa_irs_z = np.array(aoa_irs_z)
        aoa_irs_cos_z = np.array(aoa_irs_cos_z)

        # 应用缩放因子 
        # Xiyu: scale up the channel coefficients for better numerical stability
        pathloss_irs_user = np.array(pathloss_irs_user) - scale_factor / 2
        pathloss_irs_user = np.sqrt( 10 ** ((-pathloss_irs_user) / 10) )


        # set_location_user.append(np.array([aoa_irs_y, pathloss_irs_user]))
        set_location_user.append(np.array([np.arcsin(aoa_irs_y_k), d_k])[:, np.newaxis])
        
        # Xiyu: This is considered when the RIS is a rectangular array
        i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
        i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)

        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[ num_elements_irs, num_elements_irs, num_user])

        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] * aoa_irs_cos_z[k] + i2 * aoa_irs_z[k])) # steering vector norm is N_ris
            a_irs_user = a_irs_user[:, np.newaxis]

            tmp[ :,:, k] = np.sqrt(Rician_factor/(1+Rician_factor)) *(a_irs_user @ np.transpose(np.conj( a_irs_user))) + np.sqrt(1/(1+Rician_factor)) * tmp[:,:, k]
            tmp[:,:,k] = tmp[:,:, k] * pathloss_irs_user[k] # Backscattered channel

        channel_irs_user.append( x_BD* tmp + H_SI[:,:,np.newaxis])
        H_b.append(x_BD* tmp.squeeze())
    
    channels = (H_SI, np.array(channel_irs_user), H_b) 
    # Channel typle: self-interference, IRS-user, backscattered
    return channels, set_location_user

def channel_complex2real(channels):
    """complex = [real, imagnary]"""
    H_SI, channel_irs_user, channel_backscattered = channels
    (num_sample, num_elements_irs, _, num_user) = channel_irs_user.shape
    num_antenna_bs = N_ris
    
    A_T_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs, num_user])
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, num_user], dtype=complex)
    
    for kk in range(num_user):
        channel_irs_user_k = channel_irs_user[:, :, :, kk]
        # (num_sample, num_elements_irs, num_elements_irs)  
        channel_combine_irs = channel_irs_user_k.reshape(num_sample, num_elements_irs, num_elements_irs)
        set_channel_combine_irs[:, :, :, kk] = channel_combine_irs
        
        A_tmp_tran = channel_combine_irs # Xiyu: No need to transpose np.transpose(channel_combine_irs, (0, 2, 1))
        A_tmp_real1 = np.concatenate([A_tmp_tran.real, - A_tmp_tran.imag], axis=2)
        A_tmp_real2 = np.concatenate([A_tmp_tran.imag, A_tmp_tran.real], axis=2)
        A_tmp_real = np.concatenate([A_tmp_real1, A_tmp_real2], axis=1)
        A_T_real[:, :, :, kk] = A_tmp_real
    
    return A_T_real, set_channel_combine_irs

def generate_RSS_adaptive(A_T_real, the_theta, P_temp):
    """generate current position signal strength"""
    RSS_list = np.zeros(tau, dtype=float)
    for tau_i in range(tau):
        theta_i = the_theta[[tau_i], :]
        theta = np.concatenate([theta_i.real, theta_i.imag], axis=1)
        theta_T = np.reshape(theta, [-1, 1, 2 * N_ris])
        A_T_k = A_T_real[0, :, :, 0]
        theta_A_k_T = np.matmul(theta_T, A_T_k)
        theta_A_k_T_re = theta_A_k_T[:,:,0]
        theta_A_k_T_im = theta_A_k_T[:,:,1]
        RSS_i = abs((np.sqrt(P_temp)+ 1j*0.0) * (theta_A_k_T_re + 1j*theta_A_k_T_im)) ** 2 
        RSS_list[tau_i] = RSS_i.item()
    return RSS_list

def generate_radio_map(theta_test):
    """generate radio map"""
    x_lowerlimit, x_upperlimit = -35, -15
    y_lowerlimit, y_upperlimit = 25, 55.5
    z_fixed = -20
    
    x_range = int((x_upperlimit - x_lowerlimit) / 0.5) + 1
    y_range = int((y_upperlimit - y_lowerlimit) / 0.5) + 1
    radio_map = np.zeros([x_range, y_range, tau])

    for x_i in range(x_range):
        for y_i in range(y_range):
            coordinate_k = np.array([x_lowerlimit + x_i * 0.5, y_lowerlimit + y_i * 0.5, z_fixed])
            location_user = np.empty([num_users, 3])
            location_user[0, :] = coordinate_k
            
            # 这里改用简化的信道生成函数
            channel_true, set_location_user_train = generate_irs_user_channel(
                location_user, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
            A_T_real, _ = channel_complex2real(channel_true)
            RSS_offline = generate_RSS_adaptive(A_T_real, theta_test, Pvec[0])
            radio_map[x_i, y_i, :] = RSS_offline
            
    return radio_map, theta_test

def calculate_beam_pattern(theta_vector, angles):
    """Calculate beam pattern for given beamforming vector and angles"""
    beam_pattern = []
    n = np.arange(N_ris)
    
    for angle in angles:
        # Steering vector for uniform linear array
        steering_vec = np.exp(1j * np.pi * n * np.sin(angle))
        # Beam pattern: |theta^H * a(angle)|^2
        beam_gain = np.abs(np.conj(theta_vector) @ steering_vec)**2
        beam_pattern.append(beam_gain)
    
    return np.array(beam_pattern)


# %%
channel_true, set_location_user_train = generate_irs_user_channel(
                location_user, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
channel_complex2real(channel_true)
##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0) # tf.placeholder(tf.float32, shape=(), name="power") #
    from0toN = tf.cast(tf.range(0, N, 1), tf.float32)

with tf.name_scope("channel_sensing"):
    hidden_size = 128
    layer_neuron_size = 128
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
    snr_normal = snr_dB#(snr_dB - 1) / np.sqrt(1.6666)  # normalization mean=1, std=1.6666
    
    theta_list = []
    
    for t in range(tau): # training pilot length
        if t == 0: # Initialization
            y_real = tf.ones([tf.shape(loc_input)[0], 32])
            h_old = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # hidden state
            c_old = tf.zeros([tf.shape(loc_input)[0], hidden_size]) # cell state
        h_old, c_old = RNN(tf.concat([y_real, snr_normal], axis=1), h_old, c_old)

        x1 = tf.nn.relu(h_old @ A1 + b1)
        x1 = BatchNormalization()(x1)
        x2 = tf.nn.relu(x1 @ A2 + b2)
        x2 = BatchNormalization()(x2)
        x3 = tf.nn.relu(x2 @ A3 + b3)
        x3 = BatchNormalization()(x3)

        # RIS 相位设计  
        ## Xiyu: this keeps the same to be the beamforming vector
        ris_her_unnorm = x3 @ A4 + b4
        ris_her_r = ris_her_unnorm[:, 0:N_ris]  # real part
        ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris] # imaginary part
        theta_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True)) # normalization per sample
        theta_real = tf.divide(ris_her_r , theta_tmp)
        theta_imag = tf.divide(ris_her_i , theta_tmp)
        theta = tf.concat([theta_real, theta_imag], axis=1)   
        theta_T = tf.reshape(theta, [-1, 1, 2 * N_ris])
        theta_list.append(theta_T[:, 0, :])
        
        # BS 观测下一个测量
        # uses the backscattered channel RIS-BD-RIS
        A_T_k1 = channel_bs_irs_user[:, :, :, 0] # ((batch_size, 2 * N_ris, 2 * N_ris))
        A_T_k = (A_T_k1)
        
        # Real and imaginary parts
        theta_A_k_T = tf.matmul(A_T_k, tf.transpose(theta_T, perm=[0,2,1]))  # (batch_size, 2 * N_ris, 1)  H*v
        
        
        h_d_plus_h_cas = theta_A_k_T
        h_d_plus_h_cas_re = h_d_plus_h_cas[:, 0:N_ris, :]
        h_d_plus_h_cas_im = h_d_plus_h_cas[:, N_ris:2*N_ris, :]
        noise = tf.complex(tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean=0.0, stddev=noiseSTD_per_dim), 
                           tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.complex(h_d_plus_h_cas_re, h_d_plus_h_cas_im) + noise   # h_hat = h + noise
        
        if USE_FFT:
            y_fft = tf.signal.fft(y_complex)
            y_real = tf.concat([tf.real(y_fft), tf.imag(y_fft)], axis=1) #/ tf.sqrt(lay['P'])
        else:
            y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1) #/ tf.sqrt(lay['P'])
        y_real = tf.reshape(y_real, [-1, 32])  # Reshape to match the input size of RNN
            
    h_old, c_old = RNN(tf.concat([y_real, snr_normal], axis=1), h_old, c_old)
    c_old = Dense(units=200, activation='linear')(c_old)
    c_old = Dense(units=200, activation='linear')(c_old)
    
    # Xiyu: Ouput RIS weights
    x1 = tf.nn.relu(h_old @ A1 + b1)
    x1 = BatchNormalization()(x1)
    x2 = tf.nn.relu(x1 @ A2 + b2)
    x2 = BatchNormalization()(x2)
    x3 = tf.nn.relu(x2 @ A3 + b3)
    x3 = BatchNormalization()(x3)
    ris_her_unnorm = x3 @ A4 + b4
    ris_her_r = ris_her_unnorm[:, 0:N_ris]  # real part
    ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris] # imaginary part
    theta_tmp = tf.sqrt(tf.reduce_sum(tf.square(ris_her_r) + tf.square(ris_her_i), axis=1, keepdims=True)) # normalization per sample
    theta_real = ris_her_r / theta_tmp
    theta_imag = ris_her_i / theta_tmp
    
    ## Xiyu: output two dimension: angle and distance of the BD
    loc_hat = Dense(units=2, activation='relu')(c_old)  
     
    ###### Loss function -- minimize negative of SNR
    ########## Calculate matrix G = I - (H_SI * theta_T * theta_T^H * H_SI^H) / ||H_SI * theta_T||^2
    theta_T_complex = tf.complex(theta_real, theta_imag)  # (batch, N_ris)
    theta_T_complex = tf.reshape(theta_T_complex, [-1, N_ris, 1])  # (batch, N_ris, 1)

    # H_SI as tf.constant for batch processing
    # H_SI_tf = tf.constant(np.array(channel_true[0]), dtype=tf.complex64)  # (N_ris, N_ris)
    H_SI_tf = H_SI_placeholder

    # H_SI * theta_T
    # H_SI_theta = tf.matmul(H_SI_tf, theta_T_complex)  # (batch, N_ris, 1) 

    # # Norm squared: ||H_SI * theta_T||^2
    # H_SI_theta_norm_sq = tf.reduce_sum(tf.abs(H_SI_theta) ** 2, axis=[1, 2], keepdims=True)  # (batch, 1, 1)
    # H_SI_theta_norm_sq_safe = tf.maximum(H_SI_theta_norm_sq, 1e-8)  # Prevent division by zero

    # # theta_T^H (Hermitian)
    # theta_T_H = tf.linalg.adjoint(theta_T_complex)  # (batch, 1, N_ris)

    # # H_SI * theta_T * theta_T^H * H_SI^H
    # middle = tf.matmul(H_SI_theta, theta_T_H)  # (batch, N_ris, N_ris)
    # middle = tf.matmul(middle, tf.linalg.adjoint(H_SI_tf))  # (batch, N_ris, N_ris)

    # # Identity matrix
    # I = tf.eye(N_ris, batch_shape=[tf.shape(theta_T_complex)[0]], dtype=tf.complex64)  # (batch, N_ris, N_ris)
    # # Ensure dtypes match for division: middle and H_SI_theta_norm_sq must be complex
    # # tf.complex will cast H_SI_theta_norm_sq (float) to complex for safe division
    # # G matrix
    # G = I - middle / tf.complex(H_SI_theta_norm_sq_safe, tf.zeros_like(H_SI_theta_norm_sq))  # (batch, N_ris, N_ris)

    ########## 
    batch_size = tf.shape(loc_input)[0]
    def compute_G_for_batch(H_SI_batch):
        s, u, v = tf.linalg.svd(H_SI_batch)
        s_inv = tf.where(s > 1e-8, 1.0 / s, tf.zeros_like(s))
        s_inv_complex = tf.cast(s_inv, tf.complex64)
        H_SI_pinv = tf.matmul(v, tf.matmul(tf.linalg.diag(s_inv_complex), tf.linalg.adjoint(u)))
        middle = tf.matmul(H_SI_batch, H_SI_pinv)
        I = tf.eye(N_ris, dtype=tf.complex64)
        return I - middle

    G = tf.map_fn(compute_G_for_batch, H_SI_tf, dtype=tf.complex64)  # (batch, N_ris, N_ris)
    ##########


    ########### Calculate the H_b_hat -- > Backscattered signal
    # Calculate steering vector using the first entry of loc_hat (angle and distance)
    # angle = loc_input[:, 0, 0]  # real angle
    # distance = loc_input[:, 1, 0]  # real distance
    # # Clamp distance to prevent numerical issues
    # distance = tf.maximum(distance, 0.1)  # Minimum distance of 0.1 meters

    # # For a ULA along y-axis, steering vector: a = exp(1j * pi * n * sin(angle)), n=0,...,N_ris-1
    # n = tf.cast(tf.range(N_ris), tf.float32)

    # # Clamp angle to valid range [-1, 1] for sin function
    # angle_clamped = tf.clip_by_value(angle, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    # steering_vec = tf.exp(1j * tf.cast(np.pi*n* tf.expand_dims(tf.sin(angle_clamped), 1), tf.complex64))  # shape: (batch, N_ris)
    # steering_vec = tf.expand_dims(steering_vec, axis=2)  # shape: (batch, N_ris, 1)
    # # Calculate H_b_hat as 
    # # Expand distance for broadcasting
    # distance_exp = tf.expand_dims(distance, axis=1)  # (batch, 1)

    # # Scalar coefficient -- backscattered signal
    # coeff = tf.cast( Wavelength**2 / (4 * np.pi * distance_exp)**2, tf.complex64) * tf.exp(1j * tf.cast(np.pi * 2*distance_exp / Wavelength, tf.complex64))  # (batch, 1)

    # # Outer product: steering_vec * steering_vec^H
    # steering_vec_H = tf.linalg.adjoint(steering_vec)  # (batch, 1, N_ris)
    # H_b_hat = coeff[:, tf.newaxis, tf.newaxis] * tf.matmul(steering_vec, steering_vec_H)  # (batch, N_ris, N_ris)

    #### Above calculation is estimated H_b. We can also get the real H_b channel
    # H_b_hat = tf.constant(channel_true[2].squeeze()[tf.newaxis, tf.newaxis,:], dtype=tf.complex64)
    H_b_hat = H_b_placeholder

    G_Hb = tf.matmul(G, H_b_hat)  # (batch, N_ris, N_ris)
    G_Hb_theta = tf.matmul(G_Hb, theta_T_complex)  # (batch, N_ris, 1)
    # # Norm squared over last two dims
    # # Calculate power with numerical stability
    power = tf.reduce_mean(lay['P'] *tf.abs(G_Hb_theta) ** 2, axis=[1, 2])  # (batch,)
    snr_eff = power / (N_ris - 1 )/2/ (noiseSTD_per_dim**2)

    sig_pow = lay['P'] * tf.reduce_sum( tf.abs( tf.matmul(H_b_hat, theta_T_complex) )**2, axis=1 )
    interference_pow = lay['P'] * tf.reduce_sum(tf.abs( tf.matmul(H_SI_tf, theta_T_complex))**2, axis=1)
    sinr = sig_pow / (interference_pow + 2*(noiseSTD_per_dim**2)*N_ris)

loss = - tf.log(tf.reduce_mean(sinr))

# xy_pred = loc_hat[:, ]
# xy_true = loc_input[:, 0,]
# loss = tf.reduce_mean(tf.square(xy_pred - xy_true))
loss1 = loss
loss2 = tf.constant(0.0) # tf.reduce_mean(tf.square(loc_hat - loc_input[:, :, 0])) # 
user_loss = tf.stack([loss1, loss2], name='user_loss')

# H_SI: (N_ris, N_ris), theta_T: (batch, 1, 2*N_ris)
# First, get theta_T in complex form for RIS

# 优化器
global_step = tf.train.get_or_create_global_step()
l2 = 1e-4
reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss_reg = loss + l2 * reg_term
lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=500, decay_rate=0.8)
training_op = tf.train.AdamOptimizer(lr).minimize(loss_reg, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Validation Set
channel_true_val, set_location_user_val = generate_irs_user_channel(
    None, location_ris_1, num_samples=val_size_order*delta_inv, Rician_factor=Rician_factor)
A_T_1_real_val, _ = channel_complex2real(channel_true_val)

feed_dict_val = {
    loc_input: np.array(set_location_user_val),
    channel_bs_irs_user: A_T_1_real_val,
    lay['P']: Pvec[0],
    H_SI_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
    H_b_placeholder: channel_true_val[2]
}
# %%
# Training
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
        epoch_u1_losses = []
        epoch_u2_losses = []
        
        for rnd_indices in range(batch_per_epoch):
            # Training set
            channel_true_train, set_location_user_train = generate_irs_user_channel(
                None, location_ris_1, num_samples=batch_size_order*delta_inv, Rician_factor=Rician_factor)
            A_T_1_real, _ = channel_complex2real(channel_true_train)
            H_SI_batch = channel_true_train[0]  # Self-interference channel
            H_b_batch = channel_true_train[2]   # Backscattered channel
            
            feed_dict_batch = {
                loc_input: np.array(set_location_user_train),
                channel_bs_irs_user: A_T_1_real,
                lay['P']: Pvec[0],
                H_SI_placeholder: np.tile(H_SI_batch[np.newaxis, :, :], (len(set_location_user_train), 1, 1)),  # Broadcast to batch
                H_b_placeholder: H_b_batch  # Shape: (batch, N_ris, N_ris)
            }
            
            _, train_loss_val, per_user_train,  snr_eff_val, sinr_val = sess.run(
                [training_op, loss, user_loss, snr_eff, sinr], feed_dict=feed_dict_batch
            )
            
            epoch_train_losses.append(train_loss_val)
            epoch_u1_losses.append(per_user_train[0])
            epoch_u2_losses.append(per_user_train[1])
            batch_iter += 1
        
        avg_train_loss = np.mean(epoch_train_losses)
        loss_val, per_user_val = sess.run([loss, user_loss], feed_dict=feed_dict_val)
        
        print('epoch', epoch,
              '  train_loss:%2.7f' % avg_train_loss,
              '  val_loss:%2.7f' % loss_val,
              '  best_val:%2.7f' % best_val)
        if epoch % 5 == 0:
            snr_db_val = 10*np.log10(np.mean(snr_eff_val))
            print(f" SNR_eff [dB]: {snr_db_val:.6f}, SINR[dB]: {10*np.log10(np.mean(sinr_val)):.6f}")

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
            channel_bs_irs_user: np.expand_dims(A_T_1_real_test, axis=0),
            lay['P']: Pvec[0],
            H_SI_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
            H_b_placeholder: channel_true_val[2]
        }
        
        mse_loss, phi_hat_test, theta_test = sess.run([loss, loc_hat, theta_list], feed_dict=feed_dict_test)
        train_losses.append(mse_loss)
        theta_test = np.array(theta_test)
        theta_test_cplx = theta_test[:, :, 0:N_ris] + 1j * theta_test[:, :, N_ris:2 * N_ris]
        theta_test_list.append(theta_test_cplx)
        location_list.append(location_user_target)
    
    # # 保存训练样本测试结果
    # model_filename = os.path.join(drive_save_path, f'validate_data_sample.mat')
    # sio.savemat(model_filename, dict(
    #     performance=np.array(train_losses),
    #     snr_const=snr_const,
    #     N=N, N_ris=N_ris,
    #     epoch=n_epochs, 
    #     delta_inv=delta_inv,
    #     mean_true_alpha=mean_true_alpha,
    #     loc_true_list=location_list,
    #     std_per_dim_alpha=std_per_dim_alpha,
    #     noiseSTD_per_dim=noiseSTD_per_dim, 
    #     tau=tau,
    #     theta_test_list=theta_test_list
    # ))
    
    # TESTING 
    
    sinr_test_set = []
    interference_pow_set = []
    test_loc = []

    for j in range(test_size):
        
        # 使用简化的信道生成函数
        channel_true_test, set_location_user_test = generate_irs_user_channel(
            None, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
        A_T_1_real_test, _ = channel_complex2real(channel_true_test)
        
        feed_dict_test = {
            loc_input: np.array(set_location_user_test),
            channel_bs_irs_user: A_T_1_real_test,
            lay['P']: Pvec[0],
            H_SI_placeholder: np.tile(channel_true_test[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
            H_b_placeholder: channel_true_test[2]
        }
        
        mse_loss, sinr_test, theta_test, interference_pow_test = sess.run([loss, sinr, theta_list, interference_pow], feed_dict=feed_dict_test)
        theta_test = np.array(theta_test)
        theta_test_cplx = theta_test[:, :, 0:N_ris] + 1j * theta_test[:, :, N_ris:2*N_ris]
        # radio_map, theta_test = generate_radio_map(theta_test_cplx)
        sinr_test_set.append(sinr_test)
        interference_pow_set.append(interference_pow)
        test_loc.append(set_location_user_test)

# Save the final results
# model_filename = os.path.join(drive_save_path, f'TEST_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}.mat')
# sio.savemat(model_filename, dict(
#     snr_const = snr_const,
#     N = N, N_ris = N_ris, tau = tau,
#     epoch = n_epochs, 
#     theta_test = theta_test_cplx,
#     loc_true = test_loc,
#     sinr_test = sinr_test_set,
#     interference_pow = interference_pow_set
# ))


# %%
def plot_beam_patterns(theta_complex, true_location, save_path=None):
    """
    Plot beam patterns at different time steps
    """
    tau_steps = theta_complex.shape[0]
    
    # Angle range for beam pattern visualization
    angles = np.linspace(-np.pi/2, np.pi/2, 181)
    angles_deg = angles * 180 / np.pi
    
    # Calculate beam patterns for each time step
    beam_patterns = []
    for t in range(tau_steps):
        theta_t = theta_complex[t, :]
        beam_pattern = calculate_beam_pattern(theta_t, angles)
        beam_patterns.append(beam_pattern)
    
    beam_patterns = np.array(beam_patterns)
    
    num_plots = 8 if tau_steps>1 else 1
    # Create subplots
    fig, axes = plt.subplots( 1, num_plots,  figsize=(16,8))
    # axes = axes.flatten()
    
    # Calculate true UE angle based on location
    d_k = true_location[0][1]
    true_angle_deg = true_location[0][0] * 180 / np.pi
    true_angle_sin = np.arcsin(np.clip(true_angle_deg, -1, 1)) * 180 / np.pi
    
    # Plot the last 4 time steps
    for t in range(num_plots):
        if num_plots == 1:
            ax = axes
        else:
            ax = axes[t]

        # Plot beam pattern in dB
        beam_dB = 10 * np.log10(beam_patterns[tau_steps - num_plots + t] / np.max(beam_patterns[tau_steps - num_plots + t]))
        ax.plot(angles_deg, beam_dB, 'b-', linewidth=2)
        
        # Mark true UE angle
        ax.axvline(true_angle_deg, color='r', linestyle='--', linewidth=2, label='True UE')
        
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Beam Gain (dB)')
        ax.set_title(f'Time Step {t+1}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-40, 0])
        ax.set_xlim([-90, 90])
        
        if t == 0:
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return beam_patterns

# %% 
# Load the saved model and perform TESTING
with tf.Session() as sess:
    # Restore the trained model
    saver.restore(sess, f'{drive_save_path}/params_RiK10_mono_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}') #
    
    # Example: test on new random user locations
    num_test_samples = 5
    test_losses = []
    test_theta_list = []
    test_location_list = []
    
    for _ in range(num_test_samples):
        # Generate a random user location
        location_user_test = generate_location(num_users)
        channel_true_test, set_location_user_test = generate_irs_user_channel(
            location_user_test, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
        A_T_1_real_test, _ = channel_complex2real(channel_true_test)
        print(set_location_user_test)
        H_SI = channel_true_test[0][np.newaxis, :, :]  # Self-interference channel
        feed_dict_test = {
            loc_input: np.array(set_location_user_test),
            channel_bs_irs_user: A_T_1_real_test,
            lay['P']: Pvec[0],
            H_SI_placeholder: H_SI,  # Add batch dimension
            H_b_placeholder: channel_true_test[2]    # Add batch dimension
        }

        mse_loss, phi_hat_test, theta_test,  snr_eff_test, sinr_test, sig_pow_test, interference_pow_test = sess.run(
                [loss, loc_hat, theta_list, snr_eff, sinr, sig_pow, interference_pow], 
                feed_dict=feed_dict_test)
        test_losses.append(mse_loss)
        theta_test = np.array(theta_test)
        theta_test_cplx = theta_test[:, :, 0:N_ris] + 1j * theta_test[:, :, N_ris:2*N_ris]
        test_theta_list.append(theta_test_cplx)
        test_location_list.append(location_user_test)

        print(f" SNR_eff: {10*np.log10(np.mean(snr_eff_test)):.6f}")
        print(f" SINR: {10*np.log10(np.mean(sinr_test)):.6f}")
        print(f" Signal Power: {np.mean(sig_pow_test):.6f}")
        print(f" Interference Power: {np.mean(interference_pow_test):.6f}")

        # visualize 
        plot_beam_patterns(
            theta_test_cplx.squeeze(), set_location_user_test,
            save_path=None#f'{drive_save_path}/demo_beam_patterns_{i+1}.png'
        )


        ### Optimum coherent beamformer 
        i1 = np.mod(np.arange(16), 16)
        
        H_b_hat = channel_true_test[2][0]
        if np.all(H_SI) ==0:
            theta_opti = 1/np.sqrt(N_ris)*np.exp(1j * np.pi * (i1 * np.sin(set_location_user_test[0][0][0]))) 
            H_B = np.matmul(np.conj(H_b_hat).transpose(), H_b_hat)

        else:
            # Rayleigh quotient
            H_B = np.matmul(np.conj(H_b_hat).transpose(), H_b_hat)
            H_A = np.matmul(np.conj(H_SI.squeeze()).transpose(), H_SI.squeeze()) + 2*(noiseSTD_per_dim**2)/Pvec[0] * np.eye(N_ris)
            eigv, s, _ = np.linalg.svd(np.matmul(np.linalg.inv(H_A), H_B ))
            theta_opti = eigv[:,0]


        midterm = np.matmul(H_SI.squeeze(),  np.linalg.pinv(H_SI.squeeze()) )
        G = np.eye(N_ris) - midterm 

        ## estimated H_b_hat 
        angle = set_location_user_test[0][ 0][0]  # real angle (batch,)
        distance = set_location_user_test[0][ 1][0]  # real distance (batch,)

        # For a ULA along y-axis, steering vector: a = exp(1j * pi * n * sin(angle)), n=0,...,N_ris-1
        n = np.arange(N_ris)  # (N_ris,)
        steering_vec = np.exp(1j * np.pi * n * np.sin(angle))
        coeff =  (Wavelength**2 / (4 * np.pi * distance)**2) * np.exp(1j * (2 * np.pi * distance / Wavelength))
        H_b_hat = coeff * np.matmul(steering_vec[:,np.newaxis], np.conj(steering_vec[:,np.newaxis]).transpose())  # (batch, N_ris, N_ris)

        sinr_opti = Pvec[0]*np.sum(np.abs(H_b_hat @ theta_opti)**2)/(Pvec[0]*np.sum(np.abs(H_SI.squeeze() @ theta_opti)**2) +2*(noiseSTD_per_dim**2)*N_ris )
        print(f"SINR of optimal beam: {10*np.log10(sinr_opti):.3f} ")
        print(f"Signal Power: {Pvec[0]*np.sum(np.abs(H_b_hat @ theta_opti)**2):.3f} ")
        print(f"Interference Power: {Pvec[0]*np.sum(np.abs(H_SI.squeeze() @ theta_opti)**2):.3f} ")

        plot_beam_patterns(
            theta_opti[np.newaxis,:], set_location_user_test,
            save_path=None
        )    
    
    # Save test results
    # test_result_filename = os.path.join(drive_save_path, 'test_results.mat')
    # sio.savemat(test_result_filename, dict(
    #     test_losses=np.array(test_losses),
    #     test_theta_list=test_theta_list,
    #     test_location_list=test_location_list
    # ))
    # print("Testing complete. Results saved to", test_result_filename)


# %%
