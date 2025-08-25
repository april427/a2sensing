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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#####################################################
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

    wavelength = Wavelength

    # Self-interference channel
        # The BS antennas Nr = Nt
    tx = np.arange(0, N_ris, 1) * (wavelength/2)
    rx = np.arange(N_ris, N_ris*2, 1) * (wavelength/2)
    H_SI = np.zeros((N_ris, N_ris), dtype=complex)
    # for i in range(N_ris):
    #     for j in range(N_ris):
    #         H_SI[i, j] = 1/(rx[j] - tx[i]) * np.exp(- 1j * 2 * np.pi * (rx[j] - tx[i]) / wavelength)
    # H_SI = 1e-6 * H_SI * N_ris/ np.linalg.norm(H_SI,'fro')   # Normalization
    
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
        
        # 生成IRS-用户信道
        # Xiyu: This is considered when the RIS is a rectangular array
        i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
        i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)
        
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_elements_irs, num_user])
        
        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] * aoa_irs_cos_z[k] + i2 * aoa_irs_z[k])) # steering vector norm is N_ris
            a_irs_user = a_irs_user[:, np.newaxis]

            tmp[:,:, k] = np.sqrt(Rician_factor/(1+Rician_factor)) *(a_irs_user @ np.transpose(np.conj( a_irs_user))) + np.sqrt(1/(1+Rician_factor)) * tmp[:,:, k]
            tmp[:,:,k] = tmp[:,:, k] * pathloss_irs_user[k] # Backscattered channel
            
        channel_irs_user.append( x_BD* tmp + H_SI[:,:,np.newaxis])
    
    # # 为保持与原代码兼容的数据结构，我们需要返回完整的通道元组，但BS相关通道为空
    # dummy_bs_user = np.zeros((num_samples, N_ris, num_user), dtype=complex)  # 空BS-用户通道
    # dummy_bs_irs = np.zeros((num_samples, N_ris, num_elements_irs), dtype=complex)  # 空BS-IRS通道
    
    channels = (H_SI, np.array(channel_irs_user), tmp) 
    # Channel typle: self-interference, IRS-user, backscattered
    return channels, set_location_user

def channel_complex2real(channels):
    """complex = [real, imagnary]"""
    H_SI, channel_irs_user, channel_backscattered = channels
    (num_sample, num_elements_irs, _, num_user) = channel_irs_user.shape
    num_antenna_bs = N_ris
    
    # 简化后只需要IRS-用户通道的实数表示
    A_T_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs, num_user])
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, num_user], dtype=complex)
    
    for kk in range(num_user):
        channel_irs_user_k = channel_irs_user[:, :, :, kk]
        # 直接将通道reshape至 (num_sample, num_elements_irs, num_elements_irs)  
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
N_ris = 16  # Number of RIS elements, treat RIS as BS for the mono-static case
num_users = 1  
params_system = (N_ris, N_ris, num_users)
Rician_factor = 10
location_user = None

# Sensing parameters
tau = 10  # Pilot length
snr_const = 20 
snr_const = np.array([snr_const]) 
ref_dis = 5
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)
            # BD at ref_dis has received SNR of snr_const. ||v||^2 = N_ris 

'Learning Parameters'
initial_run = 1   # 0: Continue training; 1: Starts from scratch
n_epochs = 2
learning_rate = 1e-3
batch_per_epoch = 400
batch_size_order = 16
batch_size_val = 10000
scale_factor = 1
test_size_order = 782

model_path = 

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
channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_ris, num_users), name="channel_bs_irs_user")
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

        'user 1 observes the next measurement'
        y_noiseless1 = tf.transpose(tf.conj(v1), perm=[0, 2, 1]) @ A_T_k1 @ v2
        noise1 = tf.complex(tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex1 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless1 + noise1
        y_complex1 = tf.reshape(y_complex1, [-1, 1])
        y_real = tf.concat([tf.real(y_complex1), tf.imag(y_complex1)], axis=1) #/ tf.sqrt(lay['P'])

        'user 1 design next receive beamformer based on  h_old1'
        h_old1, c_old1 = RNN1((y_real, h_old1, c_old1))
        v_her = MLP_user1_transmit(h_old1)  # this is actually the MLP for receive beamformer
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        v1 = tf.complex(v_her[:, 0:N_ris], v_her[:, N_ris:2 * N_ris])
        v1 = tf.reshape(v1, [-1, N_ris, 1])

        'user 1 design next transimit beamformer based on  h_old1'
        v_her = MLP_user1_receive(h_old1)  # this is actually the MLP for transimit beamformer
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        v2 = tf.complex(v_her[:, 0:N_ris], v_her[:, N_ris:2 * N_ris])
        v2 = tf.reshape(v2, [-1, N_ris, 1])

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

    sig_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex), perm=[0,2,1]) @ tf.matmul(H_b_hat, v2))**2
    interference_pow = lay['P'] * tf.abs(tf.transpose(tf.conj(v_complex), perm=[0,2,1]) @ tf.matmul(H_SI_tf, v2))**2
    sinr = sig_pow / (interference_pow + 2*(noiseSTD_per_dim**2)*N_ris)
    

####### Loss Function
loss = -sinr
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

###########  Validation Set
channel_true_val, set_location_user_val = generate_irs_user_channel(
    None, location_ris_1, num_samples=batch_size_val, Rician_factor=Rician_factor)
A_T_1_real_val, _ = channel_complex2real(channel_true_val)

feed_dict_val = {
    loc_input: np.array(set_location_user_val),
    channel_bs_irs_user: A_T_1_real_val,
    lay['P']: Pvec[0],
    H_SI_placeholder: np.tile(channel_true_val[0][np.newaxis, :, :], (len(set_location_user_val), 1, 1)),
    H_b_placeholder: channel_true_val[2]
}

###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, model_path)
    best_loss, opt_loss, rnd_loss = sess.run([loss, bf_gain_opt, bf_gain_rnd], feed_dict=feed_dict_val)
    print(-best_loss, opt_loss, rnd_loss)
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on
    no_increase = 0
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            alpha_train = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                           size=[batch_size_train, L]) \
                          + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                                  size=[batch_size_train, L])
            phi_1_train = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_train, L])
            phi_2_train = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_train, L])
            feed_dict_train = {alpha_input: alpha_train,
                               phi_input_1: phi_1_train,
                               phi_input_2: phi_2_train,
                               lay['P']: P_snr}

            sess.run(training_op, feed_dict=feed_dict_train)
            batch_iter += 1
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch', epoch, '  loss_test:%2.5f' % -loss_val, ' dB:%2.3f' % (10 * np.log10(-best_loss)),
              '  opt_test:%2.3f' % (10 * np.log10(opt_loss)), 'no_increase:', no_increase)
        if epoch % 10 == 9:  # Every 10 iterations it checks if the validation performace is improved, then saves parameters
            if loss_val < best_loss:
                save_path = saver.save(sess, model_path)
                best_loss = loss_val
                no_increase = 0
            else:
                no_increase = no_increase + 10

    # sio.savemat('./results/RNN_tau_'+str(tau)+'.mat',{'bf_gain_dB':(10*np.log10(-best_loss)),
    #                                         'bf_gain_opt_dB':(10*np.log10(opt_loss)),
    #                                         'bf_gain_rnd_dB':(10*np.log10(rnd_loss)),
    #                                         'N1_N2_tau_L':(N1,N2,tau,L), 'phi_min_max':(phi_min,phi_max),'snrdB':snrdB})
