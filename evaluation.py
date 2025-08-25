import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
import os
import random
# from rnn_server_wall import generate_irs_user_channel, channel_complex2real, path_loss_r, generate_location

# System parameters (import from your main file or define here)
Wavelength = 3e8 / (26e9)  # Wavelength for 28 GHz
N_ris = 16
num_users = 2
tau = 16
snr_const = 35
ref_dis = 10
Pvec = 10**(snr_const/10) * N_ris / (Wavelength**2 / (4 *np.pi *ref_dis)**2)   # users at ref_dis has received SNR of 35, *N_ris is due to normalized noise 

Rician_factor = 10
location_ris_1 = np.array([-35, 40, -20])
y_wall = 55.5
noiseSTD_per_dim = np.sqrt(0.5)
drive_save_path = 'BD_beamInit'
N = 1

def path_loss_r(d, wavelength):   # return |beta|^2 in dB
    """pathloss for Rician fading in dB"""
    # wavelength = 3e8/fc
    loss = 40*np.log10(4*np.pi/wavelength) + 42.0 * np.log10(d + 1e-8)  
    return loss

def generate_location(num_users):
    """generate user position: U1:actual user U2:mirrored"""
    location_user = np.empty([num_users, 3])

    x1 = np.random.uniform(-34.5, -15)
    y1 = np.random.uniform(25, 55)
    z1 = -20.0
    location_user[0, :] = np.array([x1, y1, z1])

    if num_users >= 2:
        y2 = 2 * y_wall - y1
        location_user[1, :] = np.array([x1, y2, z1])

    return location_user


def generate_irs_user_channel(user_locations, location_irs, num_samples=1, Rician_factor=10, scale_factor=100, irs_Nh=16, x_BD = 1):

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
    # H_SI = H_SI * N_ris/ np.linalg.norm(H_SI,'fro')   # Normalization
    
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
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))


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


def visualize_beam_evolution(sess, new_ue_location, theta_list_tensor, loc_input_tensor, channel_tensor, lay_dict):
    """
    Visualize how beam patterns evolve over time for a new UE location
    """
    # Generate channel for the new UE location
    location_user_new = np.empty([num_users, 3])
    location_user_new[0, :] = new_ue_location
    if num_users >= 2:
        y2 = 2 * y_wall - new_ue_location[1]
        location_user_new[1, :] = np.array([new_ue_location[0], y2, new_ue_location[2]])
    
    channel_true_new, set_location_user_new = generate_irs_user_channel(
        location_user_new, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
    A_T_1_real_new, _ = channel_complex2real(channel_true_new)
    
    feed_dict_new = {
        loc_input_tensor: np.array(set_location_user_new),
        channel_tensor: A_T_1_real_new,
        lay_dict['P']: Pvec[0]
    }
    
    # Get the evolved beam patterns and location prediction
    theta_evolution = sess.run(theta_list_tensor, feed_dict=feed_dict_new)
    
    # Convert to complex beamforming vectors
    theta_evolution = np.array(theta_evolution)  # Shape: (tau, batch=1, 2*N_ris)
    theta_complex = theta_evolution[:, 0, 0:N_ris] + 1j * theta_evolution[:, 0, N_ris:2*N_ris]
    
    return theta_complex, set_location_user_new[0]

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
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Calculate true UE angle based on location
    d_k = true_location[0][1]
    true_angle_deg = true_location[0][0]
    true_angle_sin = np.arcsin(np.clip(true_angle_deg, -1, 1)) * 180 / np.pi
    
    
    # Plot first 8 time steps
    for t in range(min(8, tau_steps)):
        ax = axes[t]
        
        # Plot beam pattern in dB
        beam_dB = 10 * np.log10(beam_patterns[t] / np.max(beam_patterns[t]))
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

def plot_beam_evolution_animation(theta_complex, true_location, save_path=None):
    """
    Create an animated plot showing beam evolution
    """
    import matplotlib.animation as animation
    
    tau_steps = theta_complex.shape[0]
    angles = np.linspace(-np.pi/2, np.pi/2, 181)
    angles_deg = angles * 180 / np.pi
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize plot
    line, = ax.plot([], [], 'b-', linewidth=2, label='Beam Pattern')
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Beam Gain (dB)')
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 0)
    ax.grid(True, alpha=0.3)
    
    # True UE angle
    d_k = true_location[0][1]
    true_angle_deg = true_location[0][0]
    true_angle_sin = np.arcsin(np.clip(true_angle_deg, -1, 1)) * 180 / np.pi
    true_line = ax.axvline(true_angle_deg, color='r', linestyle='--', linewidth=2, label='True UE')
    
    ax.legend()
    
    def animate(t):
        theta_t = theta_complex[t, :]
        beam_pattern = calculate_beam_pattern(theta_t, angles)
        beam_dB = 10 * np.log10(beam_pattern / np.max(beam_pattern))
        
        line.set_data(angles_deg, beam_dB)
        ax.set_title(f'Beam Evolution - Time Step {t+1}/{tau_steps}')
        
        return line,
    
    anim = animation.FuncAnimation(fig, animate, frames=tau_steps, interval=500, blit=False, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
    
    plt.show()
    return anim

def plot_rss_evolution(theta_complex, channel_data, true_location, save_path=None):
    """Plot RSS evolution over time steps"""
    tau_steps = theta_complex.shape[0]
    rss_values = []
    
    # Calculate RSS for each time step
    for t in range(tau_steps):
        theta_t = theta_complex[t, :]
        
        # Combine real and imaginary parts
        theta_real_imag = np.concatenate([theta_t.real, theta_t.imag])
        theta_T = theta_real_imag.reshape(1, 1, 2 * N_ris)
        
        # Get channel for user 1 and user 2 (mirrored)
        A_T_k1 = channel_data[0, :, :, 0]
        A_T_k2 = channel_data[0, :, :, 1]
        A_T_k = (A_T_k1 + A_T_k2)
        
        # Calculate received signal
        theta_A_k_T = np.matmul(theta_T, A_T_k)
        theta_A_k_T_re = theta_A_k_T[0, 0, 0]
        theta_A_k_T_im = theta_A_k_T[0, 0, 1]
        
        # RSS calculation
        RSS_i = abs((np.sqrt(Pvec[0]) + 1j*0.0) * (theta_A_k_T_re + 1j*theta_A_k_T_im)) ** 2
        rss_values.append(RSS_i)
    
    # Plot RSS evolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, tau_steps + 1), rss_values, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Time Step')
    plt.ylabel('RSS (Received Signal Strength)')
    plt.title('RSS Evolution Over Time Steps')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, tau_steps + 1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return rss_values

def load_trained_model_and_visualize():
    """Load trained model and visualize beam evolution for different UE locations"""
    
    tf.reset_default_graph()
    
    # Recreate the exact placeholders from training
    loc_input = tf.placeholder(tf.float32, shape=(None, 2, num_users), name="loc_input")
    channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N_ris, num_users), name="channel_bs_irs_user")
    
    # Recreate the exact model architecture
    with tf.name_scope("array_response_construction"):
        lay = {}
        lay['P'] = tf.constant(1.0)
        from0toN = tf.cast(tf.range(0, N, 1), tf.float32)

    with tf.name_scope("channel_sensing"):
        hidden_size = 512
        he_init = tf.variance_scaling_initializer()
        
        # Recreate exact same variables with same names
        A1 = tf.get_variable("A1", shape=[hidden_size, 1024], dtype=tf.float32, initializer=he_init)
        A2 = tf.get_variable("A2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
        A3 = tf.get_variable("A3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
        A4 = tf.get_variable("A4", shape=[1024, 2*N_ris], dtype=tf.float32, initializer=he_init)
        
        b1 = tf.get_variable("b1", shape=[1024], dtype=tf.float32, initializer=he_init)
        b2 = tf.get_variable("b2", shape=[1024], dtype=tf.float32, initializer=he_init)
        b3 = tf.get_variable("b3", shape=[1024], dtype=tf.float32, initializer=he_init)
        b4 = tf.get_variable("b4", shape=[2*N_ris], dtype=tf.float32, initializer=he_init)
        from keras.layers import Dense, BatchNormalization
        
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
        # Recreate exact preprocessing
        snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
        snr_dB = tf.log(snr) / np.log(10)
        snr_normal = (snr_dB - 1) / np.sqrt(1.6666)
        
        theta_list = []
        
        # Recreate exact same loop structure
        for t in range(tau):
            if t == 0:  # Initialization
                y_real = tf.ones([tf.shape(loc_input)[0], 32])
                h_old = tf.zeros([tf.shape(loc_input)[0], hidden_size])
                c_old = tf.zeros([tf.shape(loc_input)[0], hidden_size])
            
            h_old, c_old = RNN(tf.concat([y_real, snr_normal], axis=1), h_old, c_old)
            
            # Feedforward network for beamforming
            x1 = tf.nn.relu(h_old @ A1 + b1)
            x1 = BatchNormalization()(x1)
            x2 = tf.nn.relu(x1 @ A2 + b2)
            x2 = BatchNormalization()(x2)
            x3 = tf.nn.relu(x2 @ A3 + b3)
            x3 = BatchNormalization()(x3)
            
            # RIS phase design
            ris_her_unnorm = x3 @ A4 + b4
            ris_her_r = ris_her_unnorm[:, 0:N_ris]
            ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris]
            theta_tmp = tf.sqrt(tf.square(ris_her_r) + tf.square(ris_her_i))
            theta_real = ris_her_r / theta_tmp
            theta_imag = ris_her_i / theta_tmp
            theta = tf.concat([theta_real, theta_imag], axis=1)
            theta_T = tf.reshape(theta, [-1, 1, 2 * N_ris])
            theta_list.append(theta_T[:, 0, :])

            # Channel observation
            A_T_k1 = channel_bs_irs_user[:, :, :, 0]
            A_T_k = (A_T_k1)
            
            theta_A_k_T = tf.matmul(A_T_k, tf.transpose(theta_T, perm=[0,2,1]))
            
            h_d_plus_h_cas = theta_A_k_T
            h_d_plus_h_cas_re = h_d_plus_h_cas[:, 0:N_ris, :]
            h_d_plus_h_cas_im = h_d_plus_h_cas[:, N_ris:2*N_ris, :]
            noise = tf.complex(tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean=0.0, stddev=noiseSTD_per_dim), 
                               tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean=0.0, stddev=noiseSTD_per_dim))
            y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.complex(h_d_plus_h_cas_re, h_d_plus_h_cas_im) + noise
            
            y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1) / tf.sqrt(lay['P'])
            y_real = tf.reshape(y_real, [-1, 32])
        
        # Final processing
        h_old, c_old = RNN(tf.concat([y_real, snr_normal], axis=1), h_old, c_old)
        c_old = Dense(units=200, activation='linear')(c_old)
        c_old = Dense(units=200, activation='linear')(c_old)
        loc_hat = Dense(units=2, activation='linear')(c_old)
    
    saver = tf.train.Saver()
    return saver


if __name__ == "__main__":
    # Test with specific UE locations

    # load the results
    # mat_file_path = os.path.join(drive_save_path, f'interpret_closeBS_fullRician_3D_1RIS_newcoordinateSISO_N_{N}_tau_{tau}_snr_{int(snr_const)}.mat')
    mat_file_path = os.path.join(drive_save_path, f'validate_data_sample.mat')
    mat_data = sio.loadmat(mat_file_path)
    theta_list = mat_data['theta_test_list']
    loc_input = mat_data['loc_true_list']
    performance = mat_data['performance'] 
    
    # Alternative: Generate synthetic beam patterns for demonstration
    print("\nGenerating synthetic beam patterns for demonstration...")
    
    for i, test_loc in enumerate(loc_input):
        print(f"\nDemo visualization for location {test_loc}")
        
        # Generate synthetic progressive beamforming (for demonstration)
        tau_demo = 8
        theta_demo = np.zeros((tau_demo, N_ris), dtype=complex)
        
        # True angle calculation
        d_k = test_loc[1]
        true_angle_sin = test_loc[0] 

        
        # Visualize
        location_demo = [test_loc]
        plot_beam_patterns(
            theta_list.squeeze(), location_demo,
            save_path=None#f'{drive_save_path}/demo_beam_patterns_{i+1}.png'
        )
        
        # plot_beam_evolution_animation(
        #     theta_list, location_demo,
        #     save_path=f'{drive_save_path}/demo_beam_evolution_{i+1}.gif'
        # )

    print(performance)
  