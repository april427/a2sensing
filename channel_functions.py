import numpy as np
from parse_args import parse_args
import matplotlib.pyplot as plt

args = parse_args()
N_ris = args.N_ris
num_users = args.num_users
fc = 10e9
Wavelength = 3e8 / fc # Wavelength for 10 GHz
tau = args.tau
location_ris_1 = np.array([0, 0, -20])       # This RIS is our BS
Rician_factor = args.rician_factor

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


    return location_user


def generate_irs_user_channel(user_locations, location_irs, num_samples=1, Rician_factor=10, scale_factor=0, irs_Nh=16, x_BD = 1):

    num_elements_irs = N_ris
    if user_locations is None:
        num_user = num_users  # 使用全局变量
    else:
        num_user = num_users#user_locations.shape[0] if user_locations.ndim == 2 else user_locations.shape[1]
    
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

        Rician_factor = 10**(Rician_factor/10)

        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[ num_elements_irs, num_elements_irs, num_user])

        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] * aoa_irs_cos_z[k] + i2 * aoa_irs_z[k])) # steering vector norm is N_ris
            a_irs_user = a_irs_user[:, np.newaxis]

            tmp[ :,:, k] = np.sqrt(Rician_factor/(1+Rician_factor)) *(a_irs_user @ np.transpose( a_irs_user)) + np.sqrt(1/(1+Rician_factor)) * tmp[:,:, k]
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

# def generate_radio_map(theta_test):
#     """generate radio map"""
#     x_lowerlimit, x_upperlimit = -35, -15
#     y_lowerlimit, y_upperlimit = 25, 55.5
#     z_fixed = -20
    
#     x_range = int((x_upperlimit - x_lowerlimit) / 0.5) + 1
#     y_range = int((y_upperlimit - y_lowerlimit) / 0.5) + 1
#     radio_map = np.zeros([x_range, y_range, tau])

#     for x_i in range(x_range):
#         for y_i in range(y_range):
#             coordinate_k = np.array([x_lowerlimit + x_i * 0.5, y_lowerlimit + y_i * 0.5, z_fixed])
#             location_user = np.empty([num_users, 3])
#             location_user[0, :] = coordinate_k
            
#             # 这里改用简化的信道生成函数
#             channel_true, set_location_user_train = generate_irs_user_channel(
#                 location_user, location_ris_1, num_samples=1, Rician_factor=Rician_factor)
#             A_T_real, _ = channel_complex2real(channel_true)
#             RSS_offline = generate_RSS_adaptive(A_T_real, theta_test, Pvec[0])
#             radio_map[x_i, y_i, :] = RSS_offline
            
#     return radio_map, theta_test

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

def calc_beam_pattern(theta_vector, angles, v_vector=None):
    """Calculate beam pattern for given beamforming vector and angles"""
    beam_pattern = []
    n = np.arange(N_ris)
    
    for angle in angles:
        # Steering vector for uniform linear array
        steering_vec = np.exp(1j * np.pi * n * np.sin(angle))[:, np.newaxis]
        H = steering_vec @ steering_vec.T
        if v_vector is not None:
            beam_gain = np.abs(np.conj(v_vector[0,]).transpose() @ H @ theta_vector[0,])**2
        else:
            beam_gain = np.abs(np.conj(theta_vector).transpose() @ H @ theta_vector)**2
        beam_pattern.append(beam_gain)
    
    return np.array(beam_pattern)

def plot_beam_patterns(theta_complex, true_location, v_vector=None, save_path=None):
    """
    Plot beam patterns at different time steps
    """
    tau_steps = theta_complex.shape[0]
    
    # Angle range for beam pattern visualization
    angles = np.linspace(-np.pi/2, np.pi/2, 360)
    angles_deg = angles * 180 / np.pi
    
    # Calculate beam patterns for each time step
    beam_patterns = []
    for t in range(tau_steps):
        theta_t = theta_complex[t, :]
        if v_vector is not None:
            v_t = v_vector[t,:]
            beam_pattern = calc_beam_pattern(theta_t, angles, v_t)
        else:
            beam_pattern = calc_beam_pattern(theta_t, angles)   
        beam_patterns.append(beam_pattern)
    
    beam_patterns = np.array(beam_patterns)
    
    num_plots = tau_steps
    # Create subplots
    if tau_steps>1:
        fig, axes = plt.subplots( 2, int(num_plots/2),  figsize=(16,8))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(4,3))
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
            ax = axes[int(np.floor(t/(num_plots/2))), int(np.mod(t, (num_plots/2)))]

        # Plot beam pattern in dB
        beam_dB = 10 * np.log10(beam_patterns[tau_steps - num_plots + t] / np.max(beam_patterns[tau_steps - num_plots + t]))
        beam_dB = beam_dB.squeeze()
        ax.plot(angles_deg, beam_dB, 'b-', linewidth=2)
        
        # Mark true UE angle
        ax.axvline(true_angle_deg, color='r', linestyle='--', linewidth=2, label='True UE')
        
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Beam Gain (dB)')
        ax.set_title(f'Time Step {t+1}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-60, 0])
        ax.set_xlim([-90, 90])
        
        if t == 0:
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return beam_patterns