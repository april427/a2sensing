import numpy as np
from parse_args import parse_args
import matplotlib.pyplot as plt

args = parse_args()
N_bs = args.N_bs
N_ris = args.N_ris
num_users = args.num_users
fc = args.fc
Wavelength = 3e8 / fc # Wavelength for 10 GHz
tau = args.tau
location_ris_1 = np.array([0, 0, -20])      
Rician_factor = args.rician_factor

def path_loss_r(d1, wavelength, d2 = None,  type = 'backscatter'):   # return |beta|^2 in dB
    """Keyhole channel: pathloss of backscatter signal for Rician fading in dB. (4 pi d / wavelength)^4 ! """
    if d2 is None and type == 'backscatter':
        loss = 40*np.log10(4*np.pi/wavelength) + 40.0 * np.log10(d1 + 1e-8)
    elif d2 is not None and type == 'backscatter':
        loss = 40*np.log10(4*np.pi/wavelength) + 20.0 * np.log10(d1 + 1e-8) + 20.0 * np.log10(d2 + 1e-8)  
    elif type == 'direct':
        loss = 20*np.log10(4*np.pi/wavelength) + 20.0 * np.log10(d1 + 1e-8)
    return loss

def generate_location(num_users):
    """
    generate user position:  
    user is on a circle 
    """
    location_user = np.empty([num_users, 3])

    angle = np.random.uniform(-np.pi, np.pi)
    dis = Wavelength*12
    x1 = dis * np.cos(angle)
    y1 = dis * np.sin(angle)

    # x1 = np.random.uniform(-40, -10)
    # y1 = np.random.uniform(20, 60)
    z1 = -20.0
    location_user[0, :] = np.array([x1, y1, z1])


    return location_user


def generate_irs_user_channel(user_locations, location_irs, num_samples=1, \
                              Rician_factor=10, scale_factor=0, irs_Nh=16, x_BD = 1):

    # num_elements_irs = N_ris
    if user_locations is None:
        num_user = num_users  
    else:
        num_user = num_users
        #user_locations.shape[0] if user_locations.ndim == 2 else user_locations.shape[1]
    
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
            pathloss_irs_user.append(path_loss_r(d_k,wavelength, type='backscatter'))
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
        i1 = np.mod(np.arange(N_ris), irs_Nh)
        i2 = np.floor(np.arange(N_ris) / irs_Nh)

        Rician_factor = 10**(Rician_factor/10)

        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_ris, N_ris, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_ris, N_ris, num_user])

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

def generate_bistatic_channels(user_locations, location_bs, location_irs, num_samples=1, \
                              Rician_factor=8, irs_Nh=16, x_BD = [0,1]):

    
    num_user = num_users
    
    g1 = [] 
    g2 = []
    set_location_user = []
    H_b = []

    wavelength = Wavelength
    Rician_factor = 10**(Rician_factor/10) # linear scale

    # direct path channel
    d_bs_irs = np.linalg.norm(location_bs - location_irs)
    pathloss_direct_db = path_loss_r(d_bs_irs, wavelength, type='direct')

    bs_array = np.column_stack((
        np.full(N_bs, location_bs[0]),
        location_bs[1] + np.arange(0, N_bs) * (wavelength / 2)
    ))
    irs_array = np.column_stack((
        np.full(N_ris, location_irs[0]), 
        np.arange(0, N_ris, 1) * (wavelength/2) + location_irs[1]))

    H_d = np.zeros((N_ris, N_bs), dtype=complex)
    for i in range(N_ris):
        for j in range(N_bs):
            H_d[i, j] = np.exp(- 1j * 2 * np.pi * (np.linalg.norm(irs_array[i,:] - bs_array[j,:])) / wavelength)

    pathloss_direct = np.sqrt(10 ** ((-pathloss_direct_db) / 10))
    H_d = pathloss_direct *   H_d 
    normalize_factor = np.linalg.norm(H_d,'fro')
    H_d = H_d * np.sqrt(N_ris) * np.sqrt(N_bs)/ normalize_factor   # Normalization
    
    location_r = []

    for ii in range(num_samples):
        # BD location
        if user_locations is None:
            location_user = generate_location(num_user)
        elif user_locations.ndim >= 3:  # For multiple samples
            location_user = user_locations[ii, :, :]
        else:
            location_user = user_locations
        
        # Reflector location
        location_r = generate_location(1)
        
        # Pathloss and AoA calculation
        # BD-RX
        pathloss_bs_bd_rx_db = []
        aoa_irs_y = []
        aoa_irs_z = []
        aoa_irs_cos_z = []
        # BS-BD
        aoa_bs_y = []
        aoa_bs_z = []
        aoa_bs_cos_z = []
        
        for k in range(num_user):
            d_k1 = np.linalg.norm(location_user[k] - location_irs) # BD-IRS distance
            d_k2 = np.linalg.norm(location_user[k] - location_bs)  # BD-BS distance
            d_k_xy = np.linalg.norm(location_user[k][0:2] - location_irs[0:2])  # horizontal distance
            pathloss_bs_bd_rx_db.append(path_loss_r(d_k1, wavelength, d_k2, type='backscatter')-20)

            aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / (d_k_xy +1e-8)     # Sine of azimuth angle
            aoa_irs_z_cos_k = d_k_xy / (d_k1 + 1e-8)  # Cosine of Elevation angle
            aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / (d_k1 +1e-8)     # Sine of Elevation angle
            aoa_irs_y.append(aoa_irs_y_k)
            aoa_irs_z.append(aoa_irs_z_k)
            aoa_irs_cos_z.append(aoa_irs_z_cos_k)

            aoa_bs_y_k = (location_user[k][1] - location_bs[1]) / (np.linalg.norm(location_user[k][0:2] - location_bs[0:2]) +1e-8)     # Sine of azimuth angle
            aoa_bs_z_cos_k = np.linalg.norm(location_user[k][0:2] - location_bs[0:2]) / (d_k2 + 1e-8)  # Cosine of Elevation angle
            aoa_bs_z_k = (location_user[k][2] - location_bs[2]) / (d_k2 +1e-8)     # Sine of Elevation angle
            aoa_bs_y.append(aoa_bs_y_k)
            aoa_bs_z.append(aoa_bs_z_k)
            aoa_bs_cos_z.append(aoa_bs_z_cos_k)

            set_location_user.append(np.array([np.arcsin(aoa_irs_y_k), d_k1, d_k2])[:, np.newaxis])
        
        aoa_irs_y = np.array(aoa_irs_y)
        aoa_irs_z = np.array(aoa_irs_z)
        aoa_irs_cos_z = np.array(aoa_irs_cos_z)

        # Xiyu: This is considered when arrays are UPA
        i1 = np.mod(np.arange(N_ris), irs_Nh)
        i2 = np.floor(np.arange(N_ris) / irs_Nh)
        if N_bs == 0:
            j1 = np.array([1])
            j2 = np.array([1])
        else:
            j1 = np.mod(np.arange(N_bs), N_bs)
            j2 = np.floor(np.arange(N_bs) / N_bs)

        # BS-Reflector-RX channel
        for kr in range(len(location_r)):
            d_r1 = np.linalg.norm(location_r[kr] - location_irs) # BD-IRS distance
            d_r2 = np.linalg.norm(location_r[kr] - location_bs)  # BD-BS distance
            d_r_xy = np.linalg.norm(location_r[kr][0:2] - location_irs[0:2])  # horizontal distance
            pathloss_reflector_db = path_loss_r(d_r1, wavelength, d_r2, type='backscatter')-20
            pathloss_reflector = np.sqrt(10 ** ((-pathloss_reflector_db) / 10))

            aoa_irs_y_r = (location_r[kr][1] - location_irs[1]) / (d_r_xy +1e-8)     # Sine of azimuth angle
            aoa_irs_cos_z_r = d_r_xy / (np.linalg.norm(location_r[kr] - location_irs) + 1e-8)  # Cosine of Elevation angle
            aoa_irs_z_r = (location_r[kr][2] - location_irs[2]) / (np.linalg.norm(location_r[kr] - location_irs) +1e-8)     # Sine of Elevation angle
            aoa_bs_y_r = (location_r[kr][1] - location_bs[1]) / (np.linalg.norm(location_r[kr][0:2] - location_bs[0:2]) +1e-8)     # Sine of azimuth angle
            aoa_bs_cos_z_r = np.linalg.norm(location_r[kr][0:2] - location_bs[0:2]) / (np.linalg.norm(location_r[kr] - location_bs) + 1e-8)  # Cosine of Elevation angle
            aoa_bs_z_r = (location_r[kr][2] - location_bs[2]) / (np.linalg.norm(location_r[kr] - location_bs) +1e-8)    
            
            ## channel caused by reflectors
            # steering vectors
            a_r_rx = np.exp(1j * np.pi * (i1 * aoa_irs_y_r * aoa_irs_cos_z_r + i2 * aoa_irs_z_r)) # steering vector norm is N_ris
            a_r_rx = a_r_rx[:, np.newaxis]
            a_bs_r = np.exp(1j * np.pi * (j1 * aoa_bs_y_r * aoa_bs_cos_z_r + j2 * aoa_bs_z_r)) # steering vector norm is N_ris
            a_bs_r = a_bs_r[:, np.newaxis]
            # channel
            H_r = (a_r_rx @ np.transpose(np.conj(a_bs_r))) 
            H_r = H_r * pathloss_reflector / normalize_factor 

        tmp = np.zeros([ N_ris, N_bs, num_user], dtype = complex)

        for k in range(num_user):
            pathloss_bs_bd_rx = np.sqrt( 10 ** ((-pathloss_bs_bd_rx_db[k]) / 10) )
            a_bd_rx = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] * aoa_irs_cos_z[k] + i2 * aoa_irs_z[k])) # steering vector norm is N_ris
            a_bd_rx = a_bd_rx[:, np.newaxis]

            a_bs_bd = np.exp(1j * np.pi * (j1 * aoa_bs_y[k] * aoa_bs_cos_z[k] + j2 * aoa_bs_z[k])) # steering vector norm is N_ris
            a_bs_bd = a_bs_bd[:, np.newaxis]

            tmp[ :,:, k] = (a_bd_rx @ np.transpose(np.conj(a_bs_bd))) 
            tmp[:,:,k] = tmp[:,:, k] * pathloss_bs_bd_rx/normalize_factor  # Backscattered channel

        g1.append( x_BD[0] * tmp + H_d[:,:,np.newaxis] + H_r[:,:,np.newaxis])
        g2.append( x_BD[1] * tmp + H_d[:,:,np.newaxis] + H_r[:,:,np.newaxis])
        H_b.append( tmp[:,:,k])

    channels = (H_d, np.array(g1), np.array(g2), np.array(H_b), np.array(H_r))
    # Channel typle: self-interference, IRS-user, backscattered
    return channels, set_location_user

def channel_complex2real(channels):
    """complex = [real, imagnary]"""
    H_SI, channel_irs_user, channel_backscattered = channels
    (num_sample, num_elements_irs, num_antenna_bs, num_user) = channel_irs_user.shape
    
    A_T_real = np.zeros([num_sample, 2 * num_antenna_bs, 2 * num_elements_irs,  num_user])
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, num_user], dtype=complex)
    
    for kk in range(num_user):
        channel_irs_user_k = channel_irs_user[:, :, :, kk]
        # (num_sample, num_elements_irs, num_elements_irs)
        channel_combine_irs = channel_irs_user_k.reshape(num_sample, num_antenna_bs, num_elements_irs)
        set_channel_combine_irs[:, :, :, kk] = channel_combine_irs
        
        A_tmp_tran = channel_combine_irs 
        A_tmp_real1 = np.concatenate([A_tmp_tran.real, - A_tmp_tran.imag], axis=2)
        A_tmp_real2 = np.concatenate([A_tmp_tran.imag, A_tmp_tran.real], axis=2)
        A_tmp_real = np.concatenate([A_tmp_real1, A_tmp_real2], axis=1)
        A_T_real[:, :, :, kk] = A_tmp_real
    return A_T_real, set_channel_combine_irs
    
def channel_bistatic_complex2real(channels):

    H_d, channel_g1, channel_g2, channel_backscattered, _ = channels
    (num_sample, num_elements_irs, num_antenna_bs, num_user) = channel_g1.shape
    
    A_T1_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs,  num_user])
    A_T2_real = A_T1_real.copy()
    
    for kk in range(num_user):
        channel_g1_k = channel_g1[:, :, :, kk]
        channel_g2_k = channel_g2[:, :, :, kk]
        # (num_sample, num_elements_irs, num_elements_irs)

        A_tmp_real1 = np.concatenate([channel_g1_k.real, - channel_g1_k.imag], axis=2)
        A_tmp_imag1 = np.concatenate([channel_g1_k.imag, channel_g1_k.real], axis=2)
        A_tmp1 = np.concatenate([A_tmp_real1, A_tmp_imag1], axis=1)

        A_tmp_real2 = np.concatenate([channel_g2_k.real, - channel_g2_k.imag], axis=2)
        A_tmp_imag2 = np.concatenate([channel_g2_k.imag, channel_g2_k.real], axis=2)
        A_tmp2 = np.concatenate([A_tmp_real2, A_tmp_imag2], axis=1)

        A_T1_real[:, :, :, kk] = A_tmp1
        A_T2_real[:, :, :, kk] = A_tmp2

    return A_T1_real, A_T2_real



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

## MIMO, UPA

def generate_location_mimo(num_users):
    """
    generate user position:  
    """
    location_user = np.empty([num_users, 3])
    dis = Wavelength*15

    azimuch_angle = np.random.uniform(-np.pi, np.pi)
    elevation_angle = np.random.uniform(np.pi/18, np.pi/2)

    x1 = dis * np.cos(elevation_angle) * np.cos(azimuch_angle)
    y1 = dis * np.cos(elevation_angle) * np.sin(azimuch_angle)
    z1 = dis * np.sin(elevation_angle)

    location_user[0, :] = np.array([x1, y1, z1])


    return location_user

def generate_upa_steering_vector(N_h, N_v, azimuth_angle, elevation_angle, \
                                 wavelength=None, near_field=False):
    """
    Generate steering vector for Uniform Planar Array (UPA)
    
    Parameters:
    -----------
    N_h : int
        Number of antenna elements in horizontal direction
    N_v : int
        Number of antenna elements in vertical direction
    azimuth_angle : float
        Azimuth angle (radians)
    elevation_angle : float
        Elevation angle (radians)
    wavelength : float, optional
        Wavelength (uses global Wavelength if None)
    
    Returns:
    --------
    steering_vector : numpy.ndarray
        Complex steering vector of shape (N_h * N_v,)
    """
    if wavelength is None:
        wavelength = Wavelength
    
    N_total = N_h * N_v
    
    # Antenna indices
    i1 = np.mod(np.arange(N_total), N_h)  # Horizontal indices
    i2 = np.floor(np.arange(N_total) / N_h)  # Vertical indices
    
    # Calculate sine/cosine components
    sin_azimuth = np.sin(azimuth_angle)
    cos_elevation = np.cos(elevation_angle)
    sin_elevation = np.sin(elevation_angle)
    
    # Steering vector for UPA
    steering_vector = np.exp(1j * np.pi * (i1 * sin_azimuth * cos_elevation + i2 * sin_elevation))

    # the middle of UPA as origin
    loc_ant_x = Wavelength/2/np.sqrt(2) * (i1 + 1 - (N_h+1)/2) 
    loc_ant_y = Wavelength/2/np.sqrt(2) * (i2 + 1 - (N_v+1)/2)
    if near_field:
        for ii1 in range(N_v):
            for jj1 in range(N_h):
                steering_vector = np.exp(1j * 2*np.pi / Wavelength * \
                                         (i1 * sin_azimuth * cos_elevation * loc_ant_x[ii1] \
                                          + i2 * sin_elevation * loc_ant_y[jj1]))

    
    return steering_vector

def generate_mimo_channel(tx_location, rx_location, scatter_location, bd_location, 
                         N_tx_h, N_tx_v, N_rx_h, N_rx_v, wavelength=None, 
                         Rician_factor=10):
    """
    Generate MIMO channel with direct path, scattered path, and BD scattered path
    
    Parameters:
    -----------
    tx_location : numpy.ndarray
        Transmitter location [x, y, z]
    rx_location : numpy.ndarray
        Receiver location [x, y, z]
    scatter_location : numpy.ndarray
        Scatterer location [x, y, z]
    bd_location : numpy.ndarray
        Backscatter device location [x, y, z]
    N_tx_h : int
        Number of TX antenna elements in horizontal direction
    N_tx_v : int
        Number of TX antenna elements in vertical direction
    N_rx_h : int
        Number of RX antenna elements in horizontal direction
    N_rx_v : int
        Number of RX antenna elements in vertical direction
    wavelength : float, optional
        Wavelength (uses global Wavelength if None)
    Rician_factor : float
        Rician K-factor in dB
        
    Returns:
    --------
    H_total : numpy.ndarray
        Total MIMO channel of shape (N_rx, N_tx)
    H_direct : numpy.ndarray
        Direct path channel
    H_scatter : numpy.ndarray
        Scattered path channel (via scatterer)
    H_bd : numpy.ndarray
        BD scattered path channel (via BD)
    """
    if wavelength is None:
        wavelength = Wavelength
    
    N_tx = N_tx_h * N_tx_v
    N_rx = N_rx_h * N_rx_v
    
    Rician_factor_linear = 10**(Rician_factor/10)
    
    # --- Direct Path Channel: TX to RX ---
    d_tx_rx = np.linalg.norm(rx_location - tx_location)
    d_tx_rx_xy = np.linalg.norm(rx_location[0:2] - tx_location[0:2])
    
    # Calculate angles for direct path
    azimuth_tx_rx = np.arctan2(rx_location[1] - tx_location[1], 
                                rx_location[0] - tx_location[0])
    elevation_tx_rx = np.arcsin((rx_location[2] - tx_location[2]) / (d_tx_rx + 1e-8))
    
    azimuth_rx_tx = np.arctan2(tx_location[1] - rx_location[1],
                                tx_location[0] - rx_location[0])
    elevation_rx_tx = np.arcsin((tx_location[2] - rx_location[2]) / (d_tx_rx + 1e-8))
    
    # Steering vectors for direct path
    a_tx_direct = generate_upa_steering_vector(N_tx_h, N_tx_v, azimuth_tx_rx, 
                                                elevation_tx_rx, wavelength)[:, np.newaxis]
    a_rx_direct = generate_upa_steering_vector(N_rx_h, N_rx_v, azimuth_rx_tx, 
                                                elevation_rx_tx, wavelength)[:, np.newaxis]
    
    # Direct path channel with pathloss
    pathloss_direct_db = path_loss_r(d_tx_rx, wavelength, type='direct')
    pathloss_direct = np.sqrt(10 ** ((-pathloss_direct_db) / 10))
    
    H_direct = pathloss_direct * (a_rx_direct @ a_tx_direct.T.conj())
    
    # --- Scattered Path Channel: TX to Scatterer to RX ---
    if scatter_location is not None:
        d_tx_scatter = np.linalg.norm(scatter_location - tx_location)
        d_scatter_rx = np.linalg.norm(rx_location - scatter_location)
        
        # TX to scatterer angles
        azimuth_tx_scatter = np.arctan2(scatter_location[1] - tx_location[1],
                                        scatter_location[0] - tx_location[0])
        elevation_tx_scatter = np.arcsin((scatter_location[2] - tx_location[2]) / (d_tx_scatter + 1e-8))
        
        # Scatterer to RX angles
        azimuth_scatter_rx = np.arctan2(rx_location[1] - scatter_location[1],
                                        rx_location[0] - scatter_location[0])
        elevation_scatter_rx = np.arcsin((rx_location[2] - scatter_location[2]) / (d_scatter_rx + 1e-8))
        
        # Steering vectors for scattered path
        a_tx_scatter = generate_upa_steering_vector(N_tx_h, N_tx_v, azimuth_tx_scatter,
                                                    elevation_tx_scatter, wavelength)[:, np.newaxis]
        a_rx_scatter = generate_upa_steering_vector(N_rx_h, N_rx_v, azimuth_scatter_rx,
                                                    elevation_scatter_rx, wavelength)[:, np.newaxis]
        
        # Scattered path channel with pathloss
        pathloss_scatter_db = path_loss_r(d_tx_scatter, wavelength, d_scatter_rx, type='backscatter')
        pathloss_scatter = np.sqrt(10 ** ((-pathloss_scatter_db) / 10))
        
        H_scatter = pathloss_scatter * (a_rx_scatter @ a_tx_scatter.T.conj())
    else:
        H_scatter = np.zeros((N_rx, N_tx), dtype=complex)
    
    # --- BD Scattered Path Channel: TX to BD to RX ---
    d_tx_bd = np.linalg.norm(bd_location - tx_location)
    d_bd_rx = np.linalg.norm(rx_location - bd_location)
    
    # TX to BD angles
    azimuth_tx_bd = np.arctan2(bd_location[1] - tx_location[1],
                                bd_location[0] - tx_location[0])
    elevation_tx_bd = np.arcsin((bd_location[2] - tx_location[2]) / (d_tx_bd + 1e-8))
    
    # BD to RX angles
    azimuth_bd_rx = np.arctan2(rx_location[1] - bd_location[1],
                                rx_location[0] - bd_location[0])
    elevation_bd_rx = np.arcsin((rx_location[2] - bd_location[2]) / (d_bd_rx + 1e-8))
    
    # Steering vectors for BD path
    a_tx_bd = generate_upa_steering_vector(N_tx_h, N_tx_v, azimuth_tx_bd,
                                            elevation_tx_bd, wavelength)[:, np.newaxis]
    a_rx_bd = generate_upa_steering_vector(N_rx_h, N_rx_v, azimuth_bd_rx,
                                            elevation_bd_rx, wavelength)[:, np.newaxis]
    
    # BD path channel with pathloss
    pathloss_bd_db = path_loss_r(d_tx_bd, wavelength, d_bd_rx, type='backscatter')
    pathloss_bd = np.sqrt(10 ** ((-pathloss_bd_db) / 10))
    
    H_bd = pathloss_bd * (a_rx_bd @ a_tx_bd.T.conj())
    
    # --- Total Channel with Rician Fading ---
    normalize_factor = np.linalg.norm(H_direct, 'fro')
    H_total = H_direct + H_scatter + H_bd 

    # scatters that are far away, modeled as Rayleigh fading
    # H_nlos = (np.random.normal(loc=0, scale=np.sqrt(0.5), size=(N_rx, N_tx)) +
            #   1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=(N_rx, N_tx)))
    
    # H_total = np.sqrt(Rician_factor_linear / (1 + Rician_factor_linear)) * H_tot + \
            #    np.sqrt(1 / (1 + Rician_factor_linear)) * H_nlos 

    
    # Normalize

    H_total = H_total * np.sqrt(N_rx * N_tx) / (normalize_factor + 1e-8)
    H_direct = H_direct * np.sqrt(N_rx * N_tx) / (normalize_factor + 1e-8)
    H_scatter = H_scatter * np.sqrt(N_rx * N_tx) / (normalize_factor + 1e-8)
    H_bd = H_bd * np.sqrt(N_rx * N_tx) / (normalize_factor + 1e-8)
    
    return H_total, H_direct, H_scatter, H_bd


