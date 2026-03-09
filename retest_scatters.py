# %%
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from keras.layers import BatchNormalization, Dense
from parse_args import parse_args
from channel_functions import *
from iter_gen_eig import *
from opti_transpose import *

args = parse_args()

def _generate_single_sample(loc_tx, loc_rx, n_scatters, n_tx, n_rx, rician):
    """Generate a single channel sample"""
    # Generate random locations
    bd_loc = generate_location_mimo(1, 'u')[0]
    scatter_loc = generate_location_mimo(n_scatters, 's')
    
    # Generate MIMO channels
    _, H_d, H_r, H_b = generate_mimo_channel(
        loc_tx, loc_rx, scatter_loc, bd_loc,
        N_tx_h=int(np.sqrt(n_tx)), N_tx_v=int(np.sqrt(n_tx)),
        N_rx_h=int(np.sqrt(n_rx)), N_rx_v=int(np.sqrt(n_rx)),
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

###############################################################
#            Params for scatters (Extension)
###############################################################
N_tx = 36
N_rx = 36
K = 1

N_scatter = 1
fc = args.fc
Wavelength = 3e8 / fc
ref_dis = 15*Wavelength
location_tx = np.array([0, 0, 0])
location_rx = np.array([0, 0, 0])

sig_pow_opti_recal = []
int_pow_opti_recal = []
sinr_opti_recal = []
rieman_opti_sinr_1b = []
sinr_test_1b = []
sinr_sweeping_1b = []
sinr_sweeping_1b_csi = []

sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweeping_2b = []
sinr_sweeping_2b_csi = []
iterative_generalized_eig = []

def dft_codebook(n, m):
    """
    Creates a DFT codebook for a ULA BS.

    Args:
        n (int): The number of antenna elements in the ULA.
        m (int): The number of beams in the codebook.
        angles_deg (list): A list of angles (in degrees) to form the beams.
        d (float): The antenna spacing (in wavelengths).

    Returns:
        codebook (numpy.ndarray): A numpy array of shape (n, m) representing the codebook.
    """
    dft_matrix = np.ones((n,m)) * complex(1, 0)
    for i in range(1, m+1):
        dft_matrix[:,i-1] = 1/np.sqrt(n) * np.array([np.exp(-1j * np.pi * j *(2 * i - 1 - m)/m) for j in range(n)])
    return dft_matrix

# %%
tau = 8
K = 5
CB = dft_codebook(N_tx, tau)
snr_const = [-5, 0, 5, 10, 15]
for i, snr in enumerate(snr_const):

       ###############################################################
       #            files with scatters (Extension)
       ###############################################################
       filename = os.path.join('Mo_mimo_sinr', \
              'ucloud/TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, tau, snr, K, N_scatter))
       data = scipy.io.loadmat(filename)

       sinr_test_2b.append(data['sinr_learned'].squeeze())

       rieman_opti_sinr_2b.append(data['sinr_optimal'].squeeze())
       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

       channel_true_val, loc_true = generate_irs_user_channel(
                            None, location_ris_1, num_samples=2000, Rician_factor=Rician_factor)
       Pvec = 10**(snr/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_tx)**2

       for j in range(test_size):

              _, H_d_test, H_r_test, H_b_test = generate_mimo_channel(
                                   location_tx, location_rx, Scatter_loc[j], BD_loc[j], N_tx, 1, N_rx, 1)

              H_I = H_d_test + H_r_test
              H_b = H_b_test

              ### Lower Bound Beam Sweeping

              observation1 = np.sqrt(Pvec)* (H_I + H_b) + \
                                          1/np.sqrt(2) *(np.random.randn(*H_I.shape) \
                                                               + 1j * np.random.randn(*H_I.shape))
              observation2 = np.sqrt(Pvec)*(H_I - H_b) + 1/np.sqrt(2) *(np.random.randn(*H_I.shape) \
                                                               + 1j * np.random.randn(*H_I.shape))
              H_b_hat = (observation1 - observation2)/2 / np.sqrt(Pvec)
              H_I_hat = (observation1 + observation2)/2 / np.sqrt(Pvec)

              si = np.abs(np.conj(CB).T @ H_I_hat @ CB)**2
              bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
              metric = bi.diagonal() 

              best_idx = np.argmax(metric)
              theta_test = CB[:, best_idx][:, np.newaxis]
              N_test = np.eye(N_tx) - ((H_I_hat @ theta_test) @ np.conj(H_I_hat @ theta_test).transpose())\
                                                 /np.linalg.norm(H_I_hat @ theta_test)**2
              v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
              
              sinr_sweeping_2b.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))


              ##### Beam sweeping with Perfect SI Channel Estimation

              si = np.abs(np.conj(CB).T @ H_I @ CB)**2
              bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
              metric = bi.diagonal() 

              best_idx = np.argmax(metric)
              theta_test = CB[:, best_idx][:, np.newaxis]
              N_test = np.eye(N_tx) - ((H_I @ theta_test) @ np.conj(H_I @ theta_test).transpose())\
                                                 /np.linalg.norm(H_I @ theta_test)**2
              v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
              sinr_sweeping_2b_csi.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))




sig_pow_opti_recal = np.mean(np.reshape(sig_pow_opti_recal, (len(snr_const), -1)), axis=1)
int_pow_opti_recal = np.mean(np.reshape(int_pow_opti_recal, (len(snr_const), -1)), axis=1)
sinr_opti_recal = np.mean(np.reshape(sinr_opti_recal, (len(snr_const), -1)), axis=1)

sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(snr_const), -1)), axis=1)



# %%
methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},      # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},      # Blue circles
    'beam_sweep': {'color': "#ff9941d2", 'marker': 'p'},     # Orange pentagons
    'beam_sweep_csi': {'color': '#ff7f0e', 'marker': 's'}  # Orange squares (CSI)
}

fig, ax = plt.subplots(1, 1, figsize=(4,3))

# w ≠ v (dashed lines)
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_test_2b], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in rieman_opti_sinr_2b], \
       marker=methods['iteropti']['marker'], linestyle='-', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='-', 
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b_csi.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='-', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping (Perfect SI CSI)')


# Create custom legend
from matplotlib.lines import Line2D

# Method legend (colors/markers)
method_legend = [
    Line2D([0], [0], color=methods['beam_sweep']['color'], 
           marker=methods['beam_sweep']['marker'], linestyle='None', 
           markersize=7, label='BeamSweeping ($\hat{\mathbf{H}}_{{SI}}$)'),
    Line2D([0], [0], color=methods['beam_sweep_csi']['color'], 
           marker=methods['beam_sweep_csi']['marker'], linestyle='None', 
           markersize=7, label='BeamSweeping ($\mathbf{H}_{{SI}}$)'),
    Line2D([0], [0], color=methods['proposed']['color'], 
           marker=methods['proposed']['marker'], linestyle='None', 
           markersize=7, label='Proposed'),
    Line2D([0], [0], color=methods['iteropti']['color'], 
           marker=methods['iteropti']['marker'], linestyle='None', 
           markersize=7, label='IterOpti'),
]

# Line style legend
style_legend = [
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1.2, 
           label=r'$\mathbf{w} = \mathbf{v}$'),
    Line2D([0], [0], color='grey', linestyle='-', linewidth=1.2, 
           label=r'$\mathbf{w} \neq \mathbf{v}$')
]

# Create two separate legends
legend1 = ax.legend(handles=method_legend, loc='upper left', 
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.5)
legend2 = ax.legend(handles=style_legend, loc='lower right', 
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.5)

# Add the first legend back (matplotlib removes it when creating the second)
ax.add_artist(legend1)

ax.set_xlabel('Effective SNR [dB]')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xticks(snr_const)
ax.set_ylim([-40, 30])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/sinr_snr.pdf', format = 'pdf', bbox_inches = 'tight')


# %%
#####################################################################################
# --------------------------------------------------------------------------------- #
#####################################################################################

snr_const = 10 
tau = [1,2,3,4,5,6,7]  

sig_pow_opti_recal = []
int_pow_opti_recal = []
sinr_opti_recal = []
num_scatters = 1


sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweeping_2b = []
sinr_sweeping_2b_csi = []
Pvec = 10**(snr/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_tx)**2

drive_save_path = 'Mo_mimo_sinr'

for i, n_tau in enumerate(tau):

       filename = os.path.join(drive_save_path, \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, N_scatter))
       data = scipy.io.loadmat(filename)

       sinr_test_2b.append(data['sinr_learned'].squeeze())

       rieman_opti_sinr_2b.append(data['sinr_optimal'].squeeze())

       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

       CB = dft_codebook(N_tx, n_tau)

       for j in range(H_d_test.shape[0]):

              _, H_d_test, H_r_test, H_b_test = generate_mimo_channel(
                     location_tx, location_rx, Scatter_loc[j], BD_loc[j], N_tx, 1, N_rx, 1)
            
              H_I = H_d_test + H_r_test
              H_b = H_b_test

              ### Lower Bound Beam Sweeping

              observation1 = np.sqrt(Pvec)* (H_I + H_b) + \
                                          1/np.sqrt(2) *(np.random.randn(*H_I.shape) \
                                                               + 1j * np.random.randn(*H_I.shape))
              observation2 = np.sqrt(Pvec)*(H_I - H_b) + 1/np.sqrt(2) *(np.random.randn(*H_I.shape) \
                                                               + 1j * np.random.randn(*H_I.shape))
              H_b_hat = (observation1 - observation2)/2 / np.sqrt(Pvec)
              H_I_hat = (observation1 + observation2)/2 / np.sqrt(Pvec)
              
              #      si = np.abs(np.conj(CB).T @ H_SI @ CB)**2
              bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
              metric = bi.diagonal() #/ (si.diagonal() + 1/Pvec)
              best_idx = np.argmax(metric)
              theta_test = CB[:, best_idx][:, np.newaxis]
              N_test = np.eye(N_tx) - ((H_I_hat @ theta_test) @ np.conj(H_I_hat @ theta_test).transpose())\
                                                 /np.linalg.norm(H_I_hat @ theta_test)**2
              v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
              
              sinr_sweeping_2b.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))
              
              ### Lower Bound Beam Sweeping
              
              H_b_hat = observation1 - H_I*np.sqrt(Pvec)
              
              bi = np.abs(np.conj(CB).T @ H_b @ CB)**2
              metric = bi.diagonal() #/ (si.diagonal() + 1/Pvec)
              best_idx = np.argmax(metric)
              theta_test = CB[:, best_idx][:, np.newaxis]
              N_test = np.eye(N_tx) - ((H_I @ theta_test) @ np.conj(H_I @ theta_test).transpose())\
                                                 /np.linalg.norm(H_I @ theta_test)**2
              v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)

              sinr_sweeping_2b_csi.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))


sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(tau), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(tau), -1)), axis=1)

# %%
methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},      # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},      # Blue circles
    'beam_sweep': {'color': "#ff9941d2", 'marker': 'p'},     # Orange pentagons
    'beam_sweep_csi': {'color': '#ff7f0e', 'marker': 's'}  # Orange squares (CSI)
}
fig, ax = plt.subplots(1, 1, figsize=(4,3))

### w ≠ v (solid lines)
ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_2b], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6)        
ax.plot(tau, [10*np.log10(np.mean(p)) for p in rieman_opti_sinr_2b], \
       marker=methods['iteropti']['marker'], linestyle='-', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6)
ax.plot(tau, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='-', 
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6)
ax.plot(tau, 10*np.log10(sinr_sweeping_2b_csi.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='-', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping')
# Create custom legend
from matplotlib.lines import Line2D
# Method legend (colors/markers)
method_legend = [
    Line2D([0], [0], color=methods['beam_sweep']['color'], 
           marker=methods['beam_sweep']['marker'], linestyle='None', 
           markersize=7, label='BeamSweeping ($\hat{\mathbf{H}}_{{SI}}$)'),
    Line2D([0], [0], color=methods['beam_sweep_csi']['color'], 
           marker=methods['beam_sweep_csi']['marker'], linestyle='None', 
           markersize=7, label='BeamSweeping ($\mathbf{H}_{{SI}}$)'),
    Line2D([0], [0], color=methods['proposed']['color'], 
           marker=methods['proposed']['marker'], linestyle='None', 
           markersize=7, label='Proposed'),
    Line2D([0], [0], color=methods['iteropti']['color'], 
           marker=methods['iteropti']['marker'], linestyle='None', 
           markersize=7, label='IterOpti'),
]
# Line style legend
style_legend = [
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1.2, 
           label=r'$\mathbf{w} = \mathbf{v}$'), 
    Line2D([0], [0], color='grey', linestyle='-', linewidth=1.2, 
           label=r'$\mathbf{w} \neq \mathbf{v}$')
]
# Create two separate legends
legend1 = ax.legend(handles=method_legend, loc='upper left', ncols =2,
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.6
                     )
                     
legend2 = ax.legend(handles=style_legend, loc='center left', 
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.6
                     )
legend2.set_bbox_to_anchor((0.0, 0.35))

ax.add_artist(legend1)

ax.set_xlabel('Preamble Length')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xticks(tau)
# ax.set_ylim([-30, 25])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/sinr_tau.pdf', format = 'pdf', bbox_inches = 'tight')

# %%
