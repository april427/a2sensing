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
tau = 16
K = 1
N_scatter = 5
CB = dft_codebook(N_tx, tau)
snr_const = [-5, 0, 5, 10, 15]
for i, snr in enumerate(snr_const):

       ###############################################################
       #            files with scatters (Extension)
       ###############################################################
       filename = os.path.join('Mo_mimo_sinr_modelsave', \
              'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, tau, snr, K, N_scatter))
       data = scipy.io.loadmat(filename)

       sinr_test_2b.append(data['sinr_learned'].squeeze())

       rieman_opti_sinr_2b.append(data['sinr_optimal'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, tau, snr, K, N_scatter))
       data1b = scipy.io.loadmat(filename)
       sinr_test_1b.append(data1b['sinr_learned'].squeeze())
       rieman_opti_sinr_1b.append(data1b['sinr_optimal'].squeeze())

       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

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

              bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
              metric = bi.diagonal() 

              best_idx = np.argmax(metric)
              theta_test = CB[:, best_idx][:, np.newaxis]
              N_test = np.eye(N_tx) - ((H_I_hat @ theta_test) @ np.conj(H_I_hat @ theta_test).transpose())\
                                                 /np.linalg.norm(H_I_hat @ theta_test)**2
              v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
              
              sinr_sweeping_2b.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))
              sinr_sweeping_1b.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))

              ##### Beam sweeping with Perfect SI Channel Estimation

              bi = np.abs(np.conj(CB).T @ H_b @ CB)**2
              metric = bi.diagonal() 

              best_idx = np.argmax(metric)
              theta_test = CB[:, best_idx][:, np.newaxis]
              N_test = np.eye(N_tx) - ((H_I @ theta_test) @ np.conj(H_I @ theta_test).transpose())\
                                                 /np.linalg.norm(H_I @ theta_test)**2
              v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
              sinr_sweeping_2b_csi.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))
              sinr_sweeping_1b_csi.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))



sig_pow_opti_recal = np.mean(np.reshape(sig_pow_opti_recal, (len(snr_const), -1)), axis=1)
int_pow_opti_recal = np.mean(np.reshape(int_pow_opti_recal, (len(snr_const), -1)), axis=1)
sinr_opti_recal = np.mean(np.reshape(sinr_opti_recal, (len(snr_const), -1)), axis=1)

sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(snr_const), -1)), axis=1)
sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(snr_const), -1)), axis=1)
sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(snr_const), -1)), axis=1)


# %%
methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},      # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},      # Blue circles
    'beam_sweep': {'color': "#ff9941d2", 'marker': 'p'},     # Orange pentagons
    'beam_sweep_csi': {'color': '#ff7f0e', 'marker': 's'}  # Orange squares (CSI)
}

fig, ax = plt.subplots(1, 1, figsize=(4,3))
## w = v (dashed lines)
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_test_1b], \
       marker=methods['proposed']['marker'], linestyle='--',
       color=methods['proposed']['color'], linewidth=1.2, markersize=6,
       label='Proposed')
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in rieman_opti_sinr_1b], \
       marker=methods['iteropti']['marker'], linestyle='--',
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6,
       label='IterOpti')
ax.plot(snr_const, 10*np.log10(sinr_sweeping_1b.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='--',
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping (Estimated SI CSI)')

ax.plot(snr_const, 10*np.log10(sinr_sweeping_1b_csi.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='--',
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping (Perfect SI CSI)')

## w ≠ v (dashed lines)
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
ax.set_ylim([-20, 20])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/scatter_sinr_snr.pdf', format = 'pdf', bbox_inches = 'tight')


# %%
#####################################################################################
# -----------------------------Pilot Length------------------------------------ #
#####################################################################################

snr_const = 10 
tau = [1,2,3,4,5,6,7,8,12,16,20,24]  
K = 1

sig_pow_opti_recal = []
int_pow_opti_recal = []
sinr_opti_recal = []
num_scatters = 1

sinr_test_1b = []
rieman_opti_sinr_1b = []
sinr_sweeping_1b = []
sinr_sweeping_1b_csi = []

sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweeping_2b = []
sinr_sweeping_2b_csi = []
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_tx)**2

drive_save_path = 'Mo_mimo_sinr'

for i, n_tau in enumerate(tau):

       filename = os.path.join(drive_save_path, \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)
       sinr_test_2b.append(data['sinr_learned'].squeeze())
       rieman_opti_sinr_2b.append(data['sinr_optimal'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_1b', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data1b = scipy.io.loadmat(filename)
       sinr_test_1b.append(data1b['sinr_learned'].squeeze())
       rieman_opti_sinr_1b.append(data1b['sinr_optimal'].squeeze())

       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

       CB = dft_codebook(N_tx, n_tau)

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
              sinr_sweeping_1b.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))
              
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
              sinr_sweeping_1b_csi.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))


sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(tau), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(tau), -1)), axis=1)
sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(tau), -1)), axis=1)
sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(tau), -1)), axis=1)

rieman_opti_sinr_1b[0] = rieman_opti_sinr_1b[1]
rieman_opti_sinr_2b[0] = rieman_opti_sinr_2b[1]

# %%
methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},      # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},      # Blue circles
    'beam_sweep': {'color': "#ff9941d2", 'marker': 'p'},     # Orange pentagons
    'beam_sweep_csi': {'color': '#ff7f0e', 'marker': 's'}  # Orange squares (CSI)
}
tau_positions = np.arange(len(tau))
tau_tick_positions = [0, 1, 2, 3, 4, 5, 6, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]
tau_tick_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '...', '12', '...', '16', '...', '20', '...', '24']
fig, ax = plt.subplots(1, 1, figsize=(4,3))

### w = v (dashed lines)
ax.plot(tau_positions, [10*np.log10(np.mean(p)) for p in sinr_test_1b], \
       marker=methods['proposed']['marker'], linestyle='--', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6,
       label='Proposed')        
ax.plot(tau_positions, [10*np.log10(np.mean(p)) for p in rieman_opti_sinr_1b], \
       marker=methods['iteropti']['marker'], linestyle='--', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6,
       label='IterOpti')
ax.plot(tau_positions, 10*np.log10(sinr_sweeping_1b.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='--', 
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping')
ax.plot(tau_positions, 10*np.log10(sinr_sweeping_1b_csi.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='--', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping')

### w ≠ v (solid lines)
ax.plot(tau_positions, [10*np.log10(np.mean(p)) for p in sinr_test_2b], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6)        
ax.plot(tau_positions, [10*np.log10(np.mean(p)) for p in rieman_opti_sinr_2b], \
       marker=methods['iteropti']['marker'], linestyle='-', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6)
ax.plot(tau_positions, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='-', 
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6)
ax.plot(tau_positions, 10*np.log10(sinr_sweeping_2b_csi.squeeze()), \
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
legend1 = ax.legend(handles=method_legend, loc='lower right', ncols =2,
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.6
                     )
                     
legend2 = ax.legend(handles=style_legend, loc='center left', 
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.6
                     )
legend2.set_bbox_to_anchor((0.0, 0.35))

ax.add_artist(legend1)

ax.set_xlabel('Preamble Length')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xlim(-0.3, len(tau) - 0.7)
ax.set_xticks(tau_tick_positions)
ax.set_xticklabels(tau_tick_labels)
ax.set_ylim([-12, 10])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/scatter_sinr_tau.pdf', format = 'pdf', bbox_inches = 'tight')

# %%
#####################################################################################
# ----------------------------- Multiple Scatters --------------------------------- #
#####################################################################################

snr_const = 10 
tau = [8,12,16,20,24]  
K = 1

sig_pow_opti_recal = []
int_pow_opti_recal = []
sinr_opti_recal = []
num_scatters = 5


sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweeping_2b = []
sinr_sweeping_2b_csi = []

sinr_test_1b = []
rieman_opti_sinr_1b = []
sinr_sweeping_1b = []
sinr_sweeping_1b_csi = []
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_tx)**2

for i, n_tau in enumerate(tau):

       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_2b.append(data['sinr_learned'].squeeze())

       rieman_opti_sinr_2b.append(data['sinr_optimal'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave_K10', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_1b.append(data['sinr_learned'].squeeze())

       rieman_opti_sinr_1b.append(data['sinr_optimal'].squeeze())

       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

       CB = dft_codebook(N_tx, n_tau)

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
              sinr_sweeping_1b.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))
              
              ### Lower Bound Beam Sweeping with perfect CSI
              
              bi = np.abs(np.conj(CB).T @ H_b @ CB)**2
              metric = bi.diagonal() #/ (si.diagonal() + 1/Pvec)
              best_idx = np.argmax(metric)
              theta_test = CB[:, best_idx][:, np.newaxis]
              N_test = np.eye(N_tx) - ((H_I @ theta_test) @ np.conj(H_I @ theta_test).transpose())\
                                                 /np.linalg.norm(H_I @ theta_test)**2
              v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)

              sinr_sweeping_2b_csi.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))
              sinr_sweeping_1b_csi.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))


sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(tau), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(tau), -1)), axis=1)
sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(tau), -1)), axis=1)
sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(tau), -1)), axis=1)

# %%
methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},      # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},      # Blue circles
    'beam_sweep': {'color': "#ff9941d2", 'marker': 'p'},     # Orange pentagons
    'beam_sweep_csi': {'color': '#ff7f0e', 'marker': 's'}  # Orange squares (CSI)
}
fig, ax = plt.subplots(1, 1, figsize=(4,3))
rieman_opti_sinr_2b[0] = rieman_opti_sinr_2b[1]
### w = v (dashed lines)
ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_1b], \
       marker=methods['proposed']['marker'], linestyle='--', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6,
       label='Proposed')        
ax.plot(tau, [10*np.log10(np.mean(p)) for p in rieman_opti_sinr_1b], \
       marker=methods['iteropti']['marker'], linestyle='--', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6,
       label='IterOpti')    
ax.plot(tau, 10*np.log10(sinr_sweeping_1b.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='--', 
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping')
ax.plot(tau, 10*np.log10(sinr_sweeping_1b_csi.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='--', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping')     
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
legend1 = ax.legend(handles=method_legend, loc='lower left', ncols =2,
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
# plt.savefig('figs/multiscatter_sinr_tau.pdf', format = 'pdf', bbox_inches = 'tight')

# %%
#####################################################################################
# ------------------------- Cell, Hidden state or both ----------------------------- #
#####################################################################################

snr_const = 10 
tau = [8,16,24]  
K = 1

sig_pow_opti_recal = []
int_pow_opti_recal = []
sinr_opti_recal = []
num_scatters = 5

sinr_test_2_lstm = []
sinr_test_2_lstm_fixedSI = []
sinr_test_both = []
sinr_test_hidden = []
sinr_test_cell = []

sinr_2_lstm_opti_sinr = []
sinr_2_lstm_fixedSI_opti_sinr = []

Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_tx)**2

for i, n_tau in enumerate(tau):
       filename = os.path.join('Mo_mimo_sinr', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_2_lstm_fixedSI.append(data['sinr_learned'].squeeze())
       sinr_2_lstm_fixedSI_opti_sinr.append(data['sinr_optimal'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_2_lstm.append(data['sinr_learned'].squeeze())
       sinr_2_lstm_opti_sinr.append(data['sinr_optimal'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_both.append(data['sinr_learned'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm_hidden', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_hidden.append(data['sinr_learned'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm_cell', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_cell.append(data['sinr_learned'].squeeze())

       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

       CB = dft_codebook(N_tx, n_tau)


methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},      # Red diamonds
    'hidden': {'color': '#1f77b4', 'marker': 'o'},      # Blue circles
    'cell': {'color': '#ff7f0e', 'marker': 's'},  # Orange squares
    'two_lstm': {'color': '#2ca02c', 'marker': '^'},  # Green triangles
    'fixedSI': {'color': '#9467bd', 'marker': 'p'}  # Purple pentagons
}
fig, ax = plt.subplots(1, 1, figsize=(4,3))

ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_2_lstm], \
       marker=methods['two_lstm']['marker'], linestyle='-',
       color=methods['two_lstm']['color'], linewidth=1.2, markersize=6,
       label='Two LSTM') 

ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_2_lstm_fixedSI], \
       marker=methods['fixedSI']['marker'], linestyle='-',
       color=methods['fixedSI']['color'], linewidth=1.2, markersize=6,
       label='2 LSTM (Fixed SI)')

ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_both], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6,
       label='Proposed')       

ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_hidden], \
       marker=methods['hidden']['marker'], linestyle='-', 
       color=methods['hidden']['color'], linewidth=1.2, markersize=6)        

ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_cell], \
       marker=methods['cell']['marker'], linestyle='-',   
       color=methods['cell']['color'], linewidth=1.2, markersize=6)

ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_2_lstm_opti_sinr], \
       marker=methods['two_lstm']['marker'], linestyle='--',
       color=methods['two_lstm']['color'], linewidth=1.2, markersize=6)

ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_2_lstm_fixedSI_opti_sinr], \
       marker=methods['fixedSI']['marker'], linestyle='--',
       color=methods['fixedSI']['color'], linewidth=1.2, markersize=6)


ax.legend([ax.lines[0], ax.lines[1], ax.lines[2], ax.lines[3], ax.lines[4], ax.lines[5], ax.lines[6]],\
           ['2 LSTM', '2 LSTM (Fixed SI)', '1 LSTM both', '1 Hidden State Only', '1 Cell State Only', '2 LSTM (Optimal)', '2 LSTM (Fixed SI) (Optimal)'],\
           loc='lower right', fontsize=9, frameon=True, fancybox=True, framealpha=0.6)

ax.set_xlabel('Preamble Length')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xticks(tau)
# ax.set_ylim([-30, 25])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# %%
