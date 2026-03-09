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

# args = parse_args()

N_ris = 16
tau = 10  
snr_const = [-10,-5, 0, 5, 10, 15, 20,25]
ref_dis = 5

location_ris_1 = np.array([0, 0, -20])  

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

for i, snr in enumerate(snr_const):
        filename = os.path.join('BD_beamInit', \
            'TEST_mono_N_%d_tau_%d_snr_%d.mat' % (N_ris, tau, int(snr)))
        data = scipy.io.loadmat(filename)
        sinr_test_1b.append(data['sinr_test'].squeeze())
        rieman_opti_sinr_1b.append(data['rieman_opti_sinr'].squeeze())

        filename2 = os.path.join('BD_2beamsInit', \
            'TEST_mono_N_%d_tau_%d_snr_%d.mat' % (N_ris, tau, int(snr)))
        data2 = scipy.io.loadmat(filename2)
        sinr_test_2b.append(data2['sinr_test'].squeeze())
        rieman_opti_sinr_2b.append(data2['rieman_opti_sinr'].squeeze())


        channel_true_val, loc_true = generate_irs_user_channel(
                                None, location_ris_1, num_samples=2000, Rician_factor=Rician_factor)
        Pvec = 10**(snr/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2

        for j in range(len(loc_true)):
            loc = loc_true[j].squeeze()
            loc_cartesian = np.array([[loc[1]*np.cos(loc[0]), loc[1]*np.sin(loc[0]), -20]])

            H_SI = channel_true_val[0]
            H_b = channel_true_val[2][j]

            ### Lower Bound Beam Sweeping with Imperfect SI Channel Estimation
            CB = dft_codebook(N_ris, tau)
            H_SI_hat =   H_SI + (np.random.randn(*H_SI.shape) + 1j * np.random.randn(*H_SI.shape))/np.sqrt(2*Pvec)
            observation = np.sqrt(Pvec)*channel_true_val[1][j].squeeze() + 1/np.sqrt(2) *(np.random.randn(*channel_true_val[1][j].squeeze().shape) + 1j * np.random.randn(*channel_true_val[1][j].squeeze().shape))
            H_b_hat = observation - H_SI_hat*np.sqrt(Pvec)

            si = np.abs(np.conj(CB).T @ H_SI @ CB)**2
            bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
            metric = bi.diagonal() 

            best_idx = np.argmax(metric)
            theta_test = CB[:, best_idx][:, np.newaxis]
            N_test = np.eye(N_ris) - ((H_SI_hat @ theta_test) @ np.conj(H_SI_hat @ theta_test).transpose())\
                                            /np.linalg.norm(H_SI_hat @ theta_test)**2
            v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
            sinr_sweeping_1b.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_SI @ theta_test)**2 + 1))
            sinr_sweeping_2b.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_SI @ theta_test)**2 + 1))
            

              ##### Beam sweeping with Perfect SI Channel Estimation
            H_b_hat = observation - H_SI*np.sqrt(Pvec)

            si = np.abs(np.conj(CB).T @ H_SI @ CB)**2
            bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
            metric = bi.diagonal() 

            best_idx = np.argmax(metric)
            theta_test = CB[:, best_idx][:, np.newaxis]
            N_test = np.eye(N_ris) - ((H_SI @ theta_test) @ np.conj(H_SI @ theta_test).transpose())\
                                            /np.linalg.norm(H_SI @ theta_test)**2
            v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
            sinr_sweeping_1b_csi.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_SI @ theta_test)**2 + 1))
            sinr_sweeping_2b_csi.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_SI @ theta_test)**2 + 1))
            

              ##### Iterative generalized eigenvalue optimization
            A = np.sqrt(Pvec) * H_b
            B = np.sqrt(Pvec) * H_SI
       #      v_star, theta_star, sinr_star = alternating_xy(A, B, c=1, y0=None, max_iter=200, tol=1e-6, verbose=False)
            theta_star, _ = solve_transpose_with_manual_restarts(A, B, 1, restarts=5, seed=123)
            sinr_star = Pvec * np.abs(np.transpose(np.conj(theta_star)) @ H_b @ theta_star)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(theta_star)) @ H_SI @ theta_star)**2 + 1)
            iterative_generalized_eig.append(sinr_star)



sig_pow_opti_recal = np.mean(np.reshape(sig_pow_opti_recal, (len(snr_const), -1)), axis=1)
int_pow_opti_recal = np.mean(np.reshape(int_pow_opti_recal, (len(snr_const), -1)), axis=1)
sinr_opti_recal = np.mean(np.reshape(sinr_opti_recal, (len(snr_const), -1)), axis=1)
sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(snr_const), -1)), axis=1)
sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(snr_const), -1)), axis=1)
iterative_generalized_eig = np.mean(np.reshape(iterative_generalized_eig, (len(snr_const), -1)), axis=1)

# fig, ax = plt.subplots(1, 2, figsize=(7, 3))
# ax[0].plot(snr_const, 10*np.log10(sig_pow_opti_recal))
# ax[0].set_title('Signal Power [dB] (Optimal)')
# ax[1].plot(snr_const, 10*np.log10(int_pow_opti_recal))
# ax[1].set_title('Interference Power [dB] (Optimal)')


# %%
methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},      # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},      # Blue circles
    'beam_sweep': {'color': "#ff9941d2", 'marker': 'p'},     # Orange pentagons
    'beam_sweep_csi': {'color': '#ff7f0e', 'marker': 's'}  # Orange squares (CSI)
}

fig, ax = plt.subplots(1, 1, figsize=(4,3))

# w = v (dash lines)
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
       label='Beam Sweeping')

ax.plot(snr_const, 10*np.log10(sinr_sweeping_1b_csi.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='--', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping (Perfect SI CSI)')

# w ≠ v (dashed lines)
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_test_2b], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, [np.mean(p) for p in rieman_opti_sinr_2b], \
       marker=methods['iteropti']['marker'], linestyle='-', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='-', 
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b_csi.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='-', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping (Perfect SI CSI)')

ax.plot(snr_const, 10*np.log10(iterative_generalized_eig.squeeze()), \
       marker='^', linestyle=':', 
       color="#000000", linewidth=1.2, markersize=6,
       label='Iterative Gen. Eig.')

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


#####################################################################################
# --------------------------------------------------------------------------------- #
#####################################################################################
# %%
snr = 10  
tau = [4,6,8,10,12,14,16]

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
Pvec = 10**(snr/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2

for i, n_tau in enumerate(tau):
        filename = os.path.join('BD_beamInit', \
            'TEST_mono_N_%d_tau_%d_snr_%d.mat' % (N_ris, n_tau, int(snr)))
        data = scipy.io.loadmat(filename)
        sinr_test_1b.append(data['sinr_test'].squeeze())
        rieman_opti_sinr_1b.append(data['rieman_opti_sinr'].squeeze())

        filename2 = os.path.join('BD_2beamsInit', \
            'TEST_mono_N_%d_tau_%d_snr_%d.mat' % (N_ris, n_tau, int(snr)))
        data2 = scipy.io.loadmat(filename2)
        sinr_test_2b.append(data2['sinr_test'].squeeze())
        rieman_opti_sinr_2b.append(data2['rieman_opti_sinr'].squeeze())


        channel_true_val, loc_true = generate_irs_user_channel(
                                None, location_ris_1, num_samples=2000, Rician_factor=Rician_factor)   

        for j in range(len(loc_true)):
            loc = loc_true[j].squeeze()
            loc_cartesian = np.array([[loc[1]*np.cos(loc[0]), loc[1]*np.sin(loc[0]), -20]])

            H_SI = channel_true_val[0]
            H_b = channel_true_val[2][j]

            ### Optimal Beamforming from \cite{barneto_beamformer_2021} 
            # steer_vec = np.exp(1j * np.pi * np.arange(N_ris) * np.sin(loc[0]))

            # theta_star = np.conj(steer_vec[:,np.newaxis])/np.linalg.norm(steer_vec)
            # N = np.eye(N_ris) - ((H_SI @ theta_star) @ np.conj(H_SI @ theta_star).transpose())\
            #                               /np.linalg.norm(H_SI @ theta_star)**2
            # v_star = (N @ steer_vec) / np.linalg.norm(N @ steer_vec)
            # sig_pow_opti_recal.append(Pvec * np.abs(np.transpose(np.conj(v_star)) @ H_b @ theta_star)**2)
            # int_pow_opti_recal.append(Pvec * np.abs(np.transpose(np.conj(v_star)) @ H_SI @ theta_star)**2)
            # sinr_opti_recal.append(sig_pow_opti_recal[-1] / (int_pow_opti_recal[-1] + 1))

            ### Lower Bound Beam Sweeping
            CB = dft_codebook(N_ris, n_tau)
            H_SI_hat =   H_SI + (np.random.randn(*H_SI.shape) + 1j * np.random.randn(*H_SI.shape))/np.sqrt(2*Pvec)
            observation = np.sqrt(Pvec)*channel_true_val[1][j].squeeze() + 1/np.sqrt(2) *(np.random.randn(*channel_true_val[1][j].squeeze().shape) + 1j * np.random.randn(*channel_true_val[1][j].squeeze().shape))
            H_b_hat = observation - H_SI_hat*np.sqrt(Pvec)
            
       #      si = np.abs(np.conj(CB).T @ H_SI @ CB)**2
            bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
            metric = bi.diagonal() #/ (si.diagonal() + 1/Pvec)
            best_idx = np.argmax(metric)
            theta_test = CB[:, best_idx][:, np.newaxis]
            N_test = np.eye(N_ris) - ((H_SI_hat @ theta_test) @ np.conj(H_SI_hat @ theta_test).transpose())\
                                            /np.linalg.norm(H_SI_hat @ theta_test)**2
            v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
            sinr_sweeping_1b.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_SI @ theta_test)**2 + 1))
            sinr_sweeping_2b.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_SI @ theta_test)**2 + 1))
            
            ### Lower Bound Beam Sweeping
           
            H_b_hat = observation - H_SI*np.sqrt(Pvec)
            
            bi = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
            metric = bi.diagonal() #/ (si.diagonal() + 1/Pvec)
            best_idx = np.argmax(metric)
            theta_test = CB[:, best_idx][:, np.newaxis]
            N_test = np.eye(N_ris) - ((H_SI @ theta_test) @ np.conj(H_SI @ theta_test).transpose())\
                                            /np.linalg.norm(H_SI @ theta_test)**2
            v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
            sinr_sweeping_1b_csi.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_SI @ theta_test)**2 + 1))
            sinr_sweeping_2b_csi.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_SI @ theta_test)**2 + 1))


sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(tau), -1)), axis=1)
sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(tau), -1)), axis=1)
sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(tau), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(tau), -1)), axis=1)

# %%

fig, ax = plt.subplots(1, 1, figsize=(4,3))
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
ax.plot(tau, [np.mean(p) for p in rieman_opti_sinr_2b], \
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
ax.set_ylim([-30, 25])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/sinr_tau.pdf', format = 'pdf', bbox_inches = 'tight')
# %%
