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
from manifold_optimization import solve_with_random_restarts, solve_x_equals_y_fast

args = parse_args()

###############################################################
#            Params for scatters (Extension)
###############################################################
N_tx = 16
N_rx = 16
K = 1

N_scatter = 1
fc = args.fc
Wavelength = 3e8 / fc
ref_dis = Wavelength*166.67
location_tx = np.array([0, 0, 0])
location_rx = np.array([0, 0, 0])


def dft_codebook(n, m):
    dft_matrix = np.ones((n,m)) * complex(1, 0)
    for i in range(1, m+1):
        dft_matrix[:,i-1] = 1/np.sqrt(n) * np.array([np.exp(-1j * np.pi * j *(2 * i - 1 - m)/m) for j in range(n)])
    return dft_matrix

methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},         # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},         # Blue circles
    'beam_sweep': {'color': "#ff7f0e", 'marker': 's'},     # Orange pentagons  (analog Rx, est. SI)
    'beam_sweep_csi': {'color': "#d8b99f", 'marker': 'p'},   # Orange squares    (analog Rx, perfect SI)
    'sweep_dig': {'color': "#9467bd", 'marker': 'v'},      # Brown triangles   (digital Rx, est. SI)
    'iteropti_hat': {'color': '#2ca02c', 'marker': '*'}     # Brown stars       (digital Rx, perfect SI)
}


def beam_groups(n, m, balanced=True):
    g = n // m
    if balanced:
        return [list(a) for a in np.array_split(np.arange(n), m)]
    groups = [list(range(k*g, (k+1)*g)) for k in range(m)]
    groups[-1] += list(range(m*g, n))
    return groups


def grouped_dft_codebook(n, m, balanced=True, phase='equal'):
    """m broadened beams, each the sum of adjacent full-resolution DFT beams.
    """
    if m >= n:
        return dft_codebook(n, m)

    basis = dft_codebook(n, n)
    codebook = np.zeros((n, m), dtype=complex)
    for k, idx in enumerate(beam_groups(n, m, balanced)):
        g = len(idx)
        if phase == 'equal':
            weights = np.ones(g)
        else:
            weights = np.exp(1j * np.pi * np.arange(g)**2 / g)
        beam = basis[:, idx] @ weights
        codebook[:, k] = beam / np.linalg.norm(beam)
    return codebook

CODEBOOK = 'grouped'


def make_codebook(n, m):
    return grouped_dft_codebook(n, m) if CODEBOOK == 'grouped' else dft_codebook(n, m)

def compute_optimal_beamformers_2b(H_b_batch, H_int_batch, noise_var_val, P_val, num_restarts=10):
    """Compute optimal beamformers for a batch - reduced restarts for 2x speedup"""
    batch_size = H_b_batch.shape[0]
    v_opt_batch = np.zeros((batch_size, H_b_batch.shape[1], 1), dtype=np.complex64)
    w_opt_batch = np.zeros((batch_size, H_b_batch.shape[2], 1), dtype=np.complex64)
    
    for i in range(batch_size):
        H_b_i = H_b_batch[i]
        H_int_i = H_int_batch[i]
        A = np.sqrt(P_val) * H_b_i
        B = np.sqrt(P_val) * H_int_i
        try:
            # Reduced restarts from 10 to 5 for ~2x speedup with minimal quality loss
            w_opt_i, v_opt_i, _ = solve_with_random_restarts(A, B, c=noise_var_val, restarts=num_restarts)
            v_opt_batch[i, :, 0] = v_opt_i
            w_opt_batch[i, :, 0] = w_opt_i
        except:
            # Fallback to SVD
            U, S, Vh = np.linalg.svd(H_b_i)
            v_opt_batch[i, :, 0] = U[:, 0]
            w_opt_batch[i, :, 0] = Vh[0, :]
    
    return v_opt_batch.astype(np.complex64), w_opt_batch.astype(np.complex64)

def compute_optimal_beamformers_1b(H_b_batch, H_int_batch, noise_var_val, P_val, num_restarts=10):
    """Compute optimal beamformers for a batch - reduced restarts for 2x speedup"""
    batch_size = H_b_batch.shape[0]
    
    w_opt_batch = np.zeros((batch_size, H_b_batch.shape[2], 1), dtype=np.complex64)
    
    for i in range(batch_size):
        H_b_i = H_b_batch[i]
        H_int_i = H_int_batch[i]
        A = np.sqrt(P_val) * H_b_i
        B = np.sqrt(P_val) * H_int_i
        try:
            # Reduced restarts from 10 to 5 for ~2x speedup with minimal quality loss
            w_opt_i, _ = solve_x_equals_y_fast(A, B, c=noise_var_val, restarts=num_restarts)

            w_opt_batch[i, :, 0] = w_opt_i
        except:
            # Fallback to SVD
            U, S, Vh = np.linalg.svd(H_b_i)

            w_opt_batch[i, :, 0] = Vh[0, :]
    
    return w_opt_batch.astype(np.complex64)
def compute_beamformer_metrics_batch(H_d_batch, H_b_batch, H_r_batch, v_batch, w_batch, noise_var_val, P_val):
    """Evaluate BD and scatter SINR for a batch of beamformers."""
    H_interference = H_d_batch + H_r_batch

    v_h = np.conjugate(np.swapaxes(v_batch, 1, 2))
    sig_bd = np.matmul(v_h, np.matmul(H_b_batch, w_batch))
    sig_bd = np.squeeze(np.abs(sig_bd) ** 2, axis=(1, 2)) * P_val

    sig_int = np.matmul(v_h, np.matmul(H_interference, w_batch))
    sig_int = np.squeeze(np.abs(sig_int) ** 2, axis=(1, 2)) * P_val
    sinr_bd = sig_bd / (sig_int + noise_var_val)

    sig_scatter = np.matmul(v_h, np.matmul(H_r_batch, w_batch))
    sig_scatter = np.squeeze(np.abs(sig_scatter) ** 2, axis=(1, 2)) * P_val

    int_scatter = np.matmul(v_h, np.matmul(H_d_batch + H_b_batch, w_batch))
    int_scatter = np.squeeze(np.abs(int_scatter) ** 2, axis=(1, 2)) * P_val
    sinr_scatter = sig_scatter / (int_scatter + noise_var_val)

    return (
        sinr_bd.astype(np.float32),
        sinr_scatter.astype(np.float32),
        sig_bd.astype(np.float32),
        sig_int.astype(np.float32),
    )

methods = {
    'proposed': {'color': '#d62728', 'marker': 'd'},         # Red diamonds
    'iteropti': {'color': '#2077b4', 'marker': 'o'},         # Blue circles
    'beam_sweep': {'color': "#ff7f0e", 'marker': 's'},     # Orange pentagons  (analog Rx, est. SI)
    'beam_sweep_csi': {'color': "#ff7f0e", 'marker': 'p'},   # Orange squares    (analog Rx, perfect SI)
    'sweep_dig': {'color': "#9467bd", 'marker': 'v'},      # Brown triangles   (digital Rx, est. SI)
    'iteropti_hat': {'color': '#2ca02c', 'marker': '*'}     # Brown stars       (digital Rx, perfect SI)
}

# %%
sig_pow_opti_recal = []
int_pow_opti_recal = []
sinr_opti_recal = []
rieman_opti_sinr_1b = []
sinr_test_1b = []
sinr_opt_hat_1b = []
sinr_opt_hat_2b = []
sinr_sweep_1b_anaw = []

sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweep_digiw = []
sinr_sweep_2b_anaw = []
iterative_generalized_eig = []
tau = 16
K = 1
N_scatter = 5
CB = dft_codebook(N_tx, tau)
snr_const = [ -10,-5, 0, 5, 10, 15, 20, 25]


for i, snr in enumerate(snr_const):

       ###############################################################
       #            files with scatters (Extension)
       ###############################################################
       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm', \
              'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, tau, snr, K, N_scatter))
       data = scipy.io.loadmat(filename)

       sinr_test_2b.append(data['sinr_learned'].squeeze())

       rieman_opti_sinr_2b.append(data['sinr_optimal'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_1b', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, tau, snr, K, N_scatter))
       data1b = scipy.io.loadmat(filename)
       sinr_test_1b.append(data1b['sinr_learned'].squeeze())
       rieman_opti_sinr_1b.append(data1b['sinr_optimal'].squeeze())

       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

       H_I_hat_batch = []
       H_b_hat_batch = []
       H_b_batch = []
       H_r_batch = []
       H_d_batch = []

       Pvec = 10**(snr/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_tx)**2

       for j in range(test_size):
              bd_loc = generate_location_mimo(1, 'u')[0]
              scatter_loc = generate_location_mimo(N_scatter, 's')
              _, H_d_test, H_r_test, H_b_test = generate_mimo_channel(
                                   location_tx, location_rx, scatter_loc, bd_loc, N_tx, 1, N_rx, 1)
              H_b_batch.append(H_b_test)
              H_r_batch.append(H_r_test)
              H_d_batch.append(H_d_test)

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

              H_b_hat_batch.append(H_b_hat)
              H_I_hat_batch.append(H_I_hat)

              ##### Beam sweeping both analog Rx and Tx
              numerator = np.abs(CB.conj().T @ H_b_hat @ CB)**2
              denominator = np.abs(CB.conj().T @ H_I_hat @ CB)**2 + 1/Pvec
              # numerator = np.linalg.norm(H_b_hat @ CB, axis=0)**2
              # denominator = np.linalg.norm(H_I_hat @ CB, axis=0)**2 + 1/Pvec

              metric_1b = np.diag(numerator) / (np.diag(denominator) )
              best_idx = np.argmax(metric_1b)
              theta_sp = CB[:, best_idx]
              sinr_sweep_1b_anaw.append(Pvec * np.abs(np.transpose(np.conj(theta_sp)) @ H_b @ theta_sp)**2 / \
                                                 (Pvec * np.abs(np.transpose(np.conj(theta_sp)) @ H_I @ theta_sp)**2 + 1))

              theta_test = theta_sp
              # N_test = np.eye(N_tx) - ((H_I_hat @ theta_test) @ np.conj(H_I_hat @ theta_test).transpose())\
              #                                                  /np.linalg.norm(H_I_hat @ theta_test)**2
              # v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
              v_test = np.linalg.solve(Pvec*(H_I_hat @ theta_test) @ np.conj(H_I_hat @ theta_test).T + np.eye(N_tx), Pvec*H_b_hat @ theta_test)
              v_test = v_test / np.linalg.norm(v_test)
              
              sinr_sweep_digiw.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))

              metric_2b = numerator / denominator
              best_idx, best_jdx = np.unravel_index(np.argmax(metric_2b), metric_2b.shape)
              best_theta = CB[:, best_jdx]
              best_v = CB[:, best_idx]       
              sinr_sweep_2b_anaw.append(Pvec * np.abs(np.transpose(np.conj(best_v)) @ H_b @ best_theta)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(best_v)) @ H_I @ best_theta)**2 + 1))

       ##### Optimal beams from H_hat
       v_sp, w_sp = compute_optimal_beamformers_2b(
                     np.array(H_b_hat_batch), np.array(H_I_hat_batch),1, Pvec, num_restarts=1)

       sinr_opt_2b, _, _, _ = compute_beamformer_metrics_batch(
              np.array(H_d_batch),np.array(H_b_batch),np.array(H_r_batch),
              np.array(v_sp),np.array(w_sp),1,Pvec,
       )
       sinr_opt_hat_2b.append(sinr_opt_2b)

       vw_opt = compute_optimal_beamformers_1b(
              np.array(H_b_hat_batch),np.array(H_I_hat_batch),
              1,Pvec,num_restarts=1,
       )
       sinr_opt_1b, _, _, _ = compute_beamformer_metrics_batch(
                     np.array(H_d_batch),np.array(H_b_batch),np.array(H_r_batch),
                     np.array(vw_opt),np.array(vw_opt),1,Pvec,
       )
       sinr_opt_hat_1b.append(sinr_opt_1b)



sinr_sweep_digiw = np.mean(np.reshape(sinr_sweep_digiw, (len(snr_const), -1)), axis=1)
sinr_sweep_2b_anaw = np.mean(np.reshape(sinr_sweep_2b_anaw, (len(snr_const), -1)), axis=1)
sinr_sweep_1b_anaw = np.mean(np.reshape(sinr_sweep_1b_anaw, (len(snr_const), -1)), axis=1)

sinr_opt_hat_1b = 0 if len(sinr_opt_hat_1b) == 0 else np.mean(np.reshape(sinr_opt_hat_1b, (len(snr_const), -1)), axis=1)
sinr_opt_hat_2b = 0 if len(sinr_opt_hat_2b) == 0 else np.mean(np.reshape(sinr_opt_hat_2b, (len(snr_const), -1)), axis=1)

# %%

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

ax.plot(snr_const, 10*np.log10(sinr_opt_hat_1b.squeeze()), \
       marker=methods['iteropti_hat']['marker'], linestyle='--',
       color=methods['iteropti_hat']['color'], linewidth=1.2, markersize=6,
       label='IterOpti_hat')

ax.plot(snr_const, 10*np.log10(sinr_sweep_1b_anaw.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='--',
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping (analog Rx)')

## w ≠ v (dashed lines)
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_test_2b], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in rieman_opti_sinr_2b], \
       marker=methods['iteropti']['marker'], linestyle='-', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, 10*np.log10(sinr_opt_hat_2b.squeeze()), \
       marker=methods['iteropti_hat']['marker'], linestyle='-',
       color=methods['iteropti_hat']['color'], linewidth=1.2, markersize=6,
       label='IterOpti_hat')

ax.plot(snr_const, 10*np.log10(sinr_sweep_digiw.squeeze()), \
       marker=methods['sweep_dig']['marker'], linestyle='-', 
       color=methods['sweep_dig']['color'], linewidth=1.2, markersize=7)

ax.plot(snr_const, 10*np.log10(sinr_sweep_2b_anaw.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='-', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6)


# Create custom legend
from matplotlib.lines import Line2D

# Method legend (colors/markers)
method_legend = [
    Line2D([0], [0], color=methods['proposed']['color'],
        marker=methods['proposed']['marker'], linestyle='None', 
        markersize=7, label='Proposed'),
    Line2D([0], [0], color=methods['iteropti']['color'], 
           marker=methods['iteropti']['marker'], linestyle='None', 
           markersize=7, label='IterOpti'),
    Line2D([0], [0], color=methods['iteropti_hat']['color'],
            marker=methods['iteropti_hat']['marker'], linestyle='None',
            markersize=7, label=r'IterOpti $\hat{\mathbf{H}}_{\rm SI}, \hat{\mathbf{H}}_{\rm b}$'),
    Line2D([0], [0], color=methods['beam_sweep']['color'],
           marker=methods['beam_sweep']['marker'], linestyle='None',
           markersize=7, label=r'Sweep (Analog $\mathbf{w}$)'),
           
    Line2D([0], [0], color=methods['sweep_dig']['color'],
           marker=methods['sweep_dig']['marker'], linestyle='None',
           markersize=7, label=r'Sweep (Digital $\mathbf{w}$)'),
#     Line2D([0], [0], color=methods['sweep_dig_csi']['color'],
#            marker=methods['sweep_dig_csi']['marker'], linestyle='None',
#            markersize=7, label=r'Sweep, Digital $\mathbf{v}$ ($\mathbf{H}_{\rm SI}$)'),
]

# Line style legend
style_legend = [
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1.2, 
           label=r'$\mathbf{w} = \mathbf{v}$'),
    Line2D([0], [0], color='grey', linestyle='-', linewidth=1.2, 
           label=r'$\mathbf{w} \neq \mathbf{v}$')
]

# Create two separate legends
legend1 = ax.legend(handles=method_legend, loc='lower right', ncols=1,
                   frameon=True, fontsize=8, fancybox=True, framealpha=0.5)
legend2 = ax.legend(handles=style_legend, loc='upper left',
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.5)

# legend1.get_texts()[2].set_position((0, 2))
from matplotlib.transforms import ScaledTranslation
offset = ScaledTranslation(0, -3 / 72, fig.dpi_scale_trans)  
for i in range(0, 2):
       legend1.get_texts()[i].set_transform(legend1.get_texts()[i].get_transform() + offset)
       legend1.legend_handles[i].set_transform(
       legend1.legend_handles[i].get_transform() + offset
       )

# Add the first legend back (matplotlib removes it when creating the second)
ax.add_artist(legend1)

ax.set_xlabel('Effective SNR [dB]')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xticks(snr_const)
ax.set_ylim([-40, 26])
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
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=7,
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
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=7)
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
           markersize=7, label='Sweep ($\hat{\mathbf{H}}_{{SI}}$)'),
    Line2D([0], [0], color=methods['beam_sweep_csi']['color'], 
           marker=methods['beam_sweep_csi']['marker'], linestyle='None', 
           markersize=7, label='Sweep ($\mathbf{H}_{{SI}}$)'),
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
# ax.set_ylim([-12, 10])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/scatter_sinr_tau.pdf', format = 'pdf', bbox_inches = 'tight')

# %%
#####################################################################################
# ----------------------------- Multiple Scatters --------------------------------- #
#####################################################################################

snr_const = 10 
# tau = [8,12,16,20,24]  
tau = [4,6,8,10,12,14,16]
K = 1

sig_pow_opti_recal = []
int_pow_opti_recal = []
sinr_opti_recal = []
num_scatters = 5
N_tx = 16
N_rx = 16

sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweep_digiw = []
sinr_sweep_2b_anaw = []

sinr_test_1b = []
rieman_opti_sinr_1b = []
sinr_sweeping_1b = []
sinr_sweep_1b_anaw = []
Pvec = 10**(snr_const/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_tx)**2

for i, n_tau in enumerate(tau):
       filename = os.path.join('Mo_mimo_sinr_modelsave', \
       'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (N_tx, N_rx, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)
       # sinr_test_1b.append(data['sinr_test'].squeeze())
       # rieman_opti_sinr_1b.append(data['rieman_opti_sinr'].squeeze())
       sinr_test_1b.append(data['sinr_learned'].squeeze())
       rieman_opti_sinr_1b.append(data['sinr_optimal'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm', \
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (16, 16, n_tau, snr_const, K, num_scatters))
       data = scipy.io.loadmat(filename)

       sinr_test_2b.append(data['sinr_learned'].squeeze())

       rieman_opti_sinr_2b.append(data['sinr_optimal'].squeeze())

       

       BD_loc = data['BD_location'].squeeze()
       Scatter_loc = data['Scatter_location'].squeeze()
       test_size = BD_loc.shape[0]

       CB = make_codebook(N_tx, n_tau)

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
              
              #  digital Rx 
              numerator = np.abs(CB.conj().T @ H_b_hat @ CB)**2
              denominator = np.abs(CB.conj().T @ H_I_hat @ CB)**2 + 1/Pvec
              metric_1b = np.diagonal(numerator) / np.diagonal(denominator)
              best_idx = np.argmax(metric_1b)
              theta_test = CB[:, best_idx][:, np.newaxis]
              # N_test = np.eye(N_tx) - ((H_I_hat @ theta_test) @ np.conj(H_I_hat @ theta_test).transpose())\
              #                                    /np.linalg.norm(H_I_hat @ theta_test)**2
              # v_test = (N_test @ theta_test) / np.linalg.norm(N_test @ theta_test)
              v_test = np.linalg.solve(Pvec*(H_I_hat @ theta_test) @ np.conj(H_I_hat @ theta_test).T + np.eye(N_tx), Pvec*H_b_hat @ theta_test)
              v_test = v_test / np.linalg.norm(v_test)
              
              sinr_sweep_digiw.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))
              sinr_sweep_1b_anaw.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))
              
              ### Analog Tx and Rx       
              
              metric_2b = numerator / (denominator)
              best_idx, best_jdx = np.unravel_index(np.argmax(metric_2b), metric_2b.shape)
              theta_test = CB[:, best_idx][:, np.newaxis]
              v_test = CB[:, best_jdx][:, np.newaxis]

              sinr_sweep_2b_anaw.append(Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_b @ theta_test)**2 / \
                                   (Pvec * np.abs(np.transpose(np.conj(v_test)) @ H_I @ theta_test)**2 + 1))
              # sinr_sweep_1b_anaw.append(Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_b @ theta_test)**2 / \
              #                      (Pvec * np.abs(np.transpose(np.conj(theta_test)) @ H_I @ theta_test)**2 + 1))


sinr_sweep_digiw = np.mean(np.reshape(sinr_sweep_digiw, (len(tau), -1)), axis=1)
sinr_sweep_2b_anaw = np.mean(np.reshape(sinr_sweep_2b_anaw, (len(tau), -1)), axis=1)
# sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(tau), -1)), axis=1)
sinr_sweep_1b_anaw = np.mean(np.reshape(sinr_sweep_1b_anaw, (len(tau), -1)), axis=1)

# %%

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
# ax.plot(tau, 10*np.log10(sinr_sweeping_1b.squeeze()), \
#        marker=methods['beam_sweep']['marker'], linestyle='--', 
#        color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6,
#        label='Beam Sweeping')
ax.plot(tau, 10*np.log10(sinr_sweep_1b_anaw.squeeze()), \
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
ax.plot(tau, 10*np.log10(sinr_sweep_digiw.squeeze()), \
       marker=methods['sweep_dig']['marker'], linestyle='-', 
       color=methods['sweep_dig']['color'], linewidth=1.2, markersize=7)
ax.plot(tau, 10*np.log10(sinr_sweep_2b_anaw.squeeze()), \
       marker=methods['beam_sweep_csi']['marker'], linestyle='-', 
       color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
       label='Beam Sweeping')
# Create custom legend
from matplotlib.lines import Line2D
# Method legend (colors/markers)
method_legend = [
    Line2D([0], [0], color=methods['sweep_dig']['color'], 
           marker=methods['sweep_dig']['marker'], linestyle='None', 
           markersize=7, label='Sweep (Digital $\mathbf{w})$'),
    Line2D([0], [0], color=methods['beam_sweep_csi']['color'], 
           marker=methods['beam_sweep_csi']['marker'], linestyle='None', 
           markersize=7, label='Sweep (Analog $\mathbf{w}$)'),
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
ax.set_ylim([-20, 10])
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
######################################################################################
#
#------------------  Multi-BD: non-learning (benchmark) methods  ------------------- #
#
# Loads the results saved by mono_1lstm_mBD.py and re-runs the classical pipeline on
# exactly the same test geometries and the same awake-BD labels.
#
# Signal model (identical to mono_1lstm_mBD.py). The Hadamard code lives in TIME,
# not in space. Sounding step t uses ONE transmit vector w_t, taken from the DFT
# codebook and held constant over the whole length-L preamble; the awake BD keys its
# own Hadamard column chip by chip while the other BDs stay silent:
#
#     y[t, p] = sqrt(P) * ( H_I + c[p, u*] * H_b[u*] ) @ w_t + n[t, p],
#     p = 0 ... L-1 chips,  each chip repeated K symbols.
#
# Pipeline:
#   tau non-adaptive DFT sounding beams (this IS the beam sweep)
#     -> despread in time into one branch per BD (+ one common branch)
#     -> detector decides which BD is awake
#     -> ridge-LS estimate of that BD's channel and of the static interference
#     -> beamformer design (alternating generalized eigenvector, or codebook sweep)
#     -> SINR evaluated on the TRUE channels, for the TRUE awake BD.
#
# Sensing cost is identical to the learned scheme: tau steps x L chips x K symbols.
# The only difference is that w_t is a fixed codebook column here, whereas the
# learned scheme steers w_t adaptively from the recurrent state.
######################################################################################
from iter_gen_eig import alternating_xy

RESULT_DIR   = 'Mo_1LSTM_3BD_results'
tau          = 16
K            = 1
N_scatter    = 5
N_bd         = 3
snr_const    = [-10, -5, 0, 5, 10, 15]

noise_var    = 1.0      # matches mono_1lstm_mBD.py (2 * noiseSTD_per_dim**2)
LS_REG       = 1e-1     # ridge on W W^H, as in the sp_beamformer scope
N_RESTART    = 3        # random restarts of the alternating solver
MAX_ITER     = 60
BENCH_SEED   = 2024
n_bench      = None     # None -> every test sample; set an int to subsample
Wavelength   = 3e8 / args.fc
ref_dis      = 166.67*Wavelength
N_tx = 36
N_rx = N_tx
#############################  benchmark building blocks  ############################

def sense_and_despread(H_I, H_b, u_star, W_sound, L, P, K, noise_var, rng):
    """Run the tau-step sounding phase and return the time-despread observations.

    Step t transmits the single beam ``W_sound[:, t]``, held constant over the
    L chips of the preamble and over the K repeated symbols per chip:

        y[t, p] = sqrt(P) * (H_I + c[p, u*] * H_b[u*]) @ w_t + n[t, p]

    The Hadamard code is therefore a TIME spreading sequence, not a second spatial
    sounding vector. Despreading in time with column u (c_u^T c_u' = L delta, and
    c_u^T 1 = 0 for the BD columns) gives

        Ybar_bd[u][:, t] = sqrt(P) * 1{u == u*} * H_b[u] @ w_t + n
        Ybar_c[:, t]     = sqrt(P) * H_I @ w_t + n,   n ~ CN(0, noise_var/(L*K))

    with noise independent across branches, because despreading is an orthogonal
    transform. Generating these directly is exactly equivalent to simulating all
    L chips, and L times cheaper.
    """
    N_rx = H_I.shape[0]
    n_bd = H_b.shape[0]
    n_steps = W_sound.shape[1]
    scale = np.sqrt(noise_var / (L * K) / 2.0)

    Ybar_bd = scale * (rng.standard_normal((n_bd, N_rx, n_steps))
                       + 1j * rng.standard_normal((n_bd, N_rx, n_steps)))
    Ybar_bd[u_star] += np.sqrt(P) * (H_b[u_star] @ W_sound)

    Ybar_c = scale * (rng.standard_normal((N_rx, n_steps))
                      + 1j * rng.standard_normal((N_rx, n_steps)))
    Ybar_c += np.sqrt(P) * (H_I @ W_sound)
    return Ybar_bd, Ybar_c


def ls_operator(W_sound, reg):
    """Right-multiplier M with H_hat = Ybar @ M / sqrt(P) (ridge LS)."""
    G = W_sound @ W_sound.conj().T + reg * np.eye(W_sound.shape[0])
    return W_sound.conj().T @ np.linalg.inv(G)


def design_alt_opt(H_b_hat, H_I_hat, P, restarts, max_iter):
    """Alternating generalized-eigenvector design on the ESTIMATED channels.

    Maximises P|v^H H_b w|^2 / (P|v^H H_I w|^2 + noise_var); the restart is picked
    by the objective the receiver can actually see, i.e. the estimated one.
    """
    best_v, best_w, best_obj = None, None, -np.inf
    A = np.sqrt(P) * H_b_hat
    B = np.sqrt(P) * H_I_hat
    for _ in range(restarts):
        v, w, obj = alternating_xy(A, B, c=noise_var, max_iter=max_iter)
        if obj > best_obj:
            best_v, best_w, best_obj = v, w, obj
    return best_v, best_w


def design_sweep_analog(H_b_hat, H_I_hat, CB, P):
    """Exhaustive analog Tx x Rx codebook sweep on the estimated channels."""
    num = np.abs(CB.conj().T @ H_b_hat @ CB) ** 2
    den = np.abs(CB.conj().T @ H_I_hat @ CB) ** 2 + noise_var / P
    i, j = np.unravel_index(np.argmax(num / den), num.shape)
    return CB[:, i], CB[:, j]


def design_sweep_digital(H_b_hat, H_I_hat, CB, P):
    """Analog Tx codebook sweep + LMMSE digital Rx combining.

    For a candidate Tx beam w the best digital receiver attains
    SINR(w) = P a^H (P b b^H + noise_var I)^{-1} a with a = H_b_hat w,
    b = H_I_hat w, which Sherman-Morrison reduces to a closed form. The Tx beam is
    picked by that score and the matching v is the LMMSE combiner.
    """
    A = H_b_hat @ CB                                  # (N_rx, n_beams)
    B = H_I_hat @ CB
    bHa = np.sum(B.conj() * A, axis=0)
    b_n2 = np.sum(np.abs(B) ** 2, axis=0)
    a_n2 = np.sum(np.abs(A) ** 2, axis=0)
    coef = P / (noise_var + P * b_n2)
    score = (P / noise_var) * (a_n2 - np.abs(bHa) ** 2 * coef)
    k = int(np.argmax(score))
    v = A[:, k] - B[:, k] * (bHa[k] * coef[k])
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, CB[:, k]


def true_sinr(v, w, H_b_active, H_I, P):
    """SINR actually delivered to the awake BD, on the true channels."""
    v = v.reshape(-1)
    w = w.reshape(-1)
    sig = P * np.abs(np.vdot(v, H_b_active @ w)) ** 2
    inter = P * np.abs(np.vdot(v, H_I @ w)) ** 2 + noise_var
    return sig / inter


#################################  benchmark sweep  ##################################

sinr_test        = []   # learned, active BD
sinr_opti_sinr   = []   # genie-CSI upper bound (saved by mono_1lstm_mBD.py)
identify_accuracy = []  # learned classifier
sinr_test_2b     = []   # single-BD learned reference

bench = {k: [] for k in ('alt_opt', 'alt_opt_genie', 'sweep_analog',
                         'sweep_digital', 'no_id')}
id_acc_energy = []   # energy detector on the despread branches (drives the designs)
id_acc_ls     = []   # LS detector on the beam-deconvolved channel estimates
sinr_when_id_ok   = []
sinr_when_id_bad  = []

for i, snr in enumerate(snr_const):
       filename = os.path.join(RESULT_DIR,
            'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat'
              % (N_tx, N_rx, tau, snr, K, N_scatter))
       data = scipy.io.loadmat(filename)
       sinr_test.append(data['sinr_learned'].squeeze())
       sinr_opti_sinr.append(data['sinr_optimal'].squeeze())
       identify_accuracy.append(data['identification_accuracy'].squeeze())

       filename = os.path.join('Mo_mimo_sinr_modelsave_one_lstm', \
                     'TEST_sinr_N_%d_%d_tau_%d_snr_%d_K_%d_Nsca_%d.mat' % (36, 36, tau, snr, K, N_scatter))
       data1b = scipy.io.loadmat(filename)
       sinr_test_2b.append(data1b['sinr_learned'].squeeze())

       # ---- everything the benchmark needs comes from the saved test file ----
       BD_loc      = data['BD_location']                       # (test, N_bd, 3)
       Scatter_loc = data['Scatter_location']                  # (test, N_sca, 3)
       active_user = data['active_user'].squeeze().astype(int)  # true awake BD
       L           = int(data['BD_preamble'].shape[0])          # preamble length
       K_file      = int(np.squeeze(data['K']))
       n_bd_file   = int(np.squeeze(data['num_users']))
       test_size   = BD_loc.shape[0]

       P = 10 ** (snr / 10) / (Wavelength ** 4 / (4 * np.pi * ref_dis) ** 4) / (N_tx * N_rx)

       # One non-adaptive sounding: tau DFT beams, one transmit vector per step.
       # This single sweep serves every baseline -- it is both the beam sweep and
       # the pilot phase for the LS estimate.
       CB   = dft_codebook(N_tx, tau)
       M_ls = ls_operator(CB, LS_REG)

       rng = np.random.default_rng(BENCH_SEED + i)
       np.random.seed(BENCH_SEED + i)   # alternating_xy uses the legacy global RNG

       idx_list = list(range(test_size)) if n_bench is None \
              else list(range(min(n_bench, test_size)))
       acc_e = acc_l = 0
       run = {k: [] for k in bench}
       ok_list, bad_list = [], []

       for j in idx_list:
              bd_loc = generate_location_mimo(N_bd, 'u')
              scatter_loc = generate_location_mimo(N_scatter, 's')
              _, H_d, H_r, H_b = generate_mimo_channel(
                     location_tx, location_rx, scatter_loc, bd_loc,
                     N_tx, 1, N_rx, 1)
              H_I = H_d + H_r
              u_star = int(active_user[j])
              H_b_true = H_b[u_star]      # scoring always uses the TRUE awake BD

              # ---------------- sensing + time despreading ----------------
              Ybar_bd, Ybar_c = sense_and_despread(
                     H_I, H_b, u_star, CB, L, P, K_file, noise_var, rng)

              # ---------------- ridge-LS channel estimates ----------------
              Hb_hat = np.stack([Ybar_bd[u] @ M_ls for u in range(n_bd_file)]) / np.sqrt(P)
              HI_hat = (Ybar_c @ M_ls) / np.sqrt(P)

              # ---------------- identification ----------------
              # (a) Non-coherent energy detector 
              T_energy = np.sum(np.abs(Ybar_bd) ** 2, axis=(1, 2))
              # (b) LS detector
              T_ls = np.sum(np.abs(Hb_hat) ** 2, axis=(1, 2))
              u_hat_e = int(np.argmax(T_energy))
              u_hat_l = int(np.argmax(T_ls))
              acc_e += (u_hat_e == u_star)
              acc_l += (u_hat_l == u_star)

              u_hat = u_hat_e            # detector driving the beamformer designs
              Hb_sel = Hb_hat[u_hat]     # channel of the BD the benchmark believes

              # ------------- estimate-then-optimize (digital BF) -------------
              v, w = design_alt_opt(Hb_sel, HI_hat, P, N_RESTART, MAX_ITER)
              s_alt = true_sinr(v, w, H_b_true, H_I, P)
              run['alt_opt'].append(s_alt)
              (ok_list if u_hat == u_star else bad_list).append(s_alt)

              # genie identification: separates estimation error from ID error
              v, w = design_alt_opt(Hb_hat[u_star], HI_hat, P, N_RESTART, MAX_ITER)
              run['alt_opt_genie'].append(true_sinr(v, w, H_b_true, H_I, P))

              # no identification: serve the coherent sum of all BD estimates,
              # i.e. what a scheme that never resolves the awake BD would do
              v, w = design_alt_opt(Hb_hat.sum(axis=0), HI_hat, P, N_RESTART, MAX_ITER)
              run['no_id'].append(true_sinr(v, w, H_b_true, H_I, P))

              # ------------- beam sweeping over the swept codebook -------------
              v, w = design_sweep_analog(Hb_sel, HI_hat, CB, P)
              run['sweep_analog'].append(true_sinr(v, w, H_b_true, H_I, P))

              v, w = design_sweep_digital(Hb_sel, HI_hat, CB, P)
              run['sweep_digital'].append(true_sinr(v, w, H_b_true, H_I, P))

       n_run = len(idx_list)
       for k in bench:
              bench[k].append(np.asarray(run[k]))
       id_acc_energy.append(acc_e / n_run)
       id_acc_ls.append(acc_l / n_run)
       sinr_when_id_ok.append(np.asarray(ok_list))
       sinr_when_id_bad.append(np.asarray(bad_list))

       print('SNR %+3d dB | ID acc  learned %.3f  energy %.3f  LS %.3f | '
             'SINR [dB]  learned %6.2f  genie-CSI %6.2f  est+opt %6.2f  '
             '(genie-ID %6.2f)  sweep-dig %6.2f  sweep-ana %6.2f  no-ID %6.2f'
             % (snr, float(identify_accuracy[i]), id_acc_energy[i], id_acc_ls[i],
                10 * np.log10(np.mean(sinr_test[i])),
                10 * np.log10(np.mean(sinr_opti_sinr[i])),
                10 * np.log10(np.mean(bench['alt_opt'][i])),
                10 * np.log10(np.mean(bench['alt_opt_genie'][i])),
                10 * np.log10(np.mean(bench['sweep_digital'][i])),
                10 * np.log10(np.mean(bench['sweep_analog'][i])),
                10 * np.log10(np.mean(bench['no_id'][i]))))

print('\nSINR delivered to the awake BD when the energy detector is right / wrong [dB]:')
for i, snr in enumerate(snr_const):
       ok = sinr_when_id_ok[i]
       bad = sinr_when_id_bad[i]
       print('  SNR %+3d dB : %6.2f (n=%d)  /  %6.2f (n=%d)'
             % (snr,
                10 * np.log10(np.mean(ok)) if ok.size else float('nan'), ok.size,
                10 * np.log10(np.mean(bad)) if bad.size else float('nan'), bad.size))


# Cache the benchmark so the sweep does not have to be re-run for replotting.
bench_file = os.path.join(RESULT_DIR,
       'BENCH_nonlearning_N_%d_%d_tau_%d_K_%d_Nsca_%d_Nbd_%d.mat'
       % (N_tx, N_rx, tau, K, N_scatter, N_bd))
scipy.io.savemat(bench_file, dict(
       snr_const=np.asarray(snr_const),
       tau=tau, K=K, N_tx=N_tx, N_rx=N_rx, N_bd=N_bd, N_scatter=N_scatter,
       n_bench=-1 if n_bench is None else n_bench,
       sinr_learned=np.asarray([np.mean(p) for p in sinr_test]),
       sinr_optimal=np.asarray([np.mean(p) for p in sinr_opti_sinr]),
       id_acc_learned=np.asarray([float(p) for p in identify_accuracy]),
       id_acc_energy=np.asarray(id_acc_energy),
       id_acc_ls=np.asarray(id_acc_ls),
       **{('sinr_' + k): np.stack(v) for k, v in bench.items()}
))
print('\nBenchmark cached to %s' % bench_file)


#####################################  plots  ########################################
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_opti_sinr],
       marker='o', linestyle='-', color='#2077b4', linewidth=1.2, markersize=6,
       label='IterOpti (Genie ID)')
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_test_2b],
       marker='d', linestyle='-', color='#8c564b', linewidth=1.0, markersize=7,
       label='Proposed (Genie ID)')
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_test],
       marker='d', linestyle='-', color='#d62728', linewidth=1.2, markersize=6,
       label='Proposed (3 BDs)')
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in bench['alt_opt']],
       marker='^', linestyle='-', color='#2ca02c', linewidth=1.2, markersize=6,
       label=r'IterOpti $\hat{\mathbf{H}}_{\rm I}, \hat{\mathbf{H}}_{\rm b}$')
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in bench['sweep_digital']],
       marker='s', linestyle='-', color='#ff7f0e', linewidth=1.2, markersize=6,
       label=r'Sweep (Digital $\mathbf{w}$)')
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in bench['sweep_analog']],
       marker='v', linestyle='-', color='#9467bd', linewidth=1.2, markersize=6,
       label=r'Sweep (Analog $\mathbf{w}$)')
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in bench['no_id']],
       marker='x', linestyle=':', color='#7f7f7f', linewidth=1.2, markersize=6,
       label='No identification')


ax.set_xlabel('Effective SNR [dB]')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xticks(snr_const)
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
legend = ax.legend(loc='lower right', fontsize=8, frameon=True, fancybox=True, framealpha=0.6)

from matplotlib.transforms import ScaledTranslation
offset = ScaledTranslation(0, -2 / 72, fig.dpi_scale_trans)
for j in range(4):
    entry_offset = ScaledTranslation(
        0,
        (-2 - 2 * int(j != 3)) / 72,
        fig.dpi_scale_trans,
    )

    legend.get_texts()[j].set_transform(
        legend.get_texts()[j].get_transform() + entry_offset
    )
    legend.legend_handles[j].set_transform(
        legend.legend_handles[j].get_transform() + entry_offset
    )

plt.tight_layout()
# fig.savefig('figs/multiBD_sinr_snr.pdf', format = 'pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(snr_const, [np.mean(p) for p in identify_accuracy],
       marker='d', linestyle='-', color='#d62728', linewidth=1.2, markersize=6,
       label='Proposed')
ax.plot(snr_const, id_acc_energy,
       marker='^', linestyle='-', color='#2ca02c', linewidth=1.2, markersize=6,
       label='Energy detector')
ax.plot(snr_const, id_acc_ls,
       marker='s', linestyle='-', color='#ff7f0e', linewidth=1.2, markersize=6,
       label='LS detector')
ax.axhline(1.0 / N_bd, color='#7f7f7f', linestyle=':', linewidth=1.0,
       label='Random guess')
ax.set_xlabel('Effective SNR [dB]')
ax.set_ylabel('Identification Accuracy')
ax.set_xticks(snr_const)
ax.set_ylim([0, 1.05])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
ax.legend(loc='lower right', fontsize=8, frameon=True, fancybox=True, framealpha=0.6)
plt.tight_layout()
# fig.savefig('figs/multiBD_id_acc_snr.pdf', format = 'pdf', bbox_inches = 'tight')
# %%
