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

# args = parse_args()

N_ris = 16
tau = 10  
snr_const = [-10,-5, 0, 5, 10, 15]

Wavelength = 3e8/(10e9)
ref_dis = Wavelength*166.67

location_ris_1 = np.array([0, 0, -20])  


def dft_codebook(n, m):
    """
    Creates a DFT codebook for a ULA BS.

    Args:
        n (int): The number of antenna elements in the ULA.
        m (int): The number of beams in the codebook.

    Returns:
        codebook (numpy.ndarray): A numpy array of shape (n, m) representing the codebook.
    """
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
    """Split the n full-resolution DFT beams into m contiguous groups.

    balanced=False: m groups of floor(n/m), with the n mod m leftovers all appended
                    to the LAST group  ->  n=16, m=10 gives [1]*9 + [7]
    balanced=True : the leftovers are spread over the first groups instead
                    ->  n=16, m=10 gives [2]*6 + [1]*4
    """
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


def pair_sweep_size(tau):
    """Beams per side of an analog Tx x analog Rx sweep that must fit in tau slots.

    An analog receiver sees one scalar per slot, so every (v_i, w_j) pair costs a
    slot of its own: m beams per side => m^2 slots. Under a budget of tau slots
    the two codebooks are therefore only sqrt(tau) wide, unlike designs (a) and
    (b) below, which spend one slot per transmit beam and can afford all tau.
    """
    return max(1, int(np.floor(np.sqrt(tau))))


def sweep_designs(H_b_hat, H_SI_hat, CB, Pvec, CB_pair=None, digital_tx='diag'):
    """Codebook beam sweeping on the ESTIMATED channels (as in retest_scatters.py).
    """
    N = CB.shape[0]
    if CB_pair is None:
        CB_pair = make_codebook(N, pair_sweep_size(CB.shape[1]))
    numerator = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
    denominator = np.abs(np.conj(CB).T @ H_SI_hat @ CB)**2 + 1/Pvec

    # (a) analog Tx and analog Rx sharing one beam: only the diagonal is reachable
    metric_same = np.diag(numerator) / np.diag(denominator)
    best_same = int(np.argmax(np.diag(numerator))) #.  np.argmax(metric_same)
    w_same = CB[:, best_same][:, np.newaxis]

    # (b) analog Tx beam + digital (LMMSE) receive combining.
    # For a Tx beam w the best digital Rx attains, with a = H_b_hat w, b = H_SI_hat w,
    #   SINR(w) = Pvec a^H (Pvec b b^H + I)^-1 a
    #           = Pvec (|a|^2 - |b^H a|^2 * Pvec / (1 + Pvec |b|^2))   [Sherman-Morrison]
    a = H_b_hat @ w_same
    b = H_SI_hat @ w_same
    v_dig = np.linalg.solve(Pvec*(b) @ np.conj(b).T + np.eye(N), Pvec*a)
    v_dig = v_dig / np.linalg.norm(v_dig)

    # (c) analog Tx and analog Rx picked independently from the codebook.
    # One slot per beam PAIR, so this sweep runs on the sqrt(tau)-wide codebook.
    num_pair = np.abs(np.conj(CB_pair).T @ H_b_hat @ CB_pair)**2
    den_pair = np.abs(np.conj(CB_pair).T @ H_SI_hat @ CB_pair)**2 + 1/Pvec
    best_rx, best_tx = np.unravel_index(np.argmax(num_pair / den_pair), num_pair.shape)
    w_ana = CB_pair[:, best_tx][:, np.newaxis]
    v_ana = CB_pair[:, best_rx][:, np.newaxis]

    return (w_same, w_same), (w_same, v_dig), (w_ana, v_ana)


def sweep_sinr(w, v, H_b, H_SI, Pvec):
    """SINR actually delivered by the pair (w, v) on the TRUE channels."""
    sig = Pvec * np.abs(np.transpose(np.conj(v)) @ H_b @ w)**2
    return sig / (Pvec * np.abs(np.transpose(np.conj(v)) @ H_SI @ w)**2 + 1)

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

# %%

sinr_opti_recal = []
rieman_opti_sinr_1b = []
sinr_test_1b = []
sinr_sweeping_1b = []          # analog Tx = analog Rx (w = v), estimated SI
sinr_sweeping_1b_csi = []      # analog Tx = analog Rx (w = v), perfect SI

sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweeping_2b = []          # analog Tx + digital LMMSE Rx, estimated SI
sinr_sweeping_2b_csi = []      # analog Tx + digital LMMSE Rx, perfect SI
sinr_sweeping_2b_ana = []      # analog Tx + analog Rx, different beams, estimated SI
sinr_sweeping_2b_ana_csi = []  # analog Tx + analog Rx, different beams, perfect SI

sinr_opt_hat_1b = []
sinr_opt_hat_2b = []

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
                                None, location_ris_1, num_samples=800, Rician_factor=Rician_factor)
        Pvec = 10**(snr/10) / (Wavelength**4 / (4 *np.pi *ref_dis)**4) / (N_ris)**2

        H_I_hat_batch = []
        H_b_hat_batch = []
        H_b_batch = []
        H_d_batch = []

        for j in range(len(loc_true)):
            loc = loc_true[j].squeeze()
            loc_cartesian = np.array([[loc[1]*np.cos(loc[0]), loc[1]*np.sin(loc[0]), -20]])

            H_SI = channel_true_val[0]
            H_b = channel_true_val[2][j]

            H_b_batch.append(H_b)
            H_d_batch.append(H_SI)

            ### Lower Bound Beam Sweeping with Imperfect SI Channel Estimation
            CB = make_codebook(N_ris, tau)
            CB_pair = make_codebook(N_ris, pair_sweep_size(tau))   # sqrt(tau) per side
            Y_observe_SI = np.sqrt(Pvec)*(H_SI)@CB + np.sqrt(1/2)*(np.random.randn(*CB.shape) + 1j * np.random.randn(*CB.shape))

            H_SI_hat = np.linalg.lstsq(np.sqrt(Pvec)*CB.T, Y_observe_SI.T, rcond=None)[0].T 

            Y_observe = np.sqrt(Pvec)*(H_SI+H_b)@CB + np.sqrt(1/2)*(np.random.randn(*CB.shape) + 1j * np.random.randn(*CB.shape))
            H_2_hat = np.linalg.lstsq(np.sqrt(Pvec)*CB.T, Y_observe.T, rcond=None)[0].T
            
            H_b_hat = H_2_hat - H_SI_hat

            H_b_hat_batch.append(H_b_hat)
            H_I_hat_batch.append(H_SI_hat)            

            (w_same, v_same), (w_dig, v_dig), (w_ana, v_ana) = \
                sweep_designs(H_b_hat, H_SI_hat, CB, Pvec, CB_pair)

            sinr_sweeping_1b.append(sweep_sinr(w_same, v_same, H_b, H_SI, Pvec))
            sinr_sweeping_2b.append(sweep_sinr(w_dig, v_dig, H_b, H_SI, Pvec))
            sinr_sweeping_2b_ana.append(sweep_sinr(w_ana, v_ana, H_b, H_SI, Pvec))


              ##### Beam sweeping with Perfect SI Channel Estimation
            # H_b_hat = (observation - H_SI*np.sqrt(Pvec)) / np.sqrt(Pvec)

            # (w_same, v_same), (w_dig, v_dig), (w_ana, v_ana) = \
            #     sweep_designs(H_b_hat, H_SI, CB, Pvec, CB_pair)

            # sinr_sweeping_1b_csi.append(sweep_sinr(w_same, v_same, H_b, H_SI, Pvec))
            # sinr_sweeping_2b_csi.append(sweep_sinr(w_dig, v_dig, H_b, H_SI, Pvec))
            # sinr_sweeping_2b_ana_csi.append(sweep_sinr(w_ana, v_ana, H_b, H_SI, Pvec))


        ##### Optimal beams from H_hat
        # v_sp, w_sp = compute_optimal_beamformers_2b(
        #                      np.array(H_b_hat_batch), np.array(H_I_hat_batch),1, Pvec, num_restarts=1)
        
        # sinr_opt_2b, _, _, _ = compute_beamformer_metrics_batch(
        #         np.array(H_d_batch),np.array(H_b_batch),np.zeros_like(H_b_batch),
        #         np.array(v_sp),np.array(w_sp),1,Pvec,
        # )
        # sinr_opt_hat_2b.append(sinr_opt_2b)

        # vw_opt = compute_optimal_beamformers_1b(
        #         np.array(H_b_hat_batch),np.array(H_I_hat_batch),
        #         1,Pvec,num_restarts=1,
        # )
        # sinr_opt_1b, _, _, _ = compute_beamformer_metrics_batch(
        #                 np.array(H_d_batch),np.array(H_b_batch),np.zeros_like(H_b_batch),
        #                 np.array(vw_opt),np.array(vw_opt),1,Pvec,
        # )
        # sinr_opt_hat_1b.append(sinr_opt_1b)

sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b_ana = np.mean(np.reshape(sinr_sweeping_2b_ana, (len(snr_const), -1)), axis=1)
# sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(snr_const), -1)), axis=1)
# sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(snr_const), -1)), axis=1)
# sinr_sweeping_2b_ana_csi = np.mean(np.reshape(sinr_sweeping_2b_ana_csi, (len(snr_const), -1)), axis=1)

sinr_opt_hat_1b = 0 if len(sinr_opt_hat_1b) == 0 else np.mean(np.reshape(sinr_opt_hat_1b, (len(snr_const), -1)), axis=1)
sinr_opt_hat_2b = 0 if len(sinr_opt_hat_2b) == 0 else np.mean(np.reshape(sinr_opt_hat_2b, (len(snr_const), -1)), axis=1)


# fig, ax = plt.subplots(1, 2, figsize=(7, 3))
# ax[0].plot(snr_const, 10*np.log10(sig_pow_opti_recal))
# ax[0].set_title('Signal Power [dB] (Optimal)')
# ax[1].plot(snr_const, 10*np.log10(int_pow_opti_recal))
# ax[1].set_title('Interference Power [dB] (Optimal)')


# %%

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

ax.plot(snr_const, 10*np.log10(sinr_opt_hat_1b.squeeze()), \
       marker=methods['iteropti_hat']['marker'], linestyle='--',
       color=methods['iteropti_hat']['color'], linewidth=1.2, markersize=6,
       label='IterOpti_hat')

# w ≠ v (dashed lines)
ax.plot(snr_const, [10*np.log10(np.mean(p)) for p in sinr_test_2b], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, [np.mean(p) for p in rieman_opti_sinr_2b], \
       marker=methods['iteropti']['marker'], linestyle='-', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b_ana.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='-',
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6)

ax.plot(snr_const, 10*np.log10(sinr_opt_hat_2b.squeeze()), \
       marker=methods['iteropti_hat']['marker'], linestyle='-',
       color=methods['iteropti_hat']['color'], linewidth=1.2, markersize=6,
       label='IterOpti_hat')

ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['sweep_dig']['marker'], linestyle='-',
       color=methods['sweep_dig']['color'], linewidth=1.2, markersize=6)


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
legend1 = ax.legend(handles=method_legend, loc='upper left', ncols=2,
                   frameon=True, fontsize=8, fancybox=True, framealpha=0.5)
legend2 = ax.legend(handles=style_legend, loc='lower right',
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.5)

# legend1.get_texts()[2].set_position((0, 2))
from matplotlib.transforms import ScaledTranslation
offset = ScaledTranslation(0, 3 / 72, fig.dpi_scale_trans)  # 2 points upward
legend1.get_texts()[2].set_transform(legend1.get_texts()[2].get_transform() + offset)
legend1.legend_handles[2].set_transform(
    legend1.legend_handles[2].get_transform() + offset
)
# Add the first legend back (matplotlib removes it when creating the second)
ax.add_artist(legend1)

ax.set_xlabel('Effective SNR [dB]')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xticks(snr_const)
ax.set_ylim([-40, 15])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/sinr_snr_1BD_v2.pdf', format = 'pdf', bbox_inches = 'tight')


#####################################################################################
# -----------------------------SINR vs Tau    ------------------------------------- #
#####################################################################################
# %%
snr = 10  
tau = [4,6,8,10,12,14,16]
n_opt = 40

sinr_opti_recal = []
rieman_opti_sinr_1b = []
sinr_test_1b = []
sinr_sweeping_1b = []          # analog Tx = analog Rx (w = v), estimated SI
sinr_sweeping_1b_csi = []      # analog Tx = analog Rx (w = v), perfect SI

sinr_opt_hat_1b = []
sinr_opt_hat_2b = []

sinr_test_2b = []
rieman_opti_sinr_2b = []
sinr_sweeping_2b = []          # analog Tx + digital LMMSE Rx, estimated SI
sinr_sweeping_2b_csi = []      # analog Tx + digital LMMSE Rx, perfect SI
sinr_sweeping_2b_ana = []      # analog Tx + analog Rx, different beams, estimated SI
sinr_sweeping_2b_ana_csi = []  # analog Tx + analog Rx, different beams, perfect SI
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
                                None, location_ris_1, num_samples=7000, Rician_factor=Rician_factor)   

        H_I_hat_batch = []
        H_b_hat_batch = []
        H_b_batch = []
        H_d_batch = []

        CB = make_codebook(N_ris, n_tau)
        CB_pair = make_codebook(N_ris, pair_sweep_size(n_tau))  # sqrt(tau) per side

        for j in range(len(loc_true)):
            loc = loc_true[j].squeeze()
            loc_cartesian = np.array([[loc[1]*np.cos(loc[0]), loc[1]*np.sin(loc[0]), -20]])

            H_SI = channel_true_val[0]
            H_b = channel_true_val[2][j]

            H_b_batch.append(H_b)
            H_d_batch.append(H_SI)

            ### Optimal Beamforming from \cite{barneto_beamformer_2021} 
            # steer_vec = np.exp(1j * np.pi * np.arange(N_ris) * np.sin(loc[0]))

            # theta_star = np.conj(steer_vec[:,np.newaxis])/np.linalg.norm(steer_vec)
            # N = np.eye(N_ris) - ((H_SI @ theta_star) @ np.conj(H_SI @ theta_star).transpose())\
            #                               /np.linalg.norm(H_SI @ theta_star)**2
            # v_star = (N @ steer_vec) / np.linalg.norm(N @ steer_vec)
            # sig_pow_opti_recal.append(Pvec * np.abs(np.transpose(np.conj(v_star)) @ H_b @ theta_star)**2)
            # int_pow_opti_recal.append(Pvec * np.abs(np.transpose(np.conj(v_star)) @ H_SI @ theta_star)**2)
            # sinr_opti_recal.append(sig_pow_opti_recal[-1] / (int_pow_opti_recal[-1] + 1))

            ### Lower Bound Beam Sweeping with Imperfect SI Channel Estimation
            
            Y_observe_SI = np.sqrt(Pvec)*(H_SI)@CB + np.sqrt(1/2)*(np.random.randn(*CB.shape) + 1j * np.random.randn(*CB.shape))
            
            H_SI_hat = Y_observe_SI @ np.linalg.pinv(CB) / np.sqrt(Pvec)

            Y_observe = np.sqrt(Pvec)*(H_SI+H_b)@CB + np.sqrt(1/2)*(np.random.randn(*CB.shape) + 1j * np.random.randn(*CB.shape))
            H_2_hat = Y_observe @ np.linalg.pinv(CB)/ np.sqrt(Pvec)
            # numerator = np.linalg.norm(Y_observe - np.sqrt(Pvec)*H_SI_hat@CB, axis=0)**2
            # denominator = np.linalg.norm(H_SI_hat@CB, axis=0)**2 * Pvec
            # inx = np.argmax(numerator/(denominator+1))
            # w_same = CB[:, inx][:, np.newaxis]
            # v_same = w_same

            # a = H_b_hat @ w_same
            # b = H_SI_hat @ w_same
            # v_dig = np.linalg.solve(Pvec*(b) @ np.conj(b).T + np.eye(CB.shape[0]), Pvec*a)
            # v_dig = v_dig / np.linalg.norm(v_dig)

            # Y_observe_pair = np.sqrt(Pvec)*(H_SI+H_b)@CB_pair + \
            #                 np.sqrt(1/2)*(np.random.randn(*CB_pair.shape) + 1j * np.random.randn(*CB_pair.shape))
            # num_pair = np.abs(np.conj(CB_pair).T @ (Y_observe_pair - np.sqrt(Pvec)*H_SI_hat @ CB_pair)**2)
            # den_pair = np.abs(np.conj(CB_pair).T @ H_SI_hat @ CB_pair)**2 + 1/Pvec
            # best_rx, best_tx = np.unravel_index(np.argmax(num_pair / den_pair), num_pair.shape)
            # w_ana = CB_pair[:, best_tx][:, np.newaxis]
            # v_ana = CB_pair[:, best_rx][:, np.newaxis]
            
            H_b_hat = H_2_hat - H_SI_hat
            if j < n_opt:
                H_b_hat_batch.append(H_b_hat)
                H_I_hat_batch.append(H_SI_hat)     

            (w_same, v_same), (w_dig, v_dig), (w_ana, v_ana) = \
                sweep_designs(H_b_hat, H_SI_hat, CB, Pvec, CB_pair)

            sinr_sweeping_1b.append(sweep_sinr(w_same, v_same, H_b, H_SI, Pvec))
            sinr_sweeping_2b.append(sweep_sinr(w_dig, v_dig, H_b, H_SI, Pvec))
            sinr_sweeping_2b_ana.append(sweep_sinr(w_ana, v_ana, H_b, H_SI, Pvec))

        ##### Optimal beams from H_hat
        v_sp, w_sp = compute_optimal_beamformers_2b(
                                np.array(H_b_hat_batch), np.array(H_I_hat_batch),1, Pvec, num_restarts=3)
        
        sinr_opt_2b, _, _, _ = compute_beamformer_metrics_batch(
                np.array(H_d_batch[:n_opt]),np.array(H_b_batch[:n_opt]),np.zeros_like(H_b_batch[:n_opt]),
                np.array(v_sp),np.array(w_sp),1,Pvec,
        )
        sinr_opt_hat_2b.append(sinr_opt_2b)

        vw_opt = compute_optimal_beamformers_1b(
                np.array(H_b_hat_batch),np.array(H_I_hat_batch),
                1,Pvec,num_restarts=3,
        )
        sinr_opt_1b, _, _, _ = compute_beamformer_metrics_batch(
                        np.array(H_d_batch[:n_opt]),np.array(H_b_batch[:n_opt]),np.zeros_like(H_b_batch[:n_opt]),
                        np.array(vw_opt),np.array(vw_opt),1,Pvec,
        )
        sinr_opt_hat_1b.append(sinr_opt_1b)



sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(tau), -1)), axis=1)
sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(tau), -1)), axis=1)
sinr_sweeping_2b_ana = np.mean(np.reshape(sinr_sweeping_2b_ana, (len(tau), -1)), axis=1)
# sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(tau), -1)), axis=1)
# sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(tau), -1)), axis=1)
# sinr_sweeping_2b_ana_csi = np.mean(np.reshape(sinr_sweeping_2b_ana_csi, (len(tau), -1)), axis=1)

sinr_opt_hat_1b = [] if len(sinr_opt_hat_1b) == 0 else np.mean(np.reshape(sinr_opt_hat_1b, (len(tau), -1)), axis=1)
sinr_opt_hat_2b = [] if len(sinr_opt_hat_2b) == 0 else np.mean(np.reshape(sinr_opt_hat_2b, (len(tau), -1)), axis=1)


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
ax.plot(tau, 10*np.log10(sinr_opt_hat_1b.squeeze()), \
       marker=methods['iteropti_hat']['marker'], linestyle='--',
       color=methods['iteropti_hat']['color'], linewidth=1.2, markersize=6,
       label='IterOpti_hat')

### w ≠ v (solid lines)
ax.plot(tau, [10*np.log10(np.mean(p)) for p in sinr_test_2b], \
       marker=methods['proposed']['marker'], linestyle='-', 
       color=methods['proposed']['color'], linewidth=1.2, markersize=6)        
ax.plot(tau, [np.mean(p) for p in rieman_opti_sinr_2b], \
       marker=methods['iteropti']['marker'], linestyle='-', 
       color=methods['iteropti']['color'], linewidth=1.2, markersize=6)
ax.plot(tau, 10*np.log10(sinr_sweeping_2b_ana.squeeze()), \
       marker=methods['beam_sweep']['marker'], linestyle='-',
       color=methods['beam_sweep']['color'], linewidth=1.2, markersize=6)
ax.plot(tau, 10*np.log10(sinr_opt_hat_2b.squeeze()), \
       marker=methods['iteropti_hat']['marker'], linestyle='-',
       color=methods['iteropti_hat']['color'], linewidth=1.2, markersize=6,
       label='IterOpti_hat')
ax.plot(tau, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['sweep_dig']['marker'], linestyle='-',
       color=methods['sweep_dig']['color'], linewidth=1.2, markersize=6)

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
]
# Line style legend
style_legend = [
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1.2, 
           label=r'$\mathbf{w} = \mathbf{v}$'), 
    Line2D([0], [0], color='grey', linestyle='-', linewidth=1.2, 
           label=r'$\mathbf{w} \neq \mathbf{v}$')
]
# Create two separate legends
legend1 = ax.legend(handles=method_legend, loc='center left', ncols =2,
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.6, borderpad=0.25,
    labelspacing=0.25,
    handletextpad=0.35,
                     )
                     
legend2 = ax.legend(handles=style_legend, loc='lower right', 
                   frameon=True, fontsize=9, fancybox=True, framealpha=0.6
                     )
legend1.set_bbox_to_anchor((0.0, 0.57))
from matplotlib.transforms import ScaledTranslation
offset = ScaledTranslation(0, 3 / 72, fig.dpi_scale_trans)  # 2 points upward
legend1.get_texts()[2].set_transform(legend1.get_texts()[2].get_transform() + offset)
legend1.legend_handles[2].set_transform(
    legend1.legend_handles[2].get_transform() + offset
)

ax.add_artist(legend1)

ax.set_xlabel('Preamble Length $T$')
ax.set_ylabel('Achieved SINR [dB]')
ax.set_xticks(tau)
# ax.set_ylim([-25, 25])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/sinr_tau_1BD_v2.pdf', format = 'pdf', bbox_inches = 'tight')
# %%
