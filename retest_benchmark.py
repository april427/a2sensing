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

# args = parse_args()

N_ris = 16
tau = 10  
snr_const = [-10,-5, 0, 5, 10, 15, 20,25]
ref_dis = 5
Wavelength = 3e8/(10e9)

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
    'sweep_dig_csi': {'color': '#8c564b', 'marker': '*'}     # Brown stars       (digital Rx, perfect SI)
}


def beam_groups(n, m, balanced=False):
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


def grouped_dft_codebook(n, m, balanced=False, phase='quadratic'):
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


def sweep_designs(H_b_hat, H_SI_hat, CB, Pvec, digital_tx='diag'):
    """Codebook beam sweeping on the ESTIMATED channels (as in retest_scatters.py).
    """
    N = CB.shape[0]
    numerator = np.abs(np.conj(CB).T @ H_b_hat @ CB)**2
    denominator = np.abs(np.conj(CB).T @ H_SI_hat @ CB)**2 + 1/Pvec

    # (a) analog Tx and analog Rx sharing one beam: only the diagonal is reachable
    metric_same = np.diag(numerator) / np.diag(denominator)
    best_same = int(np.argmax(np.diag(numerator)))
    w_same = CB[:, best_same][:, np.newaxis]

    # (b) analog Tx beam + digital (LMMSE) receive combining.
    # For a Tx beam w the best digital Rx attains, with a = H_b_hat w, b = H_SI_hat w,
    #   SINR(w) = Pvec a^H (Pvec b b^H + I)^-1 a
    #           = Pvec (|a|^2 - |b^H a|^2 * Pvec / (1 + Pvec |b|^2))   [Sherman-Morrison]
    a = H_b_hat @ w_same
    b = H_SI_hat @ w_same
    v_dig = np.linalg.solve(Pvec*(b) @ np.conj(b).T + np.eye(N), Pvec*a)
    v_dig = v_dig / np.linalg.norm(v_dig)

    # (c) analog Tx and analog Rx picked independently from the codebook
    best_rx, best_tx = np.unravel_index(np.argmax(numerator / denominator), numerator.shape)
    w_ana = CB[:, best_tx][:, np.newaxis]
    v_ana = CB[:, best_rx][:, np.newaxis]

    return (w_same, w_same), (w_same, v_dig), (w_ana, v_ana)


def sweep_sinr(w, v, H_b, H_SI, Pvec):
    """SINR actually delivered by the pair (w, v) on the TRUE channels."""
    sig = Pvec * np.abs(np.transpose(np.conj(v)) @ H_b @ w)**2
    return sig / (Pvec * np.abs(np.transpose(np.conj(v)) @ H_SI @ w)**2 + 1)

# %%
sig_pow_opti_recal = []
int_pow_opti_recal = []
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
iterative_generalized_eig = []
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

        for j in range(len(loc_true)):
            loc = loc_true[j].squeeze()
            loc_cartesian = np.array([[loc[1]*np.cos(loc[0]), loc[1]*np.sin(loc[0]), -20]])

            H_SI = channel_true_val[0]
            H_b = channel_true_val[2][j]

            ### Lower Bound Beam Sweeping with Imperfect SI Channel Estimation
            CB = make_codebook(N_ris, tau)
            H_SI_hat =   H_SI + (np.random.randn(*H_SI.shape) + 1j * np.random.randn(*H_SI.shape))/np.sqrt(2*Pvec)
            observation = np.sqrt(Pvec)*channel_true_val[1][j].squeeze() + 1/np.sqrt(2) *(np.random.randn(*channel_true_val[1][j].squeeze().shape) + 1j * np.random.randn(*channel_true_val[1][j].squeeze().shape))
            H_b_hat = (observation - H_SI_hat*np.sqrt(Pvec)) / np.sqrt(Pvec)

            (w_same, v_same), (w_dig, v_dig), (w_ana, v_ana) = \
                sweep_designs(H_b_hat, H_SI_hat, CB, Pvec)

            sinr_sweeping_1b.append(sweep_sinr(w_same, v_same, H_b, H_SI, Pvec))
            sinr_sweeping_2b.append(sweep_sinr(w_dig, v_dig, H_b, H_SI, Pvec))
            sinr_sweeping_2b_ana.append(sweep_sinr(w_ana, v_ana, H_b, H_SI, Pvec))


              ##### Beam sweeping with Perfect SI Channel Estimation
            H_b_hat = (observation - H_SI*np.sqrt(Pvec)) / np.sqrt(Pvec)

            (w_same, v_same), (w_dig, v_dig), (w_ana, v_ana) = \
                sweep_designs(H_b_hat, H_SI, CB, Pvec)

            sinr_sweeping_1b_csi.append(sweep_sinr(w_same, v_same, H_b, H_SI, Pvec))
            sinr_sweeping_2b_csi.append(sweep_sinr(w_dig, v_dig, H_b, H_SI, Pvec))
            sinr_sweeping_2b_ana_csi.append(sweep_sinr(w_ana, v_ana, H_b, H_SI, Pvec))


              ##### Iterative generalized eigenvalue optimization
       #      A = np.sqrt(Pvec) * H_b
       #      B = np.sqrt(Pvec) * H_SI
       # #      v_star, theta_star, sinr_star = alternating_xy(A, B, c=1, y0=None, max_iter=200, tol=1e-6, verbose=False)
       #      theta_star, _ = solve_transpose_with_manual_restarts(A, B, 1, restarts=5, seed=123)
       #      sinr_star = Pvec * np.abs(np.transpose(np.conj(theta_star)) @ H_b @ theta_star)**2 / \
       #                          (Pvec * np.abs(np.transpose(np.conj(theta_star)) @ H_SI @ theta_star)**2 + 1)
       #      iterative_generalized_eig.append(sinr_star)



sig_pow_opti_recal = np.mean(np.reshape(sig_pow_opti_recal, (len(snr_const), -1)), axis=1)
int_pow_opti_recal = np.mean(np.reshape(int_pow_opti_recal, (len(snr_const), -1)), axis=1)
sinr_opti_recal = np.mean(np.reshape(sinr_opti_recal, (len(snr_const), -1)), axis=1)
sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b_ana = np.mean(np.reshape(sinr_sweeping_2b_ana, (len(snr_const), -1)), axis=1)
sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(snr_const), -1)), axis=1)
sinr_sweeping_2b_ana_csi = np.mean(np.reshape(sinr_sweeping_2b_ana_csi, (len(snr_const), -1)), axis=1)
# iterative_generalized_eig = np.mean(np.reshape(iterative_generalized_eig, (len(snr_const), -1)), axis=1)

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

# ax.plot(snr_const, 10*np.log10(sinr_sweeping_1b_csi.squeeze()), \
#        marker=methods['beam_sweep_csi']['marker'], linestyle='--', 
#        color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
#        label='Beam Sweeping (Perfect SI CSI)')

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

# ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b_ana_csi.squeeze()), \
#        marker=methods['beam_sweep_csi']['marker'], linestyle='-',
#        color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
#        label='Beam Sweeping (Perfect SI CSI)')

ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['sweep_dig']['marker'], linestyle='-',
       color=methods['sweep_dig']['color'], linewidth=1.2, markersize=6)

# ax.plot(snr_const, 10*np.log10(sinr_sweeping_2b_csi.squeeze()), \
#        marker=methods['sweep_dig_csi']['marker'], linestyle='-',
#        color=methods['sweep_dig_csi']['color'], linewidth=1.2, markersize=7)

# ax.plot(snr_const, 10*np.log10(iterative_generalized_eig.squeeze()), \
#        marker='^', linestyle=':', 
#        color="#000000", linewidth=1.2, markersize=6,
#        label='Iterative Gen. Eig.')

# Create custom legend
from matplotlib.lines import Line2D

# Method legend (colors/markers)
method_legend = [
    Line2D([0], [0], color=methods['beam_sweep']['color'],
           marker=methods['beam_sweep']['marker'], linestyle='None',
           markersize=7, label=r'Sweep (Analog $\mathbf{w}$)'),
#     Line2D([0], [0], color=methods['beam_sweep_csi']['color'],
#            marker=methods['beam_sweep_csi']['marker'], linestyle='None',
#            markersize=7, label=r'Sweep, Analog $\mathbf{v}$ ($\mathbf{H}_{\rm SI}$)'),
    Line2D([0], [0], color=methods['sweep_dig']['color'],
           marker=methods['sweep_dig']['marker'], linestyle='None',
           markersize=7, label=r'Sweep (Digital $\mathbf{w}$)'),
#     Line2D([0], [0], color=methods['sweep_dig_csi']['color'],
#            marker=methods['sweep_dig_csi']['marker'], linestyle='None',
#            markersize=7, label=r'Sweep, Digital $\mathbf{v}$ ($\mathbf{H}_{\rm SI}$)'),
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
legend1 = ax.legend(handles=method_legend, loc='upper left', ncols=2,
                   frameon=True, fontsize=8, fancybox=True, framealpha=0.5)
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
# -----------------------------SINR vs Tau    ------------------------------------- #
#####################################################################################
# %%
snr = 10  
tau = [4,6,8,10,12,14,16]

sig_pow_opti_recal = []
int_pow_opti_recal = []
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
                                None, location_ris_1, num_samples=3000, Rician_factor=Rician_factor)   

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

            ### Lower Bound Beam Sweeping with Imperfect SI Channel Estimation
            CB = make_codebook(N_ris, n_tau)
            H_SI_hat =   H_SI + (np.random.randn(*H_SI.shape) + 1j * np.random.randn(*H_SI.shape))/np.sqrt(2*Pvec)
            observation = np.sqrt(Pvec)*channel_true_val[1][j].squeeze() + \
                     1/np.sqrt(2) *(np.random.randn(*channel_true_val[1][j].squeeze().shape) \
                                    + 1j * np.random.randn(*channel_true_val[1][j].squeeze().shape))
            H_b_hat = (observation - H_SI_hat*np.sqrt(Pvec)) / np.sqrt(Pvec)

            (w_same, v_same), (w_dig, v_dig), (w_ana, v_ana) = \
                sweep_designs(H_b_hat, H_SI_hat, CB, Pvec)

            sinr_sweeping_1b.append(sweep_sinr(w_same, v_same, H_b, H_SI, Pvec))
            sinr_sweeping_2b.append(sweep_sinr(w_dig, v_dig, H_b, H_SI, Pvec))
            sinr_sweeping_2b_ana.append(sweep_sinr(w_ana, v_ana, H_b, H_SI, Pvec))

            ### Beam Sweeping with Perfect SI Channel Estimation

            H_b_hat = (observation - H_SI*np.sqrt(Pvec)) / np.sqrt(Pvec)

            (w_same, v_same), (w_dig, v_dig), (w_ana, v_ana) = \
                sweep_designs(H_b_hat, H_SI, CB, Pvec)

            sinr_sweeping_1b_csi.append(sweep_sinr(w_same, v_same, H_b, H_SI, Pvec))
            sinr_sweeping_2b_csi.append(sweep_sinr(w_dig, v_dig, H_b, H_SI, Pvec))
            sinr_sweeping_2b_ana_csi.append(sweep_sinr(w_ana, v_ana, H_b, H_SI, Pvec))


sinr_sweeping_1b = np.mean(np.reshape(sinr_sweeping_1b, (len(tau), -1)), axis=1)
sinr_sweeping_2b = np.mean(np.reshape(sinr_sweeping_2b, (len(tau), -1)), axis=1)
sinr_sweeping_2b_ana = np.mean(np.reshape(sinr_sweeping_2b_ana, (len(tau), -1)), axis=1)
sinr_sweeping_1b_csi = np.mean(np.reshape(sinr_sweeping_1b_csi, (len(tau), -1)), axis=1)
sinr_sweeping_2b_csi = np.mean(np.reshape(sinr_sweeping_2b_csi, (len(tau), -1)), axis=1)
sinr_sweeping_2b_ana_csi = np.mean(np.reshape(sinr_sweeping_2b_ana_csi, (len(tau), -1)), axis=1)

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
# ax.plot(tau, 10*np.log10(sinr_sweeping_1b_csi.squeeze()), \
#        marker=methods['beam_sweep_csi']['marker'], linestyle='--', 
#        color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
#        label='Beam Sweeping')

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
# ax.plot(tau, 10*np.log10(sinr_sweeping_2b_ana_csi.squeeze()), \
#        marker=methods['beam_sweep_csi']['marker'], linestyle='-',
#        color=methods['beam_sweep_csi']['color'], linewidth=1.2, markersize=6,
#        label='Beam Sweeping')
ax.plot(tau, 10*np.log10(sinr_sweeping_2b.squeeze()), \
       marker=methods['sweep_dig']['marker'], linestyle='-',
       color=methods['sweep_dig']['color'], linewidth=1.2, markersize=6)
# ax.plot(tau, 10*np.log10(sinr_sweeping_2b_csi.squeeze()), \
#        marker=methods['sweep_dig_csi']['marker'], linestyle='-',
#        color=methods['sweep_dig_csi']['color'], linewidth=1.2, markersize=7)
# Create custom legend
from matplotlib.lines import Line2D
# Method legend (colors/markers)
method_legend = [
    Line2D([0], [0], color=methods['beam_sweep']['color'],
           marker=methods['beam_sweep']['marker'], linestyle='None',
           markersize=7, label=r'Sweep (Analog $\mathbf{w}$)'),
#     Line2D([0], [0], color=methods['beam_sweep_csi']['color'],
#            marker=methods['beam_sweep_csi']['marker'], linestyle='None',
#            markersize=7, label=r'Sweep, Analog $\mathbf{v}$ ($\mathbf{H}_{\rm SI}$)'),
    Line2D([0], [0], color=methods['sweep_dig']['color'],
           marker=methods['sweep_dig']['marker'], linestyle='None',
           markersize=7, label=r'Sweep (Digital $\mathbf{w}$)'),
#     Line2D([0], [0], color=methods['sweep_dig_csi']['color'],
       #     marker=methods['sweep_dig_csi']['marker'], linestyle='None',
       #     markersize=7, label=r'Sweep, Digital $\mathbf{v}$ ($\mathbf{H}_{\rm SI}$)'),
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
ax.set_ylim([-25, 25])
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
# plt.savefig('figs/sinr_tau_1BD.pdf', format = 'pdf', bbox_inches = 'tight')
# %%
