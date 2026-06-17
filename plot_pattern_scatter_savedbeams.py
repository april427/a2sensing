#!/usr/bin/env python3
"""
Plot beam pattern evolution over tau directly from saved TEST data.

This script reads v_list_test and w_list_test from saved TEST .mat files
and visualizes the joint beam pattern across sensing steps. It does not
restore or run any neural network model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from parse_args import parse_args
from channel_functions import generate_mimo_channel
from manifold_optimization import solve_with_random_restarts

def compute_optimal_beamformers(H_b_batch, H_int_batch, noise_var_val, P_val, num_restarts=10):
    """Compute optimal beamformers for a batch - reduced restarts for 2x speedup"""
    v_opt_batch = np.zeros(( H_b_batch.shape[0], 1), dtype=np.complex64)
    w_opt_batch = np.zeros(( H_b_batch.shape[1], 1), dtype=np.complex64)

    H_b_i = H_b_batch
    H_int_i = H_int_batch
    A = np.sqrt(P_val) * H_b_i
    B = np.sqrt(P_val) * H_int_i
    try:
        # Reduced restarts from 10 to 5 for ~2x speedup with minimal quality loss
        w_opt_i, v_opt_i, _ = solve_with_random_restarts(A, B, c=noise_var_val, restarts=num_restarts)
        v_opt_batch[:, 0] = v_opt_i
        w_opt_batch[:, 0] = w_opt_i
    except:
        # Fallback to SVD
        U, _, Vh = np.linalg.svd(H_b_i)
        v_opt_batch[:, 0] = U[:, 0]
        w_opt_batch[:, 0] = Vh[0, :]
    
    return v_opt_batch.astype(np.complex64), w_opt_batch.astype(np.complex64)

def pick_test_mat_file(result_dir, n_ant, tau, snr_db, n_symbols, n_scatters):
    """Pick TEST .mat file by exact naming first, fallback to latest TEST file."""
    exact = os.path.join(
        result_dir,
        f"TEST_sinr_N_{n_ant}_{n_ant}_tau_{tau}_snr_{int(snr_db)}_K_{n_symbols}_Nsca_{n_scatters}.mat",
    )
    if os.path.isfile(exact):
        return exact

    candidates = [
        os.path.join(result_dir, f)
        for f in os.listdir(result_dir)
        if f.startswith("TEST_sinr_") and f.endswith(".mat")
    ]
    if not candidates:
        raise FileNotFoundError(f"No TEST_sinr*.mat found in {result_dir}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def steering_vector_ula(n_ant, angle_rad):
    idx = np.arange(n_ant)
    return np.exp(1j * np.pi * idx * np.sin(angle_rad))[:, np.newaxis]


def joint_beam_pattern_db(v_vec, w_vec, angles):
    gains = np.zeros_like(angles, dtype=np.float64)
    v = v_vec.reshape(-1, 1)
    w = w_vec.reshape(-1, 1)
    for i, angle in enumerate(angles):
        a = steering_vector_ula(v.shape[0], angle)
        h = a @ a.T
        gains[i] = np.abs(np.conj(v).T @ h @ w).item() ** 2
    gains = gains / (np.max(gains) + 1e-12)
    return 10.0 * np.log10(gains + 1e-12)


def extract_locations(mat_data, num_samples):
    bd = np.asarray(mat_data.get("BD_location", np.zeros((num_samples, 3), dtype=np.float64)))
    sca = np.asarray(mat_data.get("Scatter_location", np.zeros((num_samples, 0, 3), dtype=np.float64)))

    bd = np.squeeze(bd)
    if bd.ndim == 1 and bd.shape[0] == 3:
        bd = np.tile(bd[np.newaxis, :], (num_samples, 1))
    if bd.ndim != 2 or bd.shape[1] != 3:
        bd = np.zeros((num_samples, 3), dtype=np.float64)

    sca = np.squeeze(sca)
    if sca.size == 0:
        sca = np.zeros((num_samples, 0, 3), dtype=np.float64)
    elif sca.ndim == 2 and sca.shape[1] == 3:
        sca = sca[:, np.newaxis, :]
    elif sca.ndim != 3 or sca.shape[2] != 3:
        sca = np.zeros((num_samples, 0, 3), dtype=np.float64)

    if bd.shape[0] < num_samples:
        pad = np.zeros((num_samples - bd.shape[0], 3), dtype=np.float64)
        bd = np.concatenate([bd, pad], axis=0)
    if sca.shape[0] < num_samples:
        pad = np.zeros((num_samples - sca.shape[0], 0, 3), dtype=np.float64)
        sca = np.concatenate([sca, pad], axis=0)

    return bd, sca


def compute_single_sinr_db(v_vec, w_vec, h_b, h_d, h_r, p_tx, noise_var=1.0):
    """Compute SINR in dB for one beam pair and one channel realization."""
    v = v_vec.reshape(-1, 1)
    w = w_vec.reshape(-1, 1)
    h_int = h_d + h_r
    sig = p_tx * np.abs(np.conj(v).T @ h_b @ w).item() ** 2
    inter = p_tx * np.abs(np.conj(v).T @ h_int @ w).item() ** 2
    sinr = sig / (inter + noise_var)
    return 10.0 * np.log10(max(sinr, 1e-12)), 10.0 * np.log10(max(sig, 1e-12)), 10.0 * np.log10(max(inter, 1e-15))  



args = parse_args()

n_ant = args.N_ris
tau = 24
snr_db = args.snr
n_symbols = getattr(args, "N_symbols", 1)
n_scatters = 5
rician_factor = args.rician_factor

fc = args.fc
wavelength = 3e8 / fc
ref_dis = 15 * wavelength
p_tx = 10 ** (snr_db / 10) / (wavelength**4 / (4 * np.pi * ref_dis) ** 4) / n_ant / n_ant

location_tx = np.array([0.0, 0.0, 0.0])
location_rx = np.array([0.0, 0.0, 0.0])

result_dir = "Mo_mimo_sinr_modelsave"
if not os.path.isdir(result_dir):
    raise FileNotFoundError(f"Result directory not found: {result_dir}")

mat_file = pick_test_mat_file(result_dir, n_ant, tau, snr_db, n_symbols, n_scatters)
data = sio.loadmat(mat_file)
print(f"Loaded TEST data: {mat_file}")

if "v_list_test" not in data or "w_list_test" not in data:
    raise KeyError("Missing v_list_test or w_list_test in TEST data file")
 
v_list = np.asarray(data["v_list_test"])
w_list = np.asarray(data["w_list_test"])

# Expected shape is (tau+1, num_samples, N, 1). Squeeze trailing singleton dim.
v_list = np.squeeze(v_list)
w_list = np.squeeze(w_list)

if v_list.ndim != 3 or w_list.ndim != 3:
    raise ValueError(
        "Unexpected beam tensor shape. Expected (num_steps, num_samples, N) after squeeze, "
        f"got v={v_list.shape}, w={w_list.shape}"
    )

if v_list.shape != w_list.shape:
    raise ValueError(f"v_list_test and w_list_test shape mismatch: {v_list.shape} vs {w_list.shape}")

num_steps, num_samples, n_rx = v_list.shape
print(f"Beam tensor shape: steps={num_steps}, samples={num_samples}, antennas={n_rx}")

sinr_final = np.squeeze(np.asarray(data.get("sinr_learned", np.full((num_samples,), np.nan))))
if sinr_final.ndim == 0:
    sinr_final = np.full((num_samples,), float(sinr_final))
sinr_final = np.atleast_1d(sinr_final).astype(np.float64)
if sinr_final.size < num_samples:
    padded = np.full((num_samples,), np.nan, dtype=np.float64)
    padded[: sinr_final.size] = sinr_final
    sinr_final = padded

bd_locations, scatter_locations = extract_locations(data, num_samples)

angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
db_floor = -45.0
r_max = 46.0

n_show = [245]#np.random.choice(num_samples, size=1, replace=False)
print(f"Plotting {len(n_show)} sample(s)")

for sample_idx in n_show:
    bd = bd_locations[sample_idx]
    true_bd_angle_deg = np.degrees(np.arctan2(bd[1], bd[0])) if np.linalg.norm(bd[:2]) > 0 else np.nan

    sca = scatter_locations[sample_idx]
    scatter_angles = []
    for p in sca:
        if np.linalg.norm(p[:2]) > 0:
            scatter_angles.append(np.degrees(np.arctan2(p[1], p[0])))

    scatter_for_channel = np.asarray(sca)
    if scatter_for_channel.size == 0:
        scatter_for_channel = None

    _, h_d, h_r, h_b = generate_mimo_channel(
        location_tx,
        location_rx,
        scatter_for_channel,
        bd,
        N_tx_h=n_ant,
        N_tx_v=1,
        N_rx_h=n_ant,
        N_rx_v=1,
        wavelength=wavelength,
        Rician_factor=rician_factor,
    )
    
    metrics_steps_db = np.array(
        [
            compute_single_sinr_db(v_list[t, sample_idx], w_list[t, sample_idx], h_b, h_d, h_r, p_tx)
            for t in range(num_steps)
        ],
        dtype=np.float64,
    )
    sinr_steps_db, sig_steps_db, inter_steps_db = metrics_steps_db.T

    learned_final_sinr_db = float(sinr_steps_db[-1])
    beam_learned_final = joint_beam_pattern_db(v_list[-1, sample_idx], w_list[-1, sample_idx], angles)
    beam_r_learned_final = np.clip(beam_learned_final, db_floor, 1.0) - db_floor

    v_opt, w_opt = compute_optimal_beamformers(h_b, h_d + h_r, 1, p_tx, num_restarts=10)
    optimal_final_sinr_db, _, _ = compute_single_sinr_db(v_opt, w_opt, h_b, h_d, h_r, p_tx)
    beam_opt = joint_beam_pattern_db(v_opt, w_opt, angles)
    beam_r_opt = np.clip(beam_opt, db_floor, 1.0) - db_floor

    n_plot = num_steps + 1
    n_cols = 4
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.8 * n_rows),
        subplot_kw={"projection": "polar"},
        gridspec_kw={"wspace": 0.24, "hspace": 0.24},
    )
    axes = np.atleast_1d(axes).reshape(-1)

    for empty_idx in range(n_plot, len(axes)):
        axes[empty_idx].set_visible(False)

    for step_idx in range(num_steps):
        beam_db = joint_beam_pattern_db(v_list[step_idx, sample_idx], w_list[step_idx, sample_idx], angles)
        beam_r = np.clip(beam_db, db_floor, 1.0) - db_floor

        ax = axes[step_idx]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_ylim([0, r_max])
        ax.set_thetagrids(np.arange(-90, 91, 30))
        ax.set_rticks([0, 15, 30, 45])
        ax.set_yticklabels(["-45", "-30", "-15", "0"])
        ax.plot(angles, beam_r, color="#d62728", linewidth=1.0, label="Saved beam")

        if np.isfinite(true_bd_angle_deg) and -90.0 <= true_bd_angle_deg <= 90.0:
            theta_bd = np.deg2rad(true_bd_angle_deg)
            ax.plot([theta_bd, theta_bd], [0, r_max], color="#036221", linestyle="--", linewidth=1.8, label="BD")

        for s_idx, s_ang in enumerate(scatter_angles):
            if -90.0 <= s_ang <= 90.0:
                theta_s = np.deg2rad(s_ang)
                ax.plot(
                    [theta_s, theta_s],
                    [0, r_max],
                    color="#696969",
                    linestyle=":",
                    linewidth=1.8,
                    label="Scatterers" if s_idx == 0 else None,
                )
        
        ax.set_title(
            f"step={step_idx + 1}, SINR={sinr_steps_db[step_idx]:.2f} dB\n"
            f"sig={sig_steps_db[step_idx]:.2f} dB, intf={inter_steps_db[step_idx]:.2f} dB",
            fontsize=12,
            pad=1,
        )
        if step_idx == 0:
            ax.set_ylabel("Joint gain (dB)", labelpad=18)
            ax.legend(
                fontsize=8,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=3,
                frameon=True,
            )
        ax.grid(alpha=0.25)

    # Extra panel at the end: learned-final vs optimal comparison
    ax_cmp = axes[num_steps]
    ax_cmp.set_theta_zero_location("N")
    ax_cmp.set_theta_direction(-1)
    ax_cmp.set_thetamin(-90)
    ax_cmp.set_thetamax(90)
    ax_cmp.set_ylim([0, r_max])
    ax_cmp.set_thetagrids(np.arange(-90, 91, 30))
    ax_cmp.set_rticks([0, 15, 30, 45])
    ax_cmp.set_yticklabels(["-45", "-30", "-15", "0"])
    ax_cmp.grid(alpha=0.25)

    ax_cmp.plot(
        angles,
        beam_r_learned_final,
        color="#d62728",
        linewidth=1.0,
        label=f"Learned ({learned_final_sinr_db:.2f} dB)",
    )
    ax_cmp.plot(
        angles,
        beam_r_opt,
        color="#0E8E36",
        linewidth=1.0,
        linestyle="-.",
        label=f"Optimal ({optimal_final_sinr_db:.2f} dB)",
    )

    if np.isfinite(true_bd_angle_deg) and -90.0 <= true_bd_angle_deg <= 90.0:
        theta_bd = np.deg2rad(true_bd_angle_deg)
        ax_cmp.plot([theta_bd, theta_bd], [0, r_max], color="#036221", linestyle="--", linewidth=1.8, label="BD")

    for s_idx, s_ang in enumerate(scatter_angles):
        if -90.0 <= s_ang <= 90.0:
            theta_s = np.deg2rad(s_ang)
            ax_cmp.plot(
                [theta_s, theta_s],
                [0, r_max],
                color="#696969",
                linestyle=":",
                linewidth=1.8,
                label="Scatterers" if s_idx == 0 else None,
            )

    ax_cmp.set_title(
        f"Comparison, learned={learned_final_sinr_db:.2f} dB\n"
        f"optimal={optimal_final_sinr_db:.2f} dB",
        fontsize=12,
        pad=1,
    )
    ax_cmp.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.03, 1.12))

    sinr_text = "N/A"
    if sample_idx < sinr_final.size and np.isfinite(sinr_final[sample_idx]):
        sinr_text = f"{10.0 * np.log10(max(sinr_final[sample_idx], 1e-12)):.2f} dB"

    # fig.suptitle(
    #     f"Sample {sample_idx + 1} | Final learned SINR={learned_final_sinr_db:.2f} dB | "
    #     f"BD angle={true_bd_angle_deg:.1f} deg | Scatter angles={np.round(scatter_angles, 1).tolist()}"
    # )
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.04, top=0.96)
    plt.savefig('figs/beam_pattern_evolving.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    print(
        f"Sample {sample_idx + 1}: learned final SINR={learned_final_sinr_db:.2f} dB, "
        f"optimal SINR={optimal_final_sinr_db:.2f} dB"
    )

    fig1, ax1 = plt.subplots(figsize=(5,4), subplot_kw={"projection": "polar"})

    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_thetamin(-90)
    ax1.set_thetamax(90)
    ax1.set_ylim([0, r_max])
    ax1.set_thetagrids(np.arange(-90, 91, 30))
    ax1.set_rticks([0, 15, 30, 45])
    ax1.set_yticklabels(["-45", "-30", "-15", "0"])
    ax1.grid(alpha=0.25)

    ax1.plot(
        angles,
        beam_r_learned_final,
        color="#d62728",
        linewidth=1.0,
        label=f"Learned ({learned_final_sinr_db:.2f} dB)",
    )
    ax1.plot(
        angles,
        beam_r_opt,
        color="#0E8E36",
        linewidth=1.0,
        linestyle="-.",
        label=f"Optimal ({optimal_final_sinr_db:.2f} dB)",
    )

    if np.isfinite(true_bd_angle_deg) and -90.0 <= true_bd_angle_deg <= 90.0:
        theta_bd = np.deg2rad(true_bd_angle_deg)
        ax1.plot([theta_bd, theta_bd], [0, r_max], color="#036221", linestyle="--", linewidth=1.8, label="BD")

    for s_idx, s_ang in enumerate(scatter_angles):
        if -90.0 <= s_ang <= 90.0:
            theta_s = np.deg2rad(s_ang)
            ax1.plot(
                [theta_s, theta_s],
                [0, r_max],
                color="#696969",
                linestyle=":",
                linewidth=1.8,
                label="Scatterers" if s_idx == 0 else None,
            )

    ax1.set_ylabel("Joint gain (dB)", labelpad=16)
    # ax1.set_title(f"Sample {sample_idx + 1}: Learned vs Optimal Beam")
    ax1.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.24, 1.12))
    plt.tight_layout()
    plt.savefig('figs/beam_pattern_compares.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()
