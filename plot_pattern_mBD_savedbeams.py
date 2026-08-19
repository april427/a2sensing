#!/usr/bin/env python3
"""
Plot beam pattern evolution over tau for the multi-BD scene of mono_1lstm_mBD.

The scene has ``num_users`` backscatter devices (one randomly woken up, the
others idle) plus ``num_scatters`` passive scatterers. The script reads
``v_list_test`` / ``w_list_test`` from a saved TEST .mat file produced by
``mono_1lstm_mBD.py`` and visualizes how the joint Tx-Rx beam pattern is
steered towards the *active* BD while the idle BDs and the scatterers are
progressively suppressed. No neural network is restored or run.

Channels are not stored in the .mat file, only the geometry. Two rebuild modes
are supported:

* ``exact replay`` (default): the test batch is regenerated with the same
  seed/order used by ``mono_1lstm_mBD.py`` (``seed + 10000``), which restores
  the exact channel realizations, including the random NLOS terms. The replay
  is accepted only if the regenerated BD locations match the saved ones.
* ``per-sample rebuild`` (fallback): channels are regenerated from the saved
  BD/scatterer locations only, so the small random NLOS components differ from
  those seen during testing.

Figures produced per selected sample:
  1. polar grid of the beam pattern at every sensing step + learned/optimal panel
  2. learned-final vs optimal polar comparison
  3. per-BD SINR (active vs idle) and scatterer leakage versus sensing step
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.linalg import eigh

from parse_args import parse_args
from channel_functions import generate_mimo_channel, generate_location_mimo

args = parse_args()

#####################################################
# Configuration
#####################################################
RESULT_DIR = "Mo_1LSTM_3BD_results"
FIG_DIR = "figs"

NUM_USERS = 3          # BDs in the scene (1 active + 2 idle)
NUM_SCATTERS = 5       # passive scatterers
TAU = 10
SNR_DB = 10
N_SYMBOLS = getattr(args, "N_symbols", 1)

# Test-set replay settings, must mirror mono_1lstm_mBD.py
TEST_SEED = getattr(args, "seed", 42) + 10000
TEST_SIZE = 800
EXACT_REPLAY = True

REF_DIS_LAMBDA = 166.67   # ref_dis = Wavelength * 166.67 in mono_1lstm_mBD.py
NOISE_VAR = 1.0           # 2 * noiseSTD_per_dim**2 with noiseSTD_per_dim = sqrt(0.5)
NUM_RESTARTS = 10         # random restarts of the optimal beamformer

SAMPLES_TO_PLOT = 1       # number of test samples to visualize
SAMPLE_INDICES = [165]    # e.g. [12, 57]; None picks at random
SELECTION_SEED = None     # seed of the random sample pick; None = different every run
SAVE_FIGS = False

DB_FLOOR = -45.0
R_MAX = 46.0

COLOR_LEARNED = "#d62728"
COLOR_OPTIMAL = "#2077b4"
COLOR_ACTIVE_BD = "#00c68dff"
COLOR_IDLE_BD = "#22493eff"
COLOR_PRED_BD = "#1f77b4"
COLOR_SCATTER = "#573F3F"


#####################################################
# Geometry / pattern helpers
#####################################################
def steering_vector_ula(n_ant, angle_rad):
    """ULA steering vector, matching generate_upa_steering_vector with N_v = 1."""
    idx = np.arange(n_ant)
    return np.exp(1j * np.pi * idx * np.sin(angle_rad))[:, np.newaxis]


def joint_beam_pattern_gains(v_vec, w_vec, angles):
    """|v^H a(theta) a(theta)^T w|^2, the monostatic BD channel convention.

    ``generate_mimo_channel`` builds the BD path as conj(a_rx) @ a_tx^T with
    a_rx taken towards the co-located receiver, i.e. a(theta + pi) = conj(a),
    so the round-trip response reduces to a(theta) a(theta)^T.
    """
    v = np.asarray(v_vec).reshape(-1, 1)
    w = np.asarray(w_vec).reshape(-1, 1)
    gains = np.zeros_like(angles, dtype=np.float64)
    for i, angle in enumerate(angles):
        a_rx = steering_vector_ula(v.shape[0], angle)
        a_tx = steering_vector_ula(w.shape[0], angle)
        gains[i] = np.abs((np.conj(v).T @ a_rx).item() * (a_tx.T @ w).item()) ** 2
    return gains


def gains_to_db(gains, reference_gain):
    return 10.0 * np.log10(gains / (reference_gain + 1e-12) + 1e-12)


def fold_ula_azimuth(angle_rad, sector_start=-np.pi / 2):
    """Fold azimuth into the ULA-unambiguous sector [-pi/2, pi/2)."""
    return np.mod(angle_rad - sector_start, np.pi) + sector_start


def bearings_deg(points, origin):
    """Scene bearing (deg) of each row of ``points`` seen from ``origin``."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.size == 0:
        return np.zeros(0)
    az = np.arctan2(points[:, 1] - origin[1], points[:, 0] - origin[0])
    return np.degrees(fold_ula_azimuth(az))


#####################################################
# SINR helpers (mirror the TF graph of mono_1lstm_mBD.py)
#####################################################
def per_device_metrics(v_vec, w_vec, h_b_all, h_int, p_tx, noise_var=NOISE_VAR):
    """Return (sinr_per_device, sig_per_device, interference) in linear scale.

    ``sinr_BD_per_device = P|v^H H_b[u] w|^2 / (P|v^H (H_d + H_r) w|^2 + N0)``
    exactly as in the training graph; the active-BD SINR is the entry selected
    by the woken-up device index.
    """
    v = np.asarray(v_vec).reshape(-1, 1)
    w = np.asarray(w_vec).reshape(-1, 1)
    sig = np.array(
        [p_tx * np.abs(np.conj(v).T @ h_b @ w).item() ** 2 for h_b in h_b_all],
        dtype=np.float64,
    )
    inter = p_tx * np.abs(np.conj(v).T @ h_int @ w).item() ** 2
    return sig / (inter + noise_var), sig, inter


def scatter_sinr(v_vec, w_vec, h_r, h_d, h_b_all, p_tx, noise_var=NOISE_VAR):
    """Scatterer-directed SINR, with BD + direct paths acting as interference."""
    v = np.asarray(v_vec).reshape(-1, 1)
    w = np.asarray(w_vec).reshape(-1, 1)
    sig = p_tx * np.abs(np.conj(v).T @ h_r @ w).item() ** 2
    h_int = h_d + np.sum(h_b_all, axis=0)
    inter = p_tx * np.abs(np.conj(v).T @ h_int @ w).item() ** 2
    return sig / (inter + noise_var)


def to_db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-12))


def compute_optimal_beamformers(h_b_active, h_int, p_tx, noise_var=NOISE_VAR,
                                num_restarts=NUM_RESTARTS):
    """Alternating generalized-eigenvector solution for the active-BD SINR.

    Same routine as ``compute_optimal_beamformers_batch_parallel`` in
    mono_1lstm_mBD.py, specialized to one sample and one desired BD channel.
    """
    n_rx, n_tx = h_b_active.shape
    best_sinr = -np.inf
    v_best = np.zeros((n_rx, 1), dtype=np.complex64)
    w_best = np.zeros((n_tx, 1), dtype=np.complex64)

    for _ in range(num_restarts):
        try:
            w = np.random.randn(n_tx) + 1j * np.random.randn(n_tx)
            w /= np.linalg.norm(w)
            for _ in range(20):
                desired_rx = h_b_active @ w
                r_sig_rx = p_tx * np.outer(desired_rx, desired_rx.conj())
                int_rx = h_int @ w
                r_int_rx = p_tx * np.outer(int_rx, int_rx.conj()) + noise_var * np.eye(n_rx)
                vals, vecs = eigh(r_sig_rx, r_int_rx, check_finite=False)
                v = vecs[:, np.argmax(np.real(vals))]
                v /= np.linalg.norm(v)

                desired_tx = h_b_active.conj().T @ v
                r_sig_tx = p_tx * np.outer(desired_tx, desired_tx.conj())
                int_tx = h_int.conj().T @ v
                r_int_tx = p_tx * np.outer(int_tx, int_tx.conj()) + noise_var * np.eye(n_tx)
                vals, vecs = eigh(r_sig_tx, r_int_tx, check_finite=False)
                w = vecs[:, np.argmax(np.real(vals))]
                w /= np.linalg.norm(w)
        except Exception:
            continue

        desired = p_tx * abs(np.vdot(v, h_b_active @ w)) ** 2
        interference = p_tx * abs(np.vdot(v, h_int @ w)) ** 2 + noise_var
        if desired / interference > best_sinr:
            best_sinr = desired / interference
            v_best[:, 0] = v
            w_best[:, 0] = w

    if not np.isfinite(best_sinr):
        u, _, vh = np.linalg.svd(h_b_active)
        v_best[:, 0] = u[:, 0]
        w_best[:, 0] = vh[0, :].conj()

    return v_best.astype(np.complex64), w_best.astype(np.complex64)


#####################################################
# Saved-data helpers
#####################################################
def pick_test_mat_file(result_dir, n_ant, tau, snr_db, n_symbols, n_scatters):
    """Pick the TEST .mat file by exact name, else the newest TEST file."""
    exact = os.path.join(
        result_dir,
        f"TEST_sinr_N_{n_ant}_{n_ant}_tau_{tau}_snr_{int(snr_db)}_K_{n_symbols}_Nsca_{n_scatters}.mat",
    )
    if os.path.isfile(exact):
        return exact

    candidates = [
        os.path.join(result_dir, f)
        for f in os.listdir(result_dir)
        if f.startswith(("TEST_sinr_", "TEST_joint_sinr_id_")) and f.endswith(".mat")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No TEST_sinr*.mat found in {result_dir}. Run the testing stage of "
            "mono_1lstm_mBD.py first."
        )
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def stack_locations(raw, num_samples, num_points):
    """Return saved locations as (num_samples, num_points, 3)."""
    arr = np.asarray(raw)
    if arr.dtype == object:
        entries = []
        for elem in arr.ravel():
            val = np.squeeze(np.asarray(elem, dtype=np.float64)) if elem is not None else None
            entries.append(val)
        if len(entries) == 1 and entries[0] is not None and entries[0].ndim == 3:
            arr = entries[0]
        else:
            arr = np.array(
                [
                    np.zeros((num_points, 3)) if e is None else np.reshape(e, (-1, 3))
                    for e in entries
                ],
                dtype=np.float64,
            )
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((num_samples, 0, 3))
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.shape[0] == 3:
        arr = arr.reshape(1, 1, 3)
    elif arr.ndim == 2 and arr.shape[-1] == 3:
        arr = arr[:, np.newaxis, :] if arr.shape[0] == num_samples else arr[np.newaxis, ...]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected saved location shape {np.asarray(raw).shape}")
    return arr


def as_int_vector(raw, num_samples, fill=0):
    vec = np.ravel(np.asarray(raw)).astype(np.int64)
    if vec.size == 0:
        return np.full(num_samples, fill, dtype=np.int64)
    if vec.size == 1:
        return np.full(num_samples, vec[0], dtype=np.int64)
    if vec.size < num_samples:
        vec = np.pad(vec, (0, num_samples - vec.size), constant_values=fill)
    return vec[:num_samples]


def replay_test_channels(num_samples, loc_tx, loc_rx, num_users, n_scatters,
                         n_tx, n_rx, rician, seed):
    """Regenerate the test batch in the exact order used by mono_1lstm_mBD.py.

    ``generate_location_mimo`` / ``generate_mimo_channel`` draw from the global
    NumPy RNG, so the replay has to seed it. The previous global state is saved
    and restored, otherwise everything drawn afterwards (in particular the
    random sample choice below) would be pinned to this seed as well.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        h_d_all, h_b_all, h_r_all, bd_all, sca_all = [], [], [], [], []
        for _ in range(num_samples):
            bd_loc = generate_location_mimo(num_users, 'u')
            scatter_loc = generate_location_mimo(n_scatters, 's')
            _, h_d, h_r, h_b = generate_mimo_channel(
                loc_tx, loc_rx, scatter_loc, bd_loc,
                N_tx_h=n_tx, N_tx_v=1, N_rx_h=n_rx, N_rx_v=1,
                Rician_factor=rician,
            )
            h_d_all.append(h_d)
            h_b_all.append(np.atleast_3d(h_b).reshape(num_users, n_rx, n_tx))
            h_r_all.append(h_r)
            bd_all.append(bd_loc)
            sca_all.append(scatter_loc)
    finally:
        np.random.set_state(rng_state)
    return (np.array(h_d_all), np.array(h_b_all), np.array(h_r_all),
            np.array(bd_all), np.array(sca_all))


def rebuild_sample_channels(bd_loc, scatter_loc, loc_tx, loc_rx, n_ant,
                            wavelength, rician):
    """Regenerate one sample's channels from its saved geometry."""
    scatter_in = None if np.asarray(scatter_loc).size == 0 else np.asarray(scatter_loc)
    _, h_d, h_r, h_b = generate_mimo_channel(
        loc_tx, loc_rx, scatter_in, np.atleast_2d(bd_loc),
        N_tx_h=n_ant, N_tx_v=1, N_rx_h=n_ant, N_rx_v=1,
        wavelength=wavelength, Rician_factor=rician,
    )
    return h_d, h_r, np.asarray(h_b).reshape(-1, n_ant, n_ant)


#####################################################
# Plot helpers
#####################################################
def setup_polar_axis(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([0, R_MAX])
    # Angular axis labelled in radians: -pi/2 ... pi/2.
    ax.set_thetagrids(
        [-90, -45, 0, 45, 90],
        labels=[r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"],
    )
    ax.set_rticks([0, 15, 30, 45])
    ax.set_yticklabels(["-45", "-30", "-15", "0"])
    ax.tick_params(axis="x", labelsize=14)   # angular (radian) labels
    ax.tick_params(axis="y", labelsize=11)    # radial (gain) labels
    ax.grid(alpha=0.25)


def mark_scene(ax, bd_angles_deg, scatter_angles_deg, active_idx, predicted_idx,
               annotate_labels=True, show_predicted=True, label_predicted=None):
    """Draw the active BD, the idle BDs, the inferred BD and the scatterers.

    The BD inferred by the LSTM classifier is marked with a star at the outer
    radius, so a correct decision shows up as a star sitting on the active-BD
    line and a wrong one as a star on an idle-BD line. The decision is only
    available once all sensing steps are done, so ``show_predicted`` allows
    hiding it on the intermediate panels.
    """
    correct = predicted_idx == active_idx
    if label_predicted is None:
        label_predicted = annotate_labels
    idle_labelled = False
    for device_idx, angle_deg in enumerate(bd_angles_deg):
        if not np.isfinite(angle_deg):
            continue
        theta = np.deg2rad(angle_deg)
        if device_idx == active_idx:
            ax.plot([theta, theta], [0, R_MAX], color=COLOR_ACTIVE_BD,
                    linestyle="--", linewidth=2.0, zorder=3,
                    label=f"Active BD {device_idx + 1}" if annotate_labels else None)
        else:
            label = None
            if annotate_labels and not idle_labelled:
                label = "Idle BDs"
                idle_labelled = True
            ax.plot([theta, theta], [0, R_MAX], color=COLOR_IDLE_BD,
                    linestyle="--", linewidth=1.4, alpha=0.9, zorder=2, label=label)

    if (show_predicted and predicted_idx is not None
            and 0 <= predicted_idx < len(bd_angles_deg)):
        theta = np.deg2rad(bd_angles_deg[predicted_idx])
        if not correct:
            # Highlight the whole wrong bearing, not just its tip.
            ax.plot([theta, theta], [0, R_MAX], color=COLOR_PRED_BD,
                    linestyle="-.", linewidth=1.8, zorder=4)
        ax.plot([theta], [R_MAX * 0.94], linestyle="none", marker="*",
                markersize=13, markeredgewidth=0.6,
                color=COLOR_PRED_BD if not correct else COLOR_ACTIVE_BD,
                markeredgecolor="black", zorder=5,
                label=(
                    f"Inferred BD {predicted_idx + 1} "
                    f"({'correct' if correct else 'wrong'})"
                    if label_predicted else None
                ))

    for s_idx, angle_deg in enumerate(scatter_angles_deg):
        if not -90.0 <= angle_deg <= 90.0:
            continue
        theta = np.deg2rad(angle_deg)
        ax.plot([theta, theta], [0, R_MAX], color=COLOR_SCATTER,
                linestyle=":", linewidth=1.8, zorder=1,
                label=("Scatterers" if (annotate_labels and s_idx == 0) else None))



#####################################################
# Load the saved test results
#####################################################
n_ant = args.N_ris
snr_db = SNR_DB
tau = TAU
num_users = NUM_USERS
num_scatters = NUM_SCATTERS
rician_factor = args.rician_factor
wavelength = 3e8 / args.fc

if not os.path.isdir(RESULT_DIR):
    raise FileNotFoundError(f"Result directory not found: {RESULT_DIR}")

mat_file = pick_test_mat_file(RESULT_DIR, n_ant, tau, snr_db, N_SYMBOLS, num_scatters)
data = sio.loadmat(mat_file)
print(f"Loaded TEST data: {mat_file}")

# Saved experiment parameters take precedence over the command-line defaults.
n_ant = int(np.squeeze(data.get("N_tx", n_ant)))
n_rx_saved = int(np.squeeze(data.get("N_rx", n_ant)))
tau = int(np.squeeze(data.get("tau", tau)))
snr_db = float(np.ravel(data.get("snr_const", [snr_db]))[0])
num_users = int(np.squeeze(data.get("num_users", num_users)))
num_scatters = int(np.squeeze(data.get("num_scatters", num_scatters)))
rician_factor = float(np.squeeze(data.get("rician_factor", rician_factor)))
wavelength = float(np.squeeze(data.get("wavelength", wavelength)))
location_tx = np.squeeze(data.get("location_tx", np.zeros(3))).astype(np.float64)
location_rx = np.squeeze(data.get("location_rx", np.zeros(3))).astype(np.float64)

ref_dis = wavelength * REF_DIS_LAMBDA
p_tx = 10 ** (snr_db / 10) / (wavelength ** 4 / (4 * np.pi * ref_dis) ** 4) / n_ant / n_rx_saved

if "v_list_test" not in data or "w_list_test" not in data:
    raise KeyError("Missing v_list_test or w_list_test in the TEST data file")

v_list = np.squeeze(np.asarray(data["v_list_test"]))
w_list = np.squeeze(np.asarray(data["w_list_test"]))
if v_list.ndim != 3 or w_list.ndim != 3 or v_list.shape != w_list.shape:
    raise ValueError(
        "Expected (num_steps, num_samples, N) beam tensors after squeeze, got "
        f"v={v_list.shape}, w={w_list.shape}"
    )

num_steps, num_samples, _ = v_list.shape
print(f"Beam tensor: steps={num_steps} (tau={tau} + final), samples={num_samples}, "
      f"antennas={n_ant}, BDs={num_users}, scatterers={num_scatters}")

bd_locations = stack_locations(data.get("BD_location", np.array([])), num_samples, num_users)
scatter_locations = stack_locations(data.get("Scatter_location", np.array([])),
                                    num_samples, num_scatters)
if bd_locations.shape[1] != num_users:
    print(f"Warning: saved BD locations hold {bd_locations.shape[1]} devices, "
          f"expected {num_users}")
    num_users = bd_locations.shape[1]

active_users = np.clip(as_int_vector(data.get("active_user", []), num_samples), 0, num_users - 1)
predicted_users = np.clip(as_int_vector(data.get("predicted_user", []), num_samples), 0, num_users - 1)
id_accuracy = float(np.squeeze(data.get("identification_accuracy", np.nan)))
sinr_saved = np.ravel(np.asarray(data.get("sinr_learned", np.full(num_samples, np.nan)), dtype=np.float64))
sinr_opt_saved = np.ravel(np.asarray(data.get("sinr_optimal", np.full(num_samples, np.nan)), dtype=np.float64))
print(f"Saved test ID accuracy: {100 * id_accuracy:.2f}%"
      if np.isfinite(id_accuracy) else "Saved test ID accuracy: n/a")

#####################################################
# Rebuild the channels behind the saved beams
#####################################################
replayed = None
if EXACT_REPLAY:
    replay_size = max(num_samples, TEST_SIZE)
    print(f"Replaying {replay_size} test channels with seed {TEST_SEED} ...")
    h_d_rep, h_b_rep, h_r_rep, bd_rep, sca_rep = replay_test_channels(
        replay_size, location_tx, location_rx, num_users, num_scatters,
        n_ant, n_rx_saved, rician_factor, TEST_SEED
    )
    match_err = np.max(np.abs(bd_rep[:num_samples] - bd_locations[:num_samples]))
    if match_err < 1e-6 * max(1.0, np.max(np.abs(bd_locations))):
        replayed = (h_d_rep, h_b_rep, h_r_rep)
        print("Exact channel replay verified against the saved BD locations.")
    else:
        print(f"Replay mismatch (max BD position error {match_err:.3e} m); "
              "falling back to per-sample rebuild from saved geometry.")

#####################################################
# Select samples and plot
#####################################################
if SAMPLE_INDICES is not None:
    show_indices = np.asarray(SAMPLE_INDICES, dtype=int)
else:
    # Own RNG stream: with SELECTION_SEED = None it is seeded from OS entropy,
    # so a new scene is drawn on every run regardless of any global seeding.
    selector = np.random.RandomState(SELECTION_SEED)
    show_indices = selector.choice(num_samples, size=min(SAMPLES_TO_PLOT, num_samples),
                                   replace=False)
print(f"Plotting sample(s): {show_indices.tolist()}")

angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)

for sample_idx in show_indices:
    sample_idx = int(sample_idx)
    bd_all = bd_locations[sample_idx]
    sca_all = scatter_locations[sample_idx]
    active_idx = int(active_users[sample_idx])
    predicted_idx = int(predicted_users[sample_idx])

    bd_angles_deg = bearings_deg(bd_all, location_rx)
    scatter_angles_deg = bearings_deg(sca_all, location_rx)

    if replayed is not None:
        h_d = replayed[0][sample_idx]
        h_b_devices = replayed[1][sample_idx]
        h_r = replayed[2][sample_idx]
    else:
        h_d, h_r, h_b_devices = rebuild_sample_channels(
            bd_all, sca_all, location_tx, location_rx, n_ant, wavelength, rician_factor
        )
    h_int = h_d + h_r

    # Per-step metrics
    sinr_per_device = np.zeros((num_steps, num_users))
    inter_steps = np.zeros(num_steps)
    scatter_steps = np.zeros(num_steps)
    for t in range(num_steps):
        sinr_dev, _, inter = per_device_metrics(
            v_list[t, sample_idx], w_list[t, sample_idx], h_b_devices, h_int, p_tx
        )
        sinr_per_device[t] = sinr_dev
        inter_steps[t] = inter
        scatter_steps[t] = scatter_sinr(
            v_list[t, sample_idx], w_list[t, sample_idx], h_r, h_d, h_b_devices, p_tx
        )
    sinr_active_db = to_db(sinr_per_device[:, active_idx])
    sinr_idle_db = to_db(sinr_per_device)
    inter_db = to_db(inter_steps)
    scatter_db = to_db(scatter_steps)
    learned_final_db = float(sinr_active_db[-1])

    # Beam patterns, normalized jointly over all steps
    beam_gains_steps = np.asarray([
        joint_beam_pattern_gains(v_list[t, sample_idx], w_list[t, sample_idx], angles)
        for t in range(num_steps)
    ])
    # Optimal benchmark for the active BD
    v_opt, w_opt = compute_optimal_beamformers(h_b_devices[active_idx], h_int, p_tx)
    sinr_opt_dev, _, _ = per_device_metrics(v_opt, w_opt, h_b_devices, h_int, p_tx)
    optimal_db = float(to_db(sinr_opt_dev[active_idx]))
    beam_opt_gains = joint_beam_pattern_gains(v_opt, w_opt, angles)

    # One common reference for every panel so the curves stay comparable
    ref_gain = max(np.max(beam_gains_steps), np.max(beam_opt_gains))
    beam_steps_db = gains_to_db(beam_gains_steps, ref_gain)
    beam_r_steps = np.clip(beam_steps_db, DB_FLOOR, 1.0) - DB_FLOOR
    beam_opt_db = gains_to_db(beam_opt_gains, ref_gain)
    beam_r_opt = np.clip(beam_opt_db, DB_FLOOR, 1.0) - DB_FLOOR

    id_correct = predicted_idx == active_idx
    id_flag_short = "correct" if id_correct else "wrong"
    id_flag = "correct" if id_correct else f"wrong (said BD {predicted_idx + 1})"
    print(f"\nSample {sample_idx}: active BD {active_idx + 1} at "
          f"{bd_angles_deg[active_idx]:.1f} deg, identification {id_flag}")
    print(f"  idle BDs at {np.round(np.delete(bd_angles_deg, active_idx), 1).tolist()} deg, "
          f"scatterers at {np.round(scatter_angles_deg, 1).tolist()} deg")
    print(f"  learned final SINR = {learned_final_db:.2f} dB, optimal = {optimal_db:.2f} dB, "
          f"gap = {optimal_db - learned_final_db:.2f} dB")

    #################################################
    # Figure 1: pattern evolution over the sensing steps
    #################################################
    n_plot = num_steps + 1
    n_cols = 4
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.0 * n_rows),
        subplot_kw={"projection": "polar"},
        gridspec_kw={"wspace": 0.18, "hspace": -0.1},
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for empty_idx in range(n_plot, len(axes)):
        axes[empty_idx].set_visible(False)

    for step_idx in range(num_steps):
        ax = axes[step_idx]
        setup_polar_axis(ax)
        ax.plot(angles, beam_r_steps[step_idx], color=COLOR_LEARNED,
                linewidth=1.0, label="Learned beam")
        mark_scene(ax, bd_angles_deg, scatter_angles_deg, active_idx, predicted_idx,
                   annotate_labels=False, show_predicted=False)
        step_label = f"t={step_idx + 1}" if step_idx < num_steps - 1 else "Final"
        ax.set_title(
            f"{step_label}, SINR={sinr_active_db[step_idx]:.2f} dB\n"
            f"Idle BD={np.max(np.delete(sinr_idle_db[step_idx], active_idx)):.2f} dB, "
            f"Intf={inter_db[step_idx]:.2f} dB",
            fontsize=10, pad=-19,
        )
        # ax.set_ylabel("Joint gain (dB)", labelpad=18)

    ax_cmp = axes[num_steps]
    setup_polar_axis(ax_cmp)
    ax_cmp.plot(angles, beam_r_steps[-1], color=COLOR_LEARNED, linewidth=1.0,
                label=f"Learned ({learned_final_db:.2f} dB)")
    ax_cmp.plot(angles, beam_r_opt, color=COLOR_OPTIMAL, linewidth=1.0, linestyle="-.",
                label=f"Optimal ({optimal_db:.2f} dB)")
    mark_scene(ax_cmp, bd_angles_deg, scatter_angles_deg, active_idx, predicted_idx,
               annotate_labels=True, label_predicted=True)
    ax_cmp.set_title(f"Comparison. Gap={optimal_db - learned_final_db:.2f} dB",
                    #  f"inferred BD {predicted_idx + 1} of {active_idx + 1} ({id_flag_short})",
                     fontsize=10, pad=-19)
    # Single legend for the whole figure, kept on the final panel and in one row.
    cmp_handles, cmp_labels = ax_cmp.get_legend_handles_labels()
    ax_cmp.legend(cmp_handles, cmp_labels, fontsize=9, loc="upper center",
                  bbox_to_anchor=(-0.4, 0.1), ncol=len(cmp_handles), frameon=True,
                  columnspacing=1.0, handlelength=1.6, handletextpad=0.4,
                  borderaxespad=0.0)

    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.06, top=0.94,
                        wspace=0.16, hspace=0.02)
    if True:
        fig.savefig(os.path.join(FIG_DIR, f"beam_pattern_evolving_{num_users}BD_s{sample_idx}.pdf"),
                    format="pdf", bbox_inches="tight")
    plt.show()

    #################################################
    # Figure 2: final learned vs optimal
    #################################################
    fig2, ax2 = plt.subplots(figsize=(5.2, 4.2), subplot_kw={"projection": "polar"})
    setup_polar_axis(ax2)
    ax2.plot(angles, beam_r_steps[-1], color=COLOR_LEARNED, linewidth=1.2,
             label=f"Learned ({learned_final_db:.2f} dB)")
    ax2.plot(angles, beam_r_opt, color=COLOR_OPTIMAL, linewidth=1.2, linestyle="-.",
             label=f"Optimal ({optimal_db:.2f} dB)")
    mark_scene(ax2, bd_angles_deg, scatter_angles_deg, active_idx, predicted_idx)
    ax2.set_title(f"Active BD {active_idx + 1}, inferred BD {predicted_idx + 1} "
                  f"({id_flag_short})", fontsize=10, pad=12)
    ax2.set_ylabel("Joint gain (dB)", labelpad=16)
    ax2.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig2.tight_layout()
    if SAVE_FIGS:
        fig2.savefig(os.path.join(FIG_DIR, f"beam_pattern_compare_{num_users}BD_s{sample_idx}.pdf"),
                     format="pdf", bbox_inches="tight")
    plt.show()

    #################################################
    # Figure 3: per-BD SINR and scatterer leakage vs sensing step
    #################################################
    fig3, ax3 = plt.subplots(figsize=(5.6, 3.8))
    steps_axis = np.arange(1, num_steps + 1)
    for device_idx in range(num_users):
        is_active = device_idx == active_idx
        ax3.plot(
            steps_axis, sinr_idle_db[:, device_idx],
            color=COLOR_ACTIVE_BD if is_active else COLOR_IDLE_BD,
            linestyle="-" if is_active else "--",
            marker="o" if is_active else "^",
            markersize=4, linewidth=1.8 if is_active else 1.2,
            label=(
                f"{'Active' if is_active else 'Idle'} BD {device_idx + 1} "
                f"({bd_angles_deg[device_idx]:.0f}$^\\circ$)"
                + (" [inferred]" if device_idx == predicted_idx else "")
            ),
        )
    ax3.plot(steps_axis, scatter_db, color=COLOR_SCATTER, linestyle=":", marker="s",
             markersize=3, linewidth=1.2, label="Scatterers")
    ax3.axhline(optimal_db, color=COLOR_OPTIMAL, linestyle="-.", linewidth=1.2,
                label=f"Optimal active ({optimal_db:.2f} dB)")
    ax3.set_xlabel("Sensing step $t$")
    ax3.set_ylabel("SINR (dB)")
    ax3.set_xticks(steps_axis)
    ax3.set_xticklabels([str(t) for t in steps_axis[:-1]] + ["Final"])
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               frameon=False)
    fig3.tight_layout()
    if SAVE_FIGS:
        fig3.savefig(os.path.join(FIG_DIR, f"sinr_per_BD_vs_step_{num_users}BD_s{sample_idx}.pdf"),
                     format="pdf", bbox_inches="tight")
    plt.show()
