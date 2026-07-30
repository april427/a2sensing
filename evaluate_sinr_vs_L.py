#!/usr/bin/env python3
"""Evaluate trained single-BD active-sensing models as a function of code length.

The checkpoints used two BD states per sensing step, with one noisy sample for
each state (K=1). During evaluation, the single BD uses the alternating code
[-1, +1, -1, +1, ...] of length L. The received chip observations are matched
filtered and normalized to reproduce the trained L=2 observation scale before
they are passed to the restored recurrent network.
"""

# %%
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hadamard
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import BatchNormalization, Dense

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

from channel_functions import generate_location_mimo, generate_mimo_channel
from parse_args import parse_args

args = parse_args()

# %%
#####################################################################################
#                                Test parameters                                    #
#####################################################################################

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

snr_value = 10
L_values = [2, 4, 8, 16, 32]
trained_L = 2
trained_K = 1

N_tx = args.N_ris
N_rx = args.N_ris
tau = 8
num_scatters = args.N_scatterers
Rician_factor = args.rician_factor
hidden_size = 128

test_size = 800
batch_size = 100
seed = 42

fc = args.fc
Wavelength = 3e8 / fc
ref_dis = 15 * Wavelength
noiseSTD_per_dim = np.sqrt(0.5)
noise_var = 2 * noiseSTD_per_dim**2

location_tx = np.array([0, 0, 0])
location_rx = np.array([0, 0, 0])

Pvec = (
    10 ** (snr_value / 10)
    / (Wavelength**4 / (4 * np.pi * ref_dis) ** 4)
    / (N_tx * N_rx)
)

np.random.seed(seed + 10000)


# %%
#####################################################################################
#                     Network definitions from trained models                      #
#####################################################################################

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, dims, name):
        super(MLPBlock, self).__init__()
        self.layers_list = []
        for ii in range(num_layers - 1):
            self.layers_list.append(
                Dense(
                    units=dims[ii],
                    activation="relu",
                    name=name + "_relu_" + str(ii),
                )
            )
            self.layers_list.append(BatchNormalization())
        self.layers_list.append(
            Dense(units=dims[-1], activation="linear", name=name + "_linear")
        )

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class LSTM_Cell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name):
        super(LSTM_Cell, self).__init__()
        self.layer_Ui = Dense(hidden_size, activation="linear", name="Ui" + name)
        self.layer_Wi = Dense(hidden_size, activation="linear", name="Wi" + name)
        self.layer_Uf = Dense(hidden_size, activation="linear", name="Uf" + name)
        self.layer_Wf = Dense(hidden_size, activation="linear", name="Wf" + name)
        self.layer_Uo = Dense(hidden_size, activation="linear", name="Uo" + name)
        self.layer_Wo = Dense(hidden_size, activation="linear", name="Wo" + name)
        self.layer_Uc = Dense(hidden_size, activation="linear", name="Uc" + name)
        self.layer_Wc = Dense(hidden_size, activation="linear", name="Wc" + name)

    def call(self, inputs, **kwargs):
        input_x, h_old, c_old = inputs
        i_t = tf.sigmoid(self.layer_Ui(input_x) + self.layer_Wi(h_old))
        f_t = tf.sigmoid(self.layer_Uf(input_x) + self.layer_Wf(h_old))
        o_t = tf.sigmoid(self.layer_Uo(input_x) + self.layer_Wo(h_old))
        c_t = tf.tanh(self.layer_Uc(input_x) + self.layer_Wc(h_old))
        c = i_t * c_t + f_t * c_old
        h_new = o_t * tf.tanh(c)
        return h_new, c


def hadamard_codebook(n, m):
    """Hadamard codebook used by the training scripts."""
    target_size = max(n, m)
    if target_size <= 1:
        had_size = 1
    elif target_size <= 2:
        had_size = 2
    else:
        had_size = 4
        while had_size < target_size:
            had_size *= 2

    return (hadamard(had_size)[:n, :m] / np.sqrt(n)).astype(np.complex64)


def alternating_bd_code(length):
    """Return a balanced [-1, +1, ...] code for one BD."""
    if length < 2 or length % 2:
        raise ValueError(
            f"L must be a positive even integer; received L={length}."
        )
    return np.where(np.arange(length) % 2 == 0, -1.0, 1.0).astype(
        np.float32
    )


for L in L_values:
    alternating_bd_code(L)


# %%
#####################################################################################
#                    Random locations and shared test channels                     #
#####################################################################################

H_d_test = []
H_b_test = []
H_r_test = []
set_location_user_test = []
BD_loc = []
Scatter_loc = []

print(f"Generating {test_size} random channel realizations...")

for sample_idx in range(test_size):
    bd_location = generate_location_mimo(1, "u")[0]
    scatter_location = generate_location_mimo(num_scatters, "s")

    _, H_d, H_r, H_b = generate_mimo_channel(
        location_tx,
        location_rx,
        scatter_location,
        bd_location,
        N_tx_h=N_tx,
        N_tx_v=1,
        N_rx_h=N_rx,
        N_rx_v=1,
        Rician_factor=Rician_factor,
    )

    d_bd_rx = np.linalg.norm(bd_location - location_rx)
    d_bd_tx = np.linalg.norm(bd_location - location_tx)
    azimuth_bd = np.arctan2(
        bd_location[1] - location_rx[1],
        bd_location[0] - location_rx[0],
    )

    H_d_test.append(H_d)
    H_b_test.append(H_b)
    H_r_test.append(H_r)
    set_location_user_test.append(
        np.array([azimuth_bd, d_bd_rx, d_bd_tx])[:, np.newaxis]
    )
    BD_loc.append(bd_location)
    Scatter_loc.append(scatter_location)

H_d_test = np.asarray(H_d_test, dtype=np.complex64)
H_b_test = np.asarray(H_b_test, dtype=np.complex64)
H_r_test = np.asarray(H_r_test, dtype=np.complex64)
set_location_user_test = np.asarray(set_location_user_test, dtype=np.float32)
BD_loc = np.asarray(BD_loc)
Scatter_loc = np.asarray(Scatter_loc)


# %%
#####################################################################################
#                       Shared nested noise for every L                           #
#####################################################################################

# The first L chips are used for each experiment, so every longer-code
# experiment contains exactly the same noise realization as every shorter one
# over their shared prefix.
max_L = max(L_values)
noise_rng = np.random.RandomState(seed + 20000)
noise_shape = (test_size, tau, max_L, N_rx)
noise_bank = np.empty(noise_shape, dtype=np.complex64)
noise_bank.real = noise_rng.normal(
    0.0, noiseSTD_per_dim, noise_shape
).astype(np.float32)
noise_bank.imag = noise_rng.normal(
    0.0, noiseSTD_per_dim, noise_shape
).astype(np.float32)


# %%
#####################################################################################
#                   Restore and test the two trained models                        #
#####################################################################################

model_settings = [
    (
        "two_lstm",
        "Two LSTM (trained L=2)",
        False,
        ROOT / "Mo_mimo_sinr_modelsave",
    ),
    (
        "one_lstm",
        "One fused LSTM (trained L=2)",
        True,
        ROOT / "Mo_mimo_sinr_modelsave_one_lstm",
    ),
]

sinr_results = {}

for model_key, model_label, fused_lstm, model_dir in model_settings:
    tf.keras.backend.clear_session()
    model_graph = tf.Graph()

    with model_graph.as_default():
        tf.set_random_seed(seed)

        loc_input = tf.placeholder(
            tf.float32, shape=(None, 3, 1), name="loc_input"
        )
        H_d_placeholder = tf.placeholder(
            tf.complex64, shape=(None, N_rx, N_tx), name="H_d"
        )
        H_b_placeholder = tf.placeholder(
            tf.complex64, shape=(None, N_rx, N_tx), name="H_b"
        )
        H_r_placeholder = tf.placeholder(
            tf.complex64, shape=(None, N_rx, N_tx), name="H_r"
        )
        noise_placeholder = tf.placeholder(
            tf.complex64,
            shape=(None, tau, max_L, N_rx),
            name="noise_bank",
        )
        L_placeholder = tf.placeholder(
            tf.int32, shape=(), name="evaluation_code_length"
        )
        P = tf.placeholder_with_default(
            tf.constant(1.0, dtype=tf.float32),
            shape=(),
            name="tx_power",
        )

        alternating_code_tf = tf.constant(
            alternating_bd_code(max_L),
            dtype=tf.float32,
            name="alternating_BD_code",
        )

        with tf.name_scope("active_sensing_agent"):
            if fused_lstm:
                LSTM1 = LSTM_Cell(hidden_size, name="LSTM_fused")
                LSTM2 = None
            else:
                LSTM1 = LSTM_Cell(hidden_size, name="LSTM_1")
                LSTM2 = LSTM_Cell(hidden_size, name="LSTM_2")

            mlp_ris_tx = MLPBlock(
                3,
                [2 * hidden_size, 2 * hidden_size, 2 * N_tx],
                name="RIS_transmitter",
            )
            mlp_rx_rx = MLPBlock(
                3,
                [2 * hidden_size, 2 * hidden_size, 2 * N_rx],
                name="Receiver_receiver",
            )

            snr = P * tf.ones(
                shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32
            )
            snr_normal = 10 * tf.log(snr) / np.log(10)

            Y1 = tf.zeros(
                [tf.shape(loc_input)[0], N_rx], dtype=tf.complex64
            )
            Y2 = tf.zeros(
                [tf.shape(loc_input)[0], N_rx], dtype=tf.complex64
            )
            graph_batch_size = tf.shape(loc_input)[0]

            for t in range(tau):
                if t == 0:
                    h_old = tf.zeros([graph_batch_size, hidden_size])
                    c_old = tf.zeros([graph_batch_size, hidden_size])

                    if not fused_lstm:
                        h_old2 = tf.zeros([graph_batch_size, hidden_size])
                        c_old2 = tf.zeros([graph_batch_size, hidden_size])

                    w_init_real = tf.get_variable("w_init_real", shape=(1, N_tx, 1))
                    w_init_imag = tf.get_variable("w_init_imag", shape=(1, N_tx, 1))
                    w1 = tf.complex(w_init_real, w_init_imag)
                    w1 = w1 / tf.cast(tf.norm(w1, axis=1, keepdims=True), tf.complex64)

                    v_init_real = tf.get_variable("v_init_real", shape=(1, N_rx, 1))
                    v_init_imag = tf.get_variable("v_init_imag", shape=(1, N_rx, 1))
                    v1 = tf.complex(v_init_real, v_init_imag)
                    v1 = v1 / tf.cast(tf.norm(v1, axis=1, keepdims=True), tf.complex64)

                sqrt_P = tf.complex(tf.sqrt(P), 0.0)
                H_common = H_d_placeholder + H_r_placeholder
                y_common_clean = tf.squeeze(
                    sqrt_P * tf.matmul(H_common, w1), axis=2
                )
                y_bd_clean = tf.squeeze(
                    sqrt_P * tf.matmul(H_b_placeholder, w1), axis=2
                )

                # Generate L received chips using the known alternating code.
                code_L_real = alternating_code_tf[:L_placeholder]
                code_L = tf.cast(code_L_real, tf.complex64)
                received_chips = (
                    tf.expand_dims(y_common_clean, axis=1)
                    + tf.expand_dims(y_bd_clean, axis=1)
                    * tf.reshape(code_L, [1, -1, 1])
                    + noise_placeholder[:, t, :L_placeholder, :]
                )

                # Preserve the L=2 signal scale seen during training. Longer
                # codes therefore improve only the matched-filter noise level.
                matched_filter_scale = (
                    tf.cast(trained_L, tf.complex64)
                    / tf.cast(L_placeholder, tf.complex64)
                )
                matched_bd = matched_filter_scale * tf.reduce_sum(
                    received_chips * tf.reshape(code_L, [1, -1, 1]),
                    axis=1,
                )
                matched_common = matched_filter_scale * tf.reduce_sum(
                    received_chips, axis=1
                )

                Y1 = Y1 + matched_bd
                Y2 = Y2 + matched_common

                Y1_after = tf.matmul(
                    tf.linalg.adjoint(v1),
                    tf.reshape(Y1, [-1, N_rx, 1]),
                )[:, :, 0]
                Y2_after = tf.matmul(
                    tf.linalg.adjoint(v1),
                    tf.reshape(Y2, [-1, N_rx, 1]),
                )[:, :, 0]

                y_real1 = tf.concat(
                    [
                        tf.real(Y1),
                        tf.imag(Y1),
                        tf.real(Y1_after),
                        tf.imag(Y1_after),
                    ],
                    axis=1,
                )
                y_real2 = tf.concat(
                    [
                        tf.real(Y2),
                        tf.imag(Y2),
                        tf.real(Y2_after),
                        tf.imag(Y2_after),
                    ],
                    axis=1,
                )

                if fused_lstm:
                    fused_observation = tf.concat(
                        [y_real1, y_real2, snr_normal], axis=1
                    )
                    h_old, c_old = LSTM1(
                        (fused_observation, h_old, c_old)
                    )
                    state_for_beams = tf.concat(
                        [h_old, c_old], axis=1
                    )
                else:
                    h_old, c_old = LSTM1(
                        (
                            tf.concat([y_real1, snr_normal], axis=1),
                            h_old,
                            c_old,
                        )
                    )
                    h_old2, c_old2 = LSTM2(
                        (
                            tf.concat([y_real2, snr_normal], axis=1),
                            h_old2,
                            c_old2,
                        )
                    )
                    state_for_beams = tf.concat(
                        [h_old, h_old2], axis=1
                    )

                w_her = mlp_ris_tx(state_for_beams)
                w_her = w_her / (
                    tf.reshape(tf.norm(w_her, axis=1), [-1, 1]) + 1e-8
                )
                w1 = tf.complex(
                    w_her[:, :N_tx], w_her[:, N_tx : 2 * N_tx]
                )
                w1 = tf.reshape(w1, [-1, N_tx, 1])

                v_her = mlp_rx_rx(state_for_beams)
                v_her = v_her / (
                    tf.reshape(tf.norm(v_her, axis=1), [-1, 1]) + 1e-8
                )
                v1 = tf.complex(
                    v_her[:, :N_rx], v_her[:, N_rx : 2 * N_rx]
                )
                v1 = tf.reshape(v1, [-1, N_rx, 1])

            MLP_bf_w = MLPBlock(
                3,
                [2 * hidden_size, 2 * hidden_size, 2 * N_tx],
                name="MLP_bf_w",
            )
            MLP_bf_v = MLPBlock(
                3,
                [2 * hidden_size, 2 * hidden_size, 2 * N_rx],
                name="MLP_bf_v",
            )

            if fused_lstm:
                final_state = tf.concat([h_old, c_old], axis=1)
            else:
                final_state = tf.concat([c_old, c_old2], axis=1)

            w_final_real = MLP_bf_w(final_state)
            w_final_real = w_final_real / (
                tf.reshape(tf.norm(w_final_real, axis=1), [-1, 1])
                + 1e-8
            )
            w_final = tf.complex(
                w_final_real[:, :N_tx],
                w_final_real[:, N_tx : 2 * N_tx],
            )
            w_final = tf.reshape(w_final, [-1, N_tx, 1])

            v_final_real = MLP_bf_v(final_state)
            v_final_real = v_final_real / (
                tf.reshape(tf.norm(v_final_real, axis=1), [-1, 1])
                + 1e-8
            )
            v_final = tf.complex(
                v_final_real[:, :N_rx],
                v_final_real[:, N_rx : 2 * N_rx],
            )
            v_final = tf.reshape(v_final, [-1, N_rx, 1])

        H_interference = H_d_placeholder + H_r_placeholder
        desired = tf.matmul(
            tf.linalg.adjoint(v_final),
            tf.matmul(H_b_placeholder, w_final),
        )
        interference = tf.matmul(
            tf.linalg.adjoint(v_final),
            tf.matmul(H_interference, w_final),
        )
        sinr_model = (
            tf.squeeze(tf.abs(desired) ** 2) * P
            / (
                tf.squeeze(tf.abs(interference) ** 2) * P
                + noise_var
            )
        )

        # The checkpoints use legacy flattened Keras variable names.
        restore_map = {
            "w_init_real": w_init_real,
            "w_init_imag": w_init_imag,
            "v_init_real": v_init_real,
            "v_init_imag": v_init_imag,
        }

        lstm_restore_settings = [
            ("active_sensing_agent/lstm__cell", LSTM1)
        ]
        if not fused_lstm:
            lstm_restore_settings.append(
                ("active_sensing_agent/lstm__cell_1_1", LSTM2)
            )

        for checkpoint_prefix, lstm_cell in lstm_restore_settings:
            lstm_layers = [
                lstm_cell.layer_Ui,
                lstm_cell.layer_Wi,
                lstm_cell.layer_Uf,
                lstm_cell.layer_Wf,
                lstm_cell.layer_Uo,
                lstm_cell.layer_Wo,
                lstm_cell.layer_Uc,
                lstm_cell.layer_Wc,
            ]
            for layer_idx, layer in enumerate(lstm_layers):
                suffix = "" if layer_idx == 0 else f"_{layer_idx}"
                restore_map[
                    f"{checkpoint_prefix}/kernel{suffix}"
                ] = layer.kernel
                restore_map[
                    f"{checkpoint_prefix}/bias{suffix}"
                ] = layer.bias

        mlp_restore_settings = [
            ("active_sensing_agent/mlp_block", mlp_ris_tx),
            ("active_sensing_agent/mlp_block_1_1", mlp_rx_rx),
            ("active_sensing_agent/mlp_block_2_1", MLP_bf_w),
            ("active_sensing_agent/mlp_block_3_1", MLP_bf_v),
        ]

        for checkpoint_prefix, mlp_layer in mlp_restore_settings:
            dense_layers = [
                mlp_layer.layers_list[0],
                mlp_layer.layers_list[2],
                mlp_layer.layers_list[4],
            ]
            batch_norm_layers = [
                mlp_layer.layers_list[1],
                mlp_layer.layers_list[3],
            ]

            for layer_idx, layer in enumerate(dense_layers):
                suffix = "" if layer_idx == 0 else f"_{layer_idx}"
                restore_map[
                    f"{checkpoint_prefix}/kernel{suffix}"
                ] = layer.kernel
                restore_map[
                    f"{checkpoint_prefix}/bias{suffix}"
                ] = layer.bias

            for layer_idx, layer in enumerate(batch_norm_layers):
                suffix = "" if layer_idx == 0 else f"_{layer_idx}"
                restore_map[
                    f"{checkpoint_prefix}/gamma{suffix}"
                ] = layer.gamma
                restore_map[
                    f"{checkpoint_prefix}/beta{suffix}"
                ] = layer.beta
                restore_map[
                    f"{checkpoint_prefix}/moving_mean{suffix}"
                ] = layer.moving_mean
                restore_map[
                    f"{checkpoint_prefix}/moving_variance{suffix}"
                ] = layer.moving_variance

        checkpoint_name = (
            f"params_sinr_N_{N_tx}_{N_rx}_tau_{tau}"
            f"_snr_{snr_value}_K_{trained_K}_Nsc_{num_scatters}"
        )
        checkpoint_path = str(model_dir / checkpoint_name)

        if not Path(checkpoint_path + ".index").is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}.index"
            )

        checkpoint_variables = dict(
            tf.train.list_variables(checkpoint_path)
        )
        matched_variables = {}
        missing_trainable = []
        missing_nontrainable = []
        trainable_variables = tf.trainable_variables()

        for variable_name, graph_variable in restore_map.items():
            graph_shape = tuple(graph_variable.shape.as_list())
            checkpoint_shape = checkpoint_variables.get(variable_name)
            graph_tf_variable = (
                graph_variable.value
                if isinstance(graph_variable.dtype, str)
                else graph_variable
            )

            if (
                checkpoint_shape is not None
                and tuple(checkpoint_shape) == graph_shape
            ):
                matched_variables[variable_name] = graph_variable
            elif any(
                graph_tf_variable is trainable_variable
                for trainable_variable in trainable_variables
            ):
                missing_trainable.append(variable_name)
            else:
                missing_nontrainable.append(variable_name)

        if missing_trainable:
            raise ValueError(
                "Missing trained variables in checkpoint:\n"
                + "\n".join(missing_trainable)
            )

        # These checkpoints were written with legacy TF1 tensor names. Recent
        # Keras resource-variable wrappers make tf.train.Saver append
        # "VARIABLE_VALUE" to those names, so restore the raw checkpoint
        # tensors through explicit assignment placeholders instead.
        restore_placeholders = {}
        restore_assignments = {}
        with tf.name_scope("legacy_checkpoint_restore"):
            for restore_idx, (
                variable_name,
                graph_variable,
            ) in enumerate(matched_variables.items()):
                assign_target = (
                    graph_variable.value
                    if isinstance(graph_variable.dtype, str)
                    else graph_variable
                )
                restore_placeholder = tf.placeholder(
                    assign_target.dtype,
                    shape=assign_target.shape,
                    name=f"value_{restore_idx}",
                )
                restore_placeholders[variable_name] = restore_placeholder
                restore_assignments[variable_name] = tf.assign(
                    assign_target,
                    restore_placeholder,
                    name=f"assign_{restore_idx}",
                )

        init = tf.global_variables_initializer()
        graph_variable_count = len(tf.global_variables())

    print(
        f"\nRestoring {model_label}: "
        f"{len(matched_variables)}/{graph_variable_count} variables"
    )
    if missing_nontrainable:
        print(
            f"Warning: {len(missing_nontrainable)} BatchNorm moving "
            "statistics are absent from this checkpoint. "
            "Their default mean=0 and variance=1 are used."
        )

    sinr_for_model = np.empty(
        (len(L_values), test_size), dtype=np.float64
    )

    with tf.Session(graph=model_graph) as sess:
        sess.run(init)
        checkpoint_reader = tf.train.load_checkpoint(checkpoint_path)
        restore_feed = {
            restore_placeholders[variable_name]:
                checkpoint_reader.get_tensor(variable_name)
            for variable_name in matched_variables
        }
        sess.run(
            list(restore_assignments.values()),
            feed_dict=restore_feed,
        )

        for L_idx, L in enumerate(L_values):
            for batch_start in range(0, test_size, batch_size):
                batch_end = min(batch_start + batch_size, test_size)

                feed_dict = {
                    loc_input: set_location_user_test[
                        batch_start:batch_end
                    ],
                    H_d_placeholder: H_d_test[batch_start:batch_end],
                    H_b_placeholder: H_b_test[batch_start:batch_end],
                    H_r_placeholder: H_r_test[batch_start:batch_end],
                    noise_placeholder: noise_bank[
                        batch_start:batch_end
                    ],
                    L_placeholder: L,
                    P: Pvec,
                }

                sinr_for_model[
                    L_idx, batch_start:batch_end
                ] = sess.run(sinr_model, feed_dict=feed_dict)

            mean_sinr_db = 10 * np.log10(
                np.mean(sinr_for_model[L_idx]) + 1e-12
            )
            print(f"L={L:2d}: {mean_sinr_db:7.3f} dB")

    if not np.all(np.isfinite(sinr_for_model)):
        raise FloatingPointError(
            f"{model_label} produced a non-finite SINR."
        )

    sinr_results[model_key] = sinr_for_model


# %%
#####################################################################################
#                              Beam sweeping                                        #
#####################################################################################

pilot_codebook = hadamard_codebook(N_tx, max(N_tx, tau))
s_pilot = pilot_codebook[:, :tau]
sweep_codebook = hadamard_codebook(
    N_tx, max(N_tx, 2 * tau)
)[:, : 2 * tau]

ss_h = s_pilot @ np.conj(s_pilot).T
ss_h_reg = ss_h + 1e-1 * np.eye(N_tx)
s_pilot_pinv = np.conj(
    np.linalg.solve(ss_h_reg, s_pilot)
).T

sinr_sweeping = np.empty(
    (len(L_values), test_size), dtype=np.float64
)

for L_idx, L in enumerate(L_values):
    code_L = alternating_bd_code(L).astype(np.complex64)
    matched_filter_scale = trained_L / float(L)

    for batch_start in range(0, test_size, batch_size):
        batch_end = min(batch_start + batch_size, test_size)

        H_b_batch = H_b_test[batch_start:batch_end]
        H_I_batch = (
            H_d_test[batch_start:batch_end]
            + H_r_test[batch_start:batch_end]
        )
        noise_batch = noise_bank[batch_start:batch_end]

        Y_common_clean = np.sqrt(Pvec) * np.einsum(
            "bij,jt->bit", H_I_batch, s_pilot
        )
        Y_bd_clean = np.sqrt(Pvec) * np.einsum(
            "bij,jt->bit", H_b_batch, s_pilot
        )

        received_chips = (
            Y_common_clean.transpose(0, 2, 1)[:, :, np.newaxis, :]
            + Y_bd_clean.transpose(0, 2, 1)[:, :, np.newaxis, :]
            * code_L[np.newaxis, np.newaxis, :, np.newaxis]
            + noise_batch[:, :, :L, :]
        )

        Y_bd_matched = matched_filter_scale * np.einsum(
            "l,btlr->btr", code_L, received_chips, optimize=True
        )
        Y_common_matched = matched_filter_scale * np.sum(
            received_chips, axis=2
        )

        # Matched-filter signal terms are trained_L * sqrt(P) * H * w.
        H_b_hat = (
            np.matmul(Y_bd_matched.transpose(0, 2, 1), s_pilot_pinv)
            / (trained_L * np.sqrt(Pvec))
        )
        H_I_hat = (
            np.matmul(
                Y_common_matched.transpose(0, 2, 1),
                s_pilot_pinv,
            )
            / (trained_L * np.sqrt(Pvec))
        )

        diagonal_response = np.einsum(
            "ri,brt,ti->bi",
            np.conj(sweep_codebook),
            H_b_hat,
            sweep_codebook,
            optimize=True,
        )
        best_beam_idx = np.argmax(
            np.abs(diagonal_response) ** 2, axis=1
        )
        w_sweep = sweep_codebook[:, best_beam_idx].T

        H_I_w = np.einsum(
            "bij,bj->bi", H_I_hat, w_sweep
        )
        projection_coefficient = (
            np.einsum(
                "bi,bi->b", np.conj(H_I_w), w_sweep
            )
            / (np.sum(np.abs(H_I_w) ** 2, axis=1) + 1e-10)
        )
        v_sweep = (
            w_sweep
            - H_I_w * projection_coefficient[:, np.newaxis]
        )

        v_norm = np.linalg.norm(v_sweep, axis=1)
        invalid_v = v_norm < 1e-8
        v_sweep[invalid_v] = w_sweep[invalid_v]
        v_sweep = v_sweep / (
            np.linalg.norm(v_sweep, axis=1, keepdims=True) + 1e-10
        )

        desired = np.einsum(
            "bi,bij,bj->b",
            np.conj(v_sweep),
            H_b_batch,
            w_sweep,
            optimize=True,
        )
        interference = np.einsum(
            "bi,bij,bj->b",
            np.conj(v_sweep),
            H_I_batch,
            w_sweep,
            optimize=True,
        )

        sinr_sweeping[
            L_idx, batch_start:batch_end
        ] = (
            Pvec * np.abs(desired) ** 2
            / (Pvec * np.abs(interference) ** 2 + noise_var)
        )

    mean_sinr_db = 10 * np.log10(
        np.mean(sinr_sweeping[L_idx]) + 1e-12
    )
    print(f"Beam sweeping, L={L:2d}: {mean_sinr_db:7.3f} dB")

if not np.all(np.isfinite(sinr_sweeping)):
    raise FloatingPointError(
        "Beam sweeping produced a non-finite SINR."
    )

sinr_results["beam_sweeping"] = sinr_sweeping


# %%
#####################################################################################
#                                     Plot                                          #
#####################################################################################

methods = {
    "two_lstm": {
        "label": "Two LSTM (trained L=2)",
        "color": "#2ca02c",
        "marker": "o",
    },
    "one_lstm": {
        "label": "One fused LSTM (trained L=2)",
        "color": "#d62728",
        "marker": "d",
    },
    "beam_sweeping": {
        "label": "Beam sweeping",
        "color": "#ff7f0e",
        "marker": "s",
    },
}

fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.5))

for method_name in ["two_lstm", "one_lstm", "beam_sweeping"]:
    mean_sinr_db = 10 * np.log10(
        np.mean(sinr_results[method_name], axis=1) + 1e-12
    )
    ax.plot(
        L_values,
        mean_sinr_db,
        color=methods[method_name]["color"],
        marker=methods[method_name]["marker"],
        linewidth=1.5,
        markersize=6,
        label=methods[method_name]["label"],
    )

ax.set_xlabel("Alternating BD code length, L")
ax.set_ylabel("Achieved SINR [dB]")
ax.set_xticks(L_values)
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
