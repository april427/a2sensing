#!/usr/bin/env python3
"""
Test Mo_mimo_sinr learned models and plot beam pattern variation over time.

This script restores learned NN parameters, generates BD and scatterer targets
with MIMO channel generation, runs testing, and visualizes beam evolution
across preamble steps.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from keras.layers import BatchNormalization, Dense
from parse_args import parse_args
from channel_functions import generate_location_mimo, generate_mimo_channel


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, dims, name):
        super(MLPBlock, self).__init__()
        self.layers_list = []
        for ii in range(num_layers - 1):
            self.layers_list.append(Dense(units=dims[ii], activation='relu', name=name + '_relu_' + str(ii)))
            self.layers_list.append(BatchNormalization())
        self.layers_list.append(Dense(units=dims[-1], activation='linear', name=name + '_linear'))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class LSTM_Cell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name):
        super(LSTM_Cell, self).__init__()
        self.layer_Ui = Dense(units=hidden_size, activation='linear', name='Ui' + name)
        self.layer_Wi = Dense(units=hidden_size, activation='linear', name='Wi' + name)
        self.layer_Uf = Dense(units=hidden_size, activation='linear', name='Uf' + name)
        self.layer_Wf = Dense(units=hidden_size, activation='linear', name='Wf' + name)
        self.layer_Uo = Dense(units=hidden_size, activation='linear', name='Uo' + name)
        self.layer_Wo = Dense(units=hidden_size, activation='linear', name='Wo' + name)
        self.layer_Uc = Dense(units=hidden_size, activation='linear', name='Uc' + name)
        self.layer_Wc = Dense(units=hidden_size, activation='linear', name='Wc' + name)

    def call(self, inputs, **kwargs):
        input_x, h_old, c_old = inputs
        i_t = tf.sigmoid(self.layer_Ui(input_x) + self.layer_Wi(h_old))
        f_t = tf.sigmoid(self.layer_Uf(input_x) + self.layer_Wf(h_old))
        o_t = tf.sigmoid(self.layer_Uo(input_x) + self.layer_Wo(h_old))
        c_t = tf.tanh(self.layer_Uc(input_x) + self.layer_Wc(h_old))
        c = i_t * c_t + f_t * c_old
        h_new = o_t * tf.tanh(c)
        return h_new, c


def resolve_model_path(base_dir, n_ant, tau, snr_db, n_symbols):
    base = os.path.join(base_dir, f"params_sinr_N_{n_ant}_{n_ant}_tau_{tau}_snr_{int(snr_db)}")
    if os.path.exists(base + '.meta'):
        return base

    patterns = [
        os.path.join(base_dir, f"params_sinr_N_{n_ant}_{n_ant}_tau_{tau}_snr_{int(snr_db[0])}_K_{n_symbols}*")
    ]
    for pattern in patterns:
        candidates = sorted(p for p in glob.glob(pattern) if os.path.exists(p + '.meta'))
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {base_dir} for N={n_ant}, tau={tau}, snr={snr_db}."
    )


def build_inference_graph(args):
    tf.reset_default_graph()

    n_tx = args.N_ris
    n_rx = args.N_ris
    tau = args.tau
    k_sym = getattr(args, 'N_symbols', 1)
    noise_std = np.sqrt(0.5)
    bd_mod = np.array([(-1) ** t for t in range(tau)], dtype=np.float32)

    loc_input = tf.placeholder(tf.float32, shape=(None, 3, 1), name='loc_input')
    h_d_ph = tf.placeholder(tf.complex64, shape=(None, n_rx, n_tx), name='H_d')
    h_b_ph = tf.placeholder(tf.complex64, shape=(None, n_rx, n_tx), name='H_b')
    h_r_ph = tf.placeholder(tf.complex64, shape=(None, n_rx, n_tx), name='H_r')

    lay = {}
    lay['P'] = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(), name='tx_power')
    bd_seq = tf.constant(bd_mod, dtype=tf.float32)

    with tf.name_scope('active_sensing_agent'):
        hidden_size = 128
        lstm1 = LSTM_Cell(hidden_size, name='LSTM_1')
        lstm2 = LSTM_Cell(hidden_size, name='LSTM_2')
        mlp_ris_tx = MLPBlock(3, [hidden_size * 2, hidden_size * 2, 2 * n_tx], name='RIS_transmitter')
        mlp_rx_rx = MLPBlock(3, [hidden_size * 2, hidden_size * 2, 2 * n_rx], name='Receiver_receiver')

        snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
        snr_db = 10 * tf.log(snr) / np.log(10)
        x_bd = [tf.tile(tf.reshape(bd_seq[t], [1, 1]), [tf.shape(loc_input)[0], 1]) for t in range(tau)]

        y1_acc = tf.zeros([tf.shape(loc_input)[0], n_rx], dtype=tf.complex64)
        y2_acc = tf.zeros([tf.shape(loc_input)[0], n_rx], dtype=tf.complex64)

        w_seq = []
        v_seq = []
        batch_size = tf.shape(loc_input)[0]

        for t in range(tau):
            if t == 0:
                h_old = tf.zeros([batch_size, hidden_size])
                c_old = tf.zeros([batch_size, hidden_size])
                h_old2 = tf.zeros([batch_size, hidden_size])
                c_old2 = tf.zeros([batch_size, hidden_size])

                w_init_real = tf.get_variable('w_init_real', shape=(1, n_tx, 1), trainable=True)
                w_init_imag = tf.get_variable('w_init_imag', shape=(1, n_tx, 1), trainable=True)
                w_complex_init = tf.complex(w_init_real, w_init_imag)
                w1 = w_complex_init / tf.cast(tf.norm(w_complex_init, axis=1, keepdims=True), tf.complex64)

                v_init_real = tf.get_variable('v_init_real', shape=(1, n_rx, 1), trainable=True)
                v_init_imag = tf.get_variable('v_init_imag', shape=(1, n_rx, 1), trainable=True)
                v_complex_init = tf.complex(v_init_real, v_init_imag)
                v1 = v_complex_init / tf.cast(tf.norm(v_complex_init, axis=1, keepdims=True), tf.complex64)

            x_bd_t = tf.reshape(tf.cast(x_bd[t], tf.complex64), [-1, 1, 1])
            h_eff = x_bd_t * h_b_ph + h_d_ph + h_r_ph
            h_eff2 = -x_bd_t * h_b_ph + h_d_ph + h_r_ph

            y_noiseless1 = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.matmul(h_eff, w1)
            y_noiseless1 = tf.tile(y_noiseless1, [1, 1, k_sym])
            y_noiseless2 = tf.tile(tf.complex(tf.sqrt(lay['P']), 0.0) * tf.matmul(h_eff2, w1), [1, 1, k_sym])

            noise = tf.complex(
                tf.random_normal([batch_size, n_rx, k_sym], mean=0.0, stddev=noise_std),
                tf.random_normal([batch_size, n_rx, k_sym], mean=0.0, stddev=noise_std),
            )
            noise2 = tf.complex(
                tf.random_normal([batch_size, n_rx, k_sym], mean=0.0, stddev=noise_std),
                tf.random_normal([batch_size, n_rx, k_sym], mean=0.0, stddev=noise_std),
            )

            y_complex1 = y_noiseless1 + noise
            y_complex2 = y_noiseless2 + noise2

            y1_acc = y1_acc + tf.reduce_mean(x_bd_t * y_complex1 - x_bd_t * y_complex2, axis=2)
            y2_acc = y2_acc + tf.reduce_mean(y_complex1 + y_complex2, axis=2)
            y1_after = tf.reduce_mean(tf.matmul(tf.linalg.adjoint(v1), tf.reshape(y1_acc, [-1, n_rx, 1])), axis=2)
            y2_after = tf.reduce_mean(tf.matmul(tf.linalg.adjoint(v1), tf.reshape(y2_acc, [-1, n_rx, 1])), axis=2)

            y_real = tf.concat(
                [
                    tf.cast(tf.real(y1_acc), tf.float32),
                    tf.cast(tf.imag(y1_acc), tf.float32),
                    tf.cast(tf.real(y1_after), tf.float32),
                    tf.cast(tf.imag(y1_after), tf.float32),
                ],
                axis=1,
            )
            y_real2 = tf.concat(
                [
                    tf.cast(tf.real(y2_acc), tf.float32),
                    tf.cast(tf.imag(y2_acc), tf.float32),
                    tf.cast(tf.real(y2_after), tf.float32),
                    tf.cast(tf.imag(y2_after), tf.float32),
                ],
                axis=1,
            )

            h_old, c_old = lstm1((tf.concat([y_real, snr_db], axis=1), h_old, c_old))
            h_old2, c_old2 = lstm2((tf.concat([y_real2, snr_db], axis=1), h_old2, c_old2))

            w_her = mlp_ris_tx(tf.concat([h_old, h_old2], axis=1))
            w_her = tf.divide(w_her, tf.reshape(tf.norm(w_her, axis=1), (-1, 1)) + 1e-8)
            w1 = tf.complex(w_her[:, 0:n_tx], w_her[:, n_tx:2 * n_tx])
            w1 = tf.reshape(w1, [-1, n_tx, 1])

            v_her = mlp_rx_rx(tf.concat([h_old, h_old2], axis=1))
            v_her = tf.divide(v_her, tf.reshape(tf.norm(v_her, axis=1), (-1, 1)) + 1e-8)
            v1 = tf.complex(v_her[:, 0:n_rx], v_her[:, n_rx:2 * n_rx])
            v1 = tf.reshape(v1, [-1, n_rx, 1])

            w_seq.append(w1)
            v_seq.append(v1)

        mlp_bf_w = MLPBlock(3, [2 * hidden_size, 2 * hidden_size, 2 * n_tx], name='MLP_bf_w')
        mlp_bf_v = MLPBlock(3, [2 * hidden_size, 2 * hidden_size, 2 * n_rx], name='MLP_bf_v')

        w_tmp = mlp_bf_w(tf.concat([c_old, c_old2], axis=1))
        w_tmp = tf.divide(w_tmp, tf.reshape(tf.norm(w_tmp, axis=1), (-1, 1)) + 1e-8)
        w_final = tf.complex(w_tmp[:, 0:n_tx], w_tmp[:, n_tx:2 * n_tx])
        w_final = tf.reshape(w_final, [-1, n_tx, 1])

        v_tmp = mlp_bf_v(tf.concat([c_old, c_old2], axis=1))
        v_tmp = tf.divide(v_tmp, tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1)) + 1e-8)
        v_final = tf.complex(v_tmp[:, 0:n_rx], v_tmp[:, n_rx:2 * n_rx])
        v_final = tf.reshape(v_final, [-1, n_rx, 1])

        h_int = h_d_ph + h_r_ph
        sig = tf.squeeze(tf.abs(tf.matmul(tf.linalg.adjoint(v_final), tf.matmul(h_b_ph, w_final))) ** 2) * lay['P']
        inter = tf.squeeze(tf.abs(tf.matmul(tf.linalg.adjoint(v_final), tf.matmul(h_int, w_final))) ** 2) * lay['P']
        sinr = sig / (inter + 1.0 + 1e-10)

    graph = {
        'placeholders': {
            'loc_input': loc_input,
            'H_d': h_d_ph,
            'H_b': h_b_ph,
            'H_r': h_r_ph,
            'P': lay['P'],
        },
        'outputs': {
            'w_seq': w_seq,
            'v_seq': v_seq,
            'w_final': w_final,
            'v_final': v_final,
            'sinr': sinr,
        },
    }
    return graph


def generate_test_samples(num_samples, n_tx, n_rx, n_scatters, rician_factor, location_tx, location_rx):
    samples = []
    for _ in range(num_samples):
        bd_loc = generate_location_mimo(1, 'u')[0]
        scatter_loc = generate_location_mimo(n_scatters, 's')
        _, h_d, h_r, h_b = generate_mimo_channel(
            location_tx,
            location_rx,
            scatter_loc,
            bd_loc,
            N_tx_h=n_tx,
            N_tx_v=1,
            N_rx_h=n_rx,
            N_rx_v=1,
            Rician_factor=rician_factor,
        )
        d_bd_rx = np.linalg.norm(bd_loc - location_rx)
        d_bd_tx = np.linalg.norm(bd_loc - location_tx)
        az_bd = np.arctan2(bd_loc[1] - location_rx[1], bd_loc[0] - location_rx[0])
        loc_info = np.array([az_bd, d_bd_rx, d_bd_tx], dtype=np.float32)[:, np.newaxis]
        samples.append({'H_d': h_d, 'H_r': h_r, 'H_b': h_b, 'loc_info': loc_info, 'bd_loc': bd_loc, 'scatter_loc': scatter_loc})
    return samples


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
    return 10 * np.log10(gains + 1e-12)


def compute_sinr_steps(w_seq, v_seq, h_b, h_d, h_r, p_tx, noise_var=1.0):
    sinr = np.zeros(w_seq.shape[0], dtype=np.float64)
    h_int = h_d + h_r
    for t in range(w_seq.shape[0]):
        w = w_seq[t]
        v = v_seq[t]
        sig = p_tx * np.abs(np.conj(v).T @ h_b @ w).item() ** 2
        inter = p_tx * np.abs(np.conj(v).T @ h_int @ w).item() ** 2
        sinr[t] = sig / (inter + noise_var)
    return 10 * np.log10(sinr + 1e-12)


args = parse_args()
n_tx = args.N_ris
n_rx = args.N_ris
n_scatters = 5
tau = args.tau
k_sym = getattr(args, 'N_symbols', 1)
snr_db = (args.snr)
p_tx = 10 ** (snr_db / 10)

model_dir = 'Mo_mimo_sinr'
model_path = resolve_model_path(model_dir, n_tx, tau, snr_db, k_sym)
print(f'Restoring checkpoint: {model_path}')

location_tx = np.array([0.0, 0.0, 0.0])
location_rx = np.array([0.0, 0.0, 0.0])

graph = build_inference_graph(args)
saver = tf.train.Saver()

num_samples = max(1, int(getattr(args, 'num_users', 1)))
test_samples = generate_test_samples(num_samples, n_tx, n_rx, n_scatters, args.rician_factor, location_tx, location_rx)

angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
angles_deg = np.degrees(angles)

with tf.Session() as sess:
    saver.restore(sess, model_path)
    print('Model restored successfully.')

    for idx, sample in enumerate(test_samples):
        feed = {
            graph['placeholders']['loc_input']: sample['loc_info'][np.newaxis, :, :],
            graph['placeholders']['H_d']: sample['H_d'][np.newaxis, :, :],
            graph['placeholders']['H_b']: sample['H_b'][np.newaxis, :, :],
            graph['placeholders']['H_r']: sample['H_r'][np.newaxis, :, :],
            graph['placeholders']['P']: np.float32(p_tx),
        }
        w_seq_tf, v_seq_tf, sinr_final_tf = sess.run(
            [graph['outputs']['w_seq'], graph['outputs']['v_seq'], graph['outputs']['sinr']],
            feed_dict=feed,
        )

        w_seq = np.array([w[0, :, 0] for w in w_seq_tf])
        v_seq = np.array([v[0, :, 0] for v in v_seq_tf])
        sinr_steps_db = compute_sinr_steps(
            w_seq[:, :, np.newaxis],
            v_seq[:, :, np.newaxis],
            sample['H_b'],
            sample['H_d'],
            sample['H_r'],
            p_tx,
            noise_var=1.0,
        )
        true_bd_angle_deg = np.degrees(np.arctan2(sample['bd_loc'][1], sample['bd_loc'][0]))
        scatter_angles = [np.degrees(np.arctan2(s[1], s[0])) for s in np.atleast_2d(sample['scatter_loc'])]

        n_plot = tau
        t_indices = np.linspace(0, tau - 1, n_plot, dtype=int)
        n_cols = 4
        n_rows = int(np.ceil(n_plot / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.8 * n_rows))
        axes = np.atleast_1d(axes).reshape(-1)

        # Hide empty panels in the last row when n_plot is not a multiple of n_cols.
        for empty_idx in range(n_plot, len(axes)):
            axes[empty_idx].axis('off')

        for p_idx, t in enumerate(t_indices):
            beam_db = joint_beam_pattern_db(v_seq[t], w_seq[t], angles)
            axes[p_idx].plot(angles_deg, beam_db, color='#1f77b4', linewidth=2, label='Learned')
            axes[p_idx].axvline(true_bd_angle_deg, color='#d62728', linestyle='--', linewidth=1.8, label='BD')
            for s_idx, s_ang in enumerate(scatter_angles):
                axes[p_idx].axvline(
                    s_ang,
                    color='#2ca02c',
                    linestyle=':',
                    linewidth=1.6,
                    label='Target' if s_idx == 0 else None,
                )
            axes[p_idx].set_title(f't={t + 1}, SINR={sinr_steps_db[t]:.2f} dB')
            axes[p_idx].set_xlabel('Angle (deg)')
            if p_idx == 0:
                axes[p_idx].set_ylabel('Joint gain (dB)')
                axes[p_idx].legend(fontsize=9)
            axes[p_idx].set_xlim([-90, 90])
            axes[p_idx].set_ylim([-45, 1])
            axes[p_idx].grid(alpha=0.25)

        fig.suptitle(
            f'Sample {idx + 1} | Final SINR={10*np.log10(np.maximum(sinr_final_tf[0], 1e-12)):.2f} dB | '
            f'BD angle={true_bd_angle_deg:.1f} deg | Target angles={np.round(scatter_angles, 1).tolist()}'
        )
        plt.tight_layout()
        plt.show()



