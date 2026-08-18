import argparse
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    # Filter out Jupyter kernel arguments
    filtered_argv = [arg for arg in sys.argv if 'kernel' not in arg.lower() and not arg.endswith('.json')]


    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='mono', choices=['mono', 'bi'], help='monostatic or bistatic')
    parser.add_argument('--N_ris', type=int, default=16, help='Number of receiver antennas')
    parser.add_argument('--N_bs', type=int, default=1, help='Number of BS antennas')
    parser.add_argument('--tau', type=int, default=10, help='Number of pilots')
    parser.add_argument('--snr', type=int, default=10, help='Signal-to-noise ratio (dB)')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_users', type=int, default=1, help='Number of users')
    parser.add_argument('--rician_factor', type=float, default=5.0, help='Rician factor')
    parser.add_argument('--fc', type=float, default=10e9, help='Carrier frequency in Hz')
    parser.add_argument('--N_scatterers', type=int, default=5, help='Number of scatterers')
    parser.add_argument('--N_symbols', type=int, default=1, help='Number of symbols of one sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--clip_norm', type=float, default=3.0, help='Global gradient clipping norm')
    parser.add_argument('--warmup_steps', type=int, default=600, help='Learning-rate warmup steps')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Learning-rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.97, help='Exponential decay rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='L2 regularization weight')
    parser.add_argument('--batch_per_epoch', type=int, default=128, help='Mini-batches per epoch')
    parser.add_argument('--batch_size_order', type=int, default=4, help='Batch size multiplier (batch = value * 32)')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--patience', type=int, default=-1, help='Early stopping patience; <=0 uses default max(20, 2*tau)')
    parser.add_argument('--curriculum_epochs', type=int, default=-1,
                        help='Epochs spent annealing the training SNR up to --snr; '
                             '<0 = auto (used only when --snr > 10 dB), 0 = disabled')
    parser.add_argument('--snr_start', type=float, default=0.0,
                        help='Starting SNR (dB) of the curriculum; ignored when the curriculum is off')

    args, unknown = parser.parse_known_args(filtered_argv[1:])
    return args
