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
    parser.add_argument('--N_ris', type=int, default=36, help='Number of receiver antennas')
    parser.add_argument('--N_bs', type=int, default=1, help='Number of BS antennas')
    parser.add_argument('--tau', type=int, default=20, help='Number of pilots')
    parser.add_argument('--snr', type=int, default=10, help='Signal-to-noise ratio (dB)')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num_users', type=int, default=1, help='Number of users')
    parser.add_argument('--rician_factor', type=float, default=5.0, help='Rician factor')
    parser.add_argument('--fc', type=float, default=2400000000.0, help='Carrier frequency in Hz')
    parser.add_argument('--N_scatterers', type=int, default=1, help='Number of scatterers')
    parser.add_argument('--N_symbols', type=int, default=1, help='Number of symbols of one sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args, unknown = parser.parse_known_args(filtered_argv[1:])
    return args
