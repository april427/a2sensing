import argparse
import csv
import itertools
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_metrics(stdout_text):
    patterns = {
        "test_sinr_learned_db": r"Test SINR_BD \(learned\):\s*([-+]?\d+(?:\.\d+)?)\s*dB",
        "test_sinr_optimal_db": r"Test SINR_BD \(optimal\):\s*([-+]?\d+(?:\.\d+)?)\s*dB",
        "test_sinr_sp_db": r"Test SINR_BD \(signalprocessing\):\s*([-+]?\d+(?:\.\d+)?)\s*dB",
        "test_sinr_sweep_db": r"Test SINR_BD \(sweeping\):\s*([-+]?\d+(?:\.\d+)?)\s*dB",
        "gap_to_optimal_db": r"Gap to optimal:\s*([-+]?\d+(?:\.\d+)?)\s*dB",
    }
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout_text)
        out[key] = float(match.group(1)) if match else None
    return out


def make_parser():
    parser = argparse.ArgumentParser(description="Grid sweep for mimo_mono.py hyperparameters")
    parser.add_argument("--script", type=str, default="mimo_mono.py", help="Training script to run")
    parser.add_argument("--tau_list", type=str, default="24", help="Comma-separated tau values")
    parser.add_argument("--learning_rate_list", type=str, default="1e-4,2e-4,3e-4", help="Comma-separated LR values")
    parser.add_argument("--clip_norm_list", type=str, default="3.0,5.0", help="Comma-separated clip-norm values")
    parser.add_argument("--warmup_steps_list", type=str, default="300,600", help="Comma-separated warmup-step values")
    parser.add_argument("--decay_steps_list", type=str, default="1000,3000", help="Comma-separated decay-step values")
    parser.add_argument("--decay_rate_list", type=str, default="0.95,0.97", help="Comma-separated decay-rate values")
    parser.add_argument("--hidden_size_list", type=str, default="128", help="Comma-separated hidden-size values")

    parser.add_argument("--snr", type=int, default=10, help="Fixed SNR for all runs")
    parser.add_argument("--N_scatterers", type=int, default=5, help="Fixed number of scatterers for all runs")
    parser.add_argument("--n_epochs", type=int, default=30, help="Epochs per run")
    parser.add_argument("--N_symbols", type=int, default=1, help="OFDM symbols per state")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--max_runs", type=int, default=0, help="Cap number of runs (0 means all)")
    parser.add_argument("--extra_args", type=str, default="", help="Extra args passed verbatim to training script")
    parser.add_argument("--output_dir", type=str, default="sweep_runs", help="Root output directory")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands without executing")
    return parser


def main():
    args = make_parser().parse_args()

    tau_values = parse_int_list(args.tau_list)
    lr_values = parse_float_list(args.learning_rate_list)
    clip_values = parse_float_list(args.clip_norm_list)
    warmup_values = parse_int_list(args.warmup_steps_list)
    decay_steps_values = parse_int_list(args.decay_steps_list)
    decay_rate_values = parse_float_list(args.decay_rate_list)
    hidden_values = parse_int_list(args.hidden_size_list)

    extra_args = shlex.split(args.extra_args)

    grid = list(
        itertools.product(
            tau_values,
            lr_values,
            clip_values,
            warmup_values,
            decay_steps_values,
            decay_rate_values,
            hidden_values,
        )
    )

    if args.max_runs > 0:
        grid = grid[: args.max_runs]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"sweep_{ts}")
    logs_dir = os.path.join(run_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    csv_path = os.path.join(run_root, "results.csv")

    fieldnames = [
        "run_id",
        "tau",
        "learning_rate",
        "clip_norm",
        "warmup_steps",
        "decay_steps",
        "decay_rate",
        "hidden_size",
        "snr",
        "N_scatterers",
        "n_epochs",
        "N_symbols",
        "seed",
        "duration_sec",
        "return_code",
        "test_sinr_learned_db",
        "test_sinr_optimal_db",
        "test_sinr_sp_db",
        "test_sinr_sweep_db",
        "gap_to_optimal_db",
        "log_file",
        "command",
    ]

    print(f"Planned runs: {len(grid)}")
    print(f"Output directory: {run_root}")

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (tau, lr, clip, warmup, decay_steps, decay_rate, hidden_size) in enumerate(grid, start=1):
            run_id = f"run_{idx:04d}"
            log_file = os.path.join(logs_dir, f"{run_id}.log")
            run_seed = args.seed + idx - 1

            cmd = [
                sys.executable,
                args.script,
                "--tau",
                str(tau),
                "--snr",
                str(args.snr),
                "--N_scatterers",
                str(args.N_scatterers),
                "--n_epochs",
                str(args.n_epochs),
                "--N_symbols",
                str(args.N_symbols),
                "--seed",
                str(run_seed),
                "--learning_rate",
                str(lr),
                "--clip_norm",
                str(clip),
                "--warmup_steps",
                str(warmup),
                "--decay_steps",
                str(decay_steps),
                "--decay_rate",
                str(decay_rate),
                "--hidden_size",
                str(hidden_size),
            ] + extra_args

            cmd_str = " ".join(shlex.quote(x) for x in cmd)
            print(f"[{idx}/{len(grid)}] {cmd_str}")

            if args.dry_run:
                row = {
                    "run_id": run_id,
                    "tau": tau,
                    "learning_rate": lr,
                    "clip_norm": clip,
                    "warmup_steps": warmup,
                    "decay_steps": decay_steps,
                    "decay_rate": decay_rate,
                    "hidden_size": hidden_size,
                    "snr": args.snr,
                    "N_scatterers": args.N_scatterers,
                    "n_epochs": args.n_epochs,
                    "N_symbols": args.N_symbols,
                    "seed": run_seed,
                    "duration_sec": 0.0,
                    "return_code": 0,
                    "test_sinr_learned_db": None,
                    "test_sinr_optimal_db": None,
                    "test_sinr_sp_db": None,
                    "test_sinr_sweep_db": None,
                    "gap_to_optimal_db": None,
                    "log_file": log_file,
                    "command": cmd_str,
                }
                writer.writerow(row)
                f_csv.flush()
                continue

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - t0

            combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            with open(log_file, "w") as f_log:
                f_log.write(combined_out)

            metrics = parse_metrics(combined_out)
            row = {
                "run_id": run_id,
                "tau": tau,
                "learning_rate": lr,
                "clip_norm": clip,
                "warmup_steps": warmup,
                "decay_steps": decay_steps,
                "decay_rate": decay_rate,
                "hidden_size": hidden_size,
                "snr": args.snr,
                "N_scatterers": args.N_scatterers,
                "n_epochs": args.n_epochs,
                "N_symbols": args.N_symbols,
                "seed": run_seed,
                "duration_sec": round(duration, 2),
                "return_code": proc.returncode,
                "test_sinr_learned_db": metrics["test_sinr_learned_db"],
                "test_sinr_optimal_db": metrics["test_sinr_optimal_db"],
                "test_sinr_sp_db": metrics["test_sinr_sp_db"],
                "test_sinr_sweep_db": metrics["test_sinr_sweep_db"],
                "gap_to_optimal_db": metrics["gap_to_optimal_db"],
                "log_file": log_file,
                "command": cmd_str,
            }
            writer.writerow(row)
            f_csv.flush()

            learned = metrics["test_sinr_learned_db"]
            sweep = metrics["test_sinr_sweep_db"]
            if learned is None:
                print(f"  -> run failed or metrics missing (return_code={proc.returncode})")
            else:
                delta_vs_sweep = None if sweep is None else round(learned - sweep, 2)
                print(
                    f"  -> learned={learned:.2f} dB, sweep={sweep if sweep is not None else 'NA'} dB, "
                    f"delta={delta_vs_sweep if delta_vs_sweep is not None else 'NA'} dB"
                )

    print("Sweep complete.")
    print(f"Results CSV: {csv_path}")


if __name__ == "__main__":
    main()
