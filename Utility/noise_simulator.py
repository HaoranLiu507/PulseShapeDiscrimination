import os
import sys
import math
import argparse
from typing import Tuple, List, Optional

import numpy as np


def load_txt_matrix(file_path: str) -> Tuple[np.ndarray, int]:
    """Load a numeric matrix from a .txt file, skipping any leading comment/header lines.

    Returns the matrix and the number of skipped header lines.
    """
    # Detect header lines that start with '#'
    header_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.lstrip().startswith('#'):
                header_lines += 1
            else:
                break

    data = np.loadtxt(file_path, comments='#')
    if data.ndim == 1:
        # Ensure 2D shape for consistency (one row)
        data = data[np.newaxis, :]
    return data, header_lines


def save_txt_matrix(file_path: str, data: np.ndarray, header: str = '') -> None:
    """Save 2D matrix to a .txt file with optional header (not commented).

    Uses 6 decimal places to match existing dataset style.
    """
    if header:
        # Write header line(s) as-is (no '#') to mirror the provided files
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            if not header.endswith('\n'):
                f.write('\n')
        with open(file_path, 'ab') as f:
            np.savetxt(f, data, fmt='%1.6f')
    else:
        np.savetxt(file_path, data, fmt='%1.6f')


def apply_lowpass_filter(row: np.ndarray, filter_width: int) -> np.ndarray:
    """Apply a simple moving-average low-pass filter."""
    width = max(1, int(filter_width))
    if width <= 1:
        return row
    kernel = np.ones(width, dtype=np.float64) / float(width)
    # Use convolution with same output length; pad with edge values for stability
    padded = np.pad(row, (width // 2, width - 1 - width // 2), mode='edge')
    filtered = np.convolve(padded, kernel, mode='valid')
    return filtered.astype(row.dtype, copy=False)


def enforce_clip_and_power(
    signal: np.ndarray,
    noise: np.ndarray,
    limit: Optional[float],
    target_power: float,
    max_iter: int = 6,
    tol: float = 1e-3,
) -> np.ndarray:
    """Iteratively clip and rescale noise to satisfy both amplitude limit and target power."""
    if limit is None or limit <= 0.0:
        return noise
    target_power = float(target_power)
    if target_power <= 0.0 or not np.isfinite(target_power):
        return np.zeros_like(signal)

    signal = signal.astype(np.float64, copy=False)
    noise = noise.astype(np.float64, copy=False)
    limit = float(limit)

    for _ in range(max_iter):
        noisy_row = np.clip(signal + noise, -limit, limit)
        noise = noisy_row - signal
        current_power = float(np.mean(noise ** 2))
        if current_power <= 0.0 or not np.isfinite(current_power):
            return np.zeros_like(signal)
        if abs(current_power - target_power) / target_power <= tol:
            return noise
        scale = math.sqrt(target_power / current_power)
        noise *= scale

    return noise


def compute_noise_for_target_snr(
    signal_row: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
    filter_width: int = 1,
    clip_sigma: Optional[float] = None,
) -> np.ndarray:
    """Generate Gaussian noise so that resulting row has the target SNR (in dB).

    Noise is optionally low-pass filtered before being rescaled to hit the target power.
    """
    signal = signal_row.astype(np.float64)
    signal_power = float(np.mean(signal ** 2))
    if signal_power <= 0.0 or not np.isfinite(signal_power):
        return np.zeros_like(signal)

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = rng.standard_normal(signal.shape)
    noise = apply_lowpass_filter(noise, filter_width)
    limit = None
    if clip_sigma:
        signal_std = float(np.std(signal))
        if signal_std > 0.0 and np.isfinite(signal_std):
            limit = clip_sigma * signal_std
    if limit and limit > 0.0:
        noise = enforce_clip_and_power(signal, noise, limit, noise_power)
    current_noise_power = float(np.mean(noise ** 2))
    if current_noise_power <= 0.0 or not np.isfinite(current_noise_power):
        return np.zeros_like(signal)
    scale = math.sqrt(noise_power / current_noise_power)
    return noise * scale


def add_noise_instant(
    data: np.ndarray,
    target_snr_db: float,
    seed: int,
    filter_width: int,
    clip_sigma: Optional[float],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = np.empty_like(data, dtype=np.float64)
    clip = clip_sigma if clip_sigma and clip_sigma > 0 else None
    for i in range(data.shape[0]):
        row = data[i, :].astype(np.float64)
        noise = compute_noise_for_target_snr(row, target_snr_db, rng, filter_width, clip)
        noisy_row = row + noise
        noisy[i, :] = noisy_row
    return noisy


def add_noise_schedule(
    data: np.ndarray,
    target_snr_db: float,
    seed: int,
    filter_width: int,
    steps: int,
    clip_sigma: Optional[float],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = np.empty_like(data, dtype=np.float64)
    steps = max(1, int(steps))
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    for i in range(data.shape[0]):
        signal = data[i, :].astype(np.float64)
        signal_power = float(np.mean(signal ** 2))
        if signal_power <= 0.0 or not np.isfinite(signal_power):
            noisy[i, :] = signal
            continue

        target_noise_power = signal_power / snr_linear
        cumulative_noise = np.zeros_like(signal)
        current_power = 0.0
        base_sigma = float(np.std(signal))

        for step_idx in range(steps):
            remaining_power = target_noise_power - current_power
            if remaining_power <= 0.0:
                break

            sample = rng.standard_normal(signal.shape)
            sample = apply_lowpass_filter(sample, filter_width)
            sample_power = float(np.mean(sample ** 2))
            if sample_power <= 0.0 or not np.isfinite(sample_power):
                continue

            remaining_steps = steps - step_idx
            desired_increment = remaining_power / remaining_steps
            scale = math.sqrt(desired_increment / sample_power)
            additive = sample * scale
            cumulative_noise += additive

            noisy_row = signal + cumulative_noise

            current_power = float(np.mean(cumulative_noise ** 2))

        if current_power > 0.0 and np.isfinite(current_power):
            rescale = math.sqrt(target_noise_power / current_power)
            cumulative_noise *= rescale
        else:
            cumulative_noise.fill(0.0)

        if clip_sigma and base_sigma > 0.0:
            limit = clip_sigma * base_sigma
            cumulative_noise = enforce_clip_and_power(signal, cumulative_noise, limit, target_noise_power)

        noisy[i, :] = signal + cumulative_noise

    return noisy


def add_noise_to_matrix(
    data: np.ndarray,
    target_snr_db: float,
    seed: int = 42,
    mode: str = "instant",
    filter_width: int = 1,
    schedule_steps: int = 5,
    clip_sigma: Optional[float] = None,
) -> np.ndarray:
    if mode == "instant":
        return add_noise_instant(data, target_snr_db, seed, filter_width, clip_sigma)
    if mode == "schedule":
        return add_noise_schedule(data, target_snr_db, seed, filter_width, schedule_steps, clip_sigma)
    raise ValueError(f"Unknown noise mode: {mode}")


def build_output_path(input_path: str, snr_value: float) -> str:
    """Construct output filename by inserting _noisy_SNR<value> before the .txt extension."""
    directory, basename = os.path.split(input_path)
    name, ext = os.path.splitext(basename)
    # Keep integer if whole number, else keep as compact string
    if abs(snr_value - round(snr_value)) < 1e-9:
        snr_str = f"{int(round(snr_value))}"
    else:
        # Avoid trailing zeros; limit precision
        snr_str = f"{snr_value:.4g}"
    return os.path.join(directory, f"{name}_noise_SNR{snr_str}{ext}")


def find_data_files(root: str) -> List[str]:
    """Find .txt data files under Data/Train, Data/Test, and Data/Validation."""
    candidates: List[str] = []
    for sub in ("Train", "Test", "Validation"):
        subdir = os.path.join(root, sub)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.lower().endswith('.txt'):
                candidates.append(os.path.join(subdir, fname))
    return candidates


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Gaussian noise to dataset to achieve target SNR per pulse.")
    parser.add_argument("--snr", type=float, default=None, help="Target SNR in dB (e.g., 10 for 10 dB)")
    parser.add_argument("--data-root", type=str, default=os.path.join("Data"), help="Root data folder containing Train/Test/Validation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--mode", choices=["instant", "schedule"], default="instant", help="Noise addition mode")
    parser.add_argument("--filter-width", type=int, default=5, help="Filter width for low-pass smoothing")
    parser.add_argument("--schedule-steps", type=int, default=5, help="Number of increments for schedule mode")
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=0.0,
        help="Sigma multiple for optional clipping (set > 0 to enable)",
    )
    return parser.parse_args(argv)


def interactive_prompt_for_snr() -> float:
    while True:
        user_input = input("Enter target SNR in dB (e.g., 10): ").strip()
        try:
            value = float(user_input)
            return value
        except ValueError:
            print("Invalid number. Please enter a numeric SNR in dB.")


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    target_snr_db = args.snr if args.snr is not None else interactive_prompt_for_snr()
    data_root = args.data_root
    seed = args.seed
    mode = args.mode
    filter_width = args.filter_width
    schedule_steps = args.schedule_steps
    clip_sigma = None if args.clip_sigma <= 0 else args.clip_sigma

    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    files = find_data_files(data_root)
    if not files:
        print(f"No .txt data files found under {data_root}/(Train|Test|Validation).")
        sys.exit(1)

    print(f"Processing {len(files)} files with target SNR = {target_snr_db} dB...")
    for path in files:
        try:
            # Preserve the first header line if present (e.g., '# Gamma Train Signals')
            raw_header = ''
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line.lstrip().startswith('#'):
                    raw_header = first_line.rstrip('\n')

            data, _ = load_txt_matrix(path)
            noisy = add_noise_to_matrix(
                data,
                target_snr_db,
                seed=seed,
                mode=mode,
                filter_width=filter_width,
                schedule_steps=schedule_steps,
                clip_sigma=clip_sigma,
            )
            out_path = build_output_path(path, target_snr_db)
            save_txt_matrix(out_path, noisy, header=raw_header)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed on {path}: {e}")


if __name__ == "__main__":
    main()


