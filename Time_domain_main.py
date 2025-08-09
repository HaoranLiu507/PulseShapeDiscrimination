import os
import argparse
import numpy as np
import importlib
import time
from functools import partial
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom
from Utility import filters

def load_dataset(file_path: str) -> np.ndarray:
    """Load dataset from a text file and validate its contents."""
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f'Data import failed. Details: {e}')

    if data.size == 0:
        raise ValueError('Data import failed. Dataset is empty.')

    return data


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize each row of the input data to the range [0, 1].

    Raises:
        ValueError: if input is not a numpy array or is empty.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if data.size == 0:
        raise ValueError("Input data is empty.")

    min_values = np.min(data, axis=1, keepdims=True)
    max_values = np.max(data, axis=1, keepdims=True)
    diff = max_values - min_values

    zero_division_mask = diff == 0
    if np.any(zero_division_mask):
        print("Division by zero encountered; setting affected rows to 0.")

    normalized_data = np.where(zero_division_mask, 0, (data - min_values) / diff)
    return normalized_data


def main() -> None:
    # CLI (optional). If any flags are missing, fall back to interactive prompts
    parser = argparse.ArgumentParser(description="Time-domain statistical PSD entry point (interactive by default; CLI flags optional)")
    parser.add_argument("--method", type=str, help="Method (CC, CI, FEPS, GP, LLR, LMT, PCA, PGA, PR, ZC)")
    parser.add_argument("--filter", type=int, choices=list(range(0, 12)), help="Filter id (0-11)")
    args = parser.parse_args()
    # Determine CLI mode (any flag provided)
    cli_mode = any(v is not None for v in vars(args).values())
    if cli_mode:
        # Disable interactive plots globally
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.show = lambda *a, **k: None  # no-op
        except Exception:
            pass
    # Load data
    data_file = os.path.join('Data', 'Validation', 'EJ299_33_AmBe_9414.txt')
    pulse_signal_original = load_dataset(data_file)

    # Assume a sampling frequency of 1000 Hz for frequency-based filters
    fs = 1000

    # Define filter options with names and functions (with default parameters)
    filter_options = {
        '0': ('No Filter', lambda x: x),
        '1': ('Butterworth Filter', partial(filters.butterworth_filter, order=5, cutoff_freq=200, fs=fs, btype='low')),
        '2': ('Chebyshev Filter', partial(filters.chebyshev_filter, order=5, ripple=1, cutoff_freq=200, fs=fs, btype='low')),
        '3': ('Elliptic Filter', partial(filters.elliptic_filter, order=5, rp=1, rs=40, cutoff_freq=200, fs=fs, btype='low')),
        '4': ('Fourier Filter', partial(filters.fourier_filter, cutoff_freq=200, fs=fs)),
        '5': ('Least Mean Square Adaptive Filter', partial(filters.least_mean_square_filter, mu=0.01, order=10)),
        '6': ('Median Filter', partial(filters.median_filter, kernel_size=3)),
        '7': ('Morphological Filter', partial(filters.morphological_filter, size=3)),
        '8': ('Moving Average Filter', partial(filters.moving_average_filter, window_size=5)),
        '9': ('Wavelet Filter', partial(filters.wavelet_filter, wavelet='db4', level=2, threshold=0.01)),
        '10': ('Wiener Filter', partial(filters.wiener_filter, mysize=5)),
        '11': ('Windowed-Sinc Filter', partial(filters.windowed_sinc_filter, cutoff_freq=200, fs=fs, numtaps=51, window='hamming')),
    }

    # Prompt user to select a filter (unless provided by CLI)
    if args.filter is None:
        print("Select a filter to apply:")
        for key, (name, _) in filter_options.items():
            print(f"{key}: {name}")
        choice = input("Enter the number of the filter (0 for no filter): ")
        if choice not in filter_options:
            print("\nInvalid choice. Proceeding with no filter.")
            choice = '0'
        else:
            print("\nStart filtering...")
    else:
        choice = str(args.filter)
        if choice not in filter_options:
            print("\nInvalid --filter provided. Proceeding with no filter.")
            choice = '0'

    # Get the selected filter function and apply it
    selected_filter = filter_options[choice][1]
    filtered_data = selected_filter(pulse_signal_original)

    # Normalize the filtered data
    normalized_data = normalize(filtered_data)

    # List of available methods and mapping to module paths
    available_methods = [
        'CC   - Charge Comparison',
        'CI   - Charge Integration',
        'FEPS - Falling-Edge Percentage Slope',
        'GP   - Gatti Parameter',
        'LLR  - Log-Likelihood Ratio',
        'LMT  - Log of Mean Time',
        'PCA  - Principal Component Analysis',
        'PGA  - Pulse Gradient Analysis',
        'PR   - Pattern Recognition',
        'ZC   - Zero Crossing',
    ]
    BASE_PATH = 'Method.Statistical_methods.Time_domain'
    method_modules = {
        'CC': f'{BASE_PATH}.cc',
        'CI': f'{BASE_PATH}.ci',
        'FEPS': f'{BASE_PATH}.feps',
        'GP': f'{BASE_PATH}.gp',
        'LLR': f'{BASE_PATH}.llr',
        'LMT': f'{BASE_PATH}.lmt',
        'PCA': f'{BASE_PATH}.pca',
        'PGA': f'{BASE_PATH}.pga',
        'PR': f'{BASE_PATH}.pr',
        'ZC': f'{BASE_PATH}.zc',
    }

    pulse_shape_discrimination_factor = None

    while pulse_shape_discrimination_factor is None:
        if args.method is None:
            print('Select a PSD method from the following options:')
            print('\n'.join(available_methods))
            method_name = input('Enter the method name (e.g., pr): ').upper().strip()
        else:
            method_name = args.method.strip().upper()

        if method_name in method_modules:
            module_name = method_modules[method_name]
            print(f"\nAttempting to import module: {module_name}")
            try:
                method_module = importlib.import_module(module_name)
                if hasattr(method_module, 'get_psd_factor'):
                    print(f"\nComputing with {method_name} method...")
                    validation_start = time.time()
                    pulse_shape_discrimination_factor = method_module.get_psd_factor(normalized_data)
                    validation_end = time.time()
                    print(f"Validation process took {validation_end - validation_start:.4f} seconds.")
                else:
                    print(f"Error: Module {module_name} does not have a 'get_psd_factor' function")
            except ImportError as e:
                print(f"Error: Could not import {module_name}. Details: {e}")
            except Exception as e:
                print(f"Error during computation with {method_name}: {e}")
        else:
            print(f"Invalid method name: {method_name}. Available methods: {', '.join(method_modules.keys())}")
            print()

    os.makedirs('Output/Validation_results', exist_ok=True)

    if pulse_shape_discrimination_factor is not None:
        # Normalize Validation_pred to range [0, 1]
        pulse_shape_discrimination_factor = (
            pulse_shape_discrimination_factor - np.min(pulse_shape_discrimination_factor)
        ) / (
            np.max(pulse_shape_discrimination_factor) - np.min(pulse_shape_discrimination_factor)
        )
        miu, sigma, fom = histogram_fitting_compute_fom(pulse_shape_discrimination_factor,  method_name, show_plot=(not cli_mode))
        print(f"Validation set PSD factors computed. Figure of Merit (FOM): {fom}")
        print(f"Sample of Validation PSD factors: {pulse_shape_discrimination_factor[:5]}")
        np.savetxt(f'Output/Validation_results/{method_name}.txt', pulse_shape_discrimination_factor, fmt='%1.6f')
        print(f"Validation results saved to 'Output/Validation_results/{method_name}.txt'.")
    else:
        print("No valid method was successfully executed.")


if __name__ == "__main__":
    main()
