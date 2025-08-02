import os
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
        '9': ('Wavelet Filter', partial(filters.wavelet_filter, wavelet='db4', level=4, threshold=0.2)),
        '10': ('Wiener Filter', partial(filters.wiener_filter, mysize=5)),
        '11': ('Windowed-Sinc Filter', partial(filters.windowed_sinc_filter, cutoff_freq=200, fs=fs, numtaps=51, window='hamming')),
    }

    # Prompt user to select a filter
    print("Select a filter to apply:")
    for key, (name, _) in filter_options.items():
        print(f"{key}: {name}")
    choice = input("Enter the number of the filter (0 for no filter): ")

    # Validate the choice; default to no filter if invalid
    if choice not in filter_options:
        print("\nInvalid choice. Proceeding with no filter.")
        choice = '0'
    else:
        print("\nStart filtering...")

    # Get the selected filter function and apply it
    selected_filter = filter_options[choice][1]
    filtered_data = selected_filter(pulse_signal_original)

    # Normalize the filtered data
    normalized_data = normalize(filtered_data)

    # List of available neural network methods and mapping to module paths
    available_methods = [
        'HQC  - Heterogeneous Quasi-Continuous Spiking Cortical Model',
        'LG   - Ladder Gradient',
        'PCNN - Pulse-Coupled Neural Network',
        'RCNN - Random-Coupled Neural Network',
        'SCM  - Spiking Cortical Model',
    ]
    BASE_PATH = 'Method.Statistical_methods.Neural_network'
    method_modules = {
        'HQC': f'{BASE_PATH}.hqc',
        'LG': f'{BASE_PATH}.lg',
        'PCNN': f'{BASE_PATH}.pcnn',
        'RCNN': f'{BASE_PATH}.rcnn',
        'SCM': f'{BASE_PATH}.scm',
    }

    pulse_shape_discrimination_factor = None

    while pulse_shape_discrimination_factor is None:
        print('Select a neural network method from the following options:')
        print('\n'.join(available_methods))
        method_name = input('Enter the method name (e.g., lg): ').upper().strip()

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
        miu, sigma, fom = histogram_fitting_compute_fom(pulse_shape_discrimination_factor, method_name, show_plot=True)
        print(f"Validation set PSD factors computed. Figure of Merit (FOM): {fom}")
        print(f"Sample of Validation PSD factors: {pulse_shape_discrimination_factor[:5]}")
        np.savetxt(f'Output/Validation_results/{method_name}.txt', pulse_shape_discrimination_factor, fmt='%1.6f')
        print(f"Validation results saved to 'Output/Validation_results/{method_name}.txt'.")
    else:
        print("No valid method was successfully executed.")

if __name__ == "__main__":
    main()
