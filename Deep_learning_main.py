import os
import argparse
import numpy as np
import importlib
import time
import Utility.filters as filters
from functools import partial
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom


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
        ValueError: If input is not a numpy array or is empty.
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
    parser = argparse.ArgumentParser(description="Deep Learning PSD entry point (interactive by default; CLI flags optional)")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], help="Task type")
    parser.add_argument("--method", type=str, help="Deep learning method (e.g., MLP3, CNNDEEP, TRAN, MAM, ...)")
    parser.add_argument("--feat", type=str, help="Feature extractor when task=regression (e.g., CC, GP, SD, ...)")
    parser.add_argument("--filter", type=int, choices=list(range(0, 12)), help="Filter id (0-11)")
    parser.add_argument("--train", type=str, choices=["yes", "no"], help="Whether to train a new model")
    parser.add_argument("--validate", type=str, choices=["yes", "no"], help="Whether to run validation evaluation")
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
    # Load and normalize data
    data_train_1 = load_dataset(os.path.join('Data', 'Train', 'EJ299_33_AmBe_9414_neutron_train.txt'))
    data_train_2 = load_dataset(os.path.join('Data', 'Train', 'EJ299_33_AmBe_9414_gamma_train.txt'))
    data_test_1 = load_dataset(os.path.join('Data', 'Test', 'EJ299_33_AmBe_9414_neutron_test.txt'))
    data_test_2 = load_dataset(os.path.join('Data', 'Test', 'EJ299_33_AmBe_9414_gamma_test.txt'))
    data_validation = load_dataset(os.path.join('Data', 'Validation', 'EJ299_33_AmBe_9414.txt'))

    # Assume a sampling frequency of 1000 Hz for frequency-based filters. We do not use typical
    # radiation pulse signal sample frequencies of several MS/s or GS/s here to avoid generating 
    # excessively large numbers of frequencies for frequency-based filters.
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

    # Get the selected filter function
    selected_filter = filter_options[choice][1]

    # Apply the selected filter
    filtered_train_1 = selected_filter(data_train_1)
    filtered_train_2 = selected_filter(data_train_2)
    filtered_test_1 = selected_filter(data_test_1)
    filtered_test_2 = selected_filter(data_test_2)
    filtered_validation = selected_filter(data_validation)

    # Normalize the filtered data
    normalized_train_1 = normalize(filtered_train_1)
    normalized_train_2 = normalize(filtered_train_2)
    normalized_test_1 = normalize(filtered_test_1)
    normalized_test_2 = normalize(filtered_test_2)
    normalized_validation = normalize(filtered_validation)

    # Prepare training and test data with labels
    Train_data = np.vstack((normalized_train_1, normalized_train_2))
    Train_labels = np.concatenate((np.zeros(len(normalized_train_1)), np.ones(len(normalized_train_2))))
    Test_data = np.vstack((normalized_test_1, normalized_test_2))
    Test_labels = np.concatenate((np.zeros(len(normalized_test_1)), np.ones(len(normalized_test_2))))

    # Define available PSD methods and feature extractors
    BASE_PATH = 'Method.Prior_knowledge_methods.Deep_learning'
    TIME_FEAT_PATH = 'Method.Statistical_methods.Time_domain'
    FREQ_FEAT_PATH = 'Method.Statistical_methods.Frequency_domain'
    NURAL_FEAT_PATH = 'Method.Statistical_methods.Neural_network'
    available_psd_methods = [
        'CNNDEEP  - Convolutional Neural Network with Deep Layers',
        'CNNFT    - Convolutional Neural Network with Fourier Transform',
        'CNNSHAL  - 1D Convolutional Neural Network with Shallow Layers',
        'CNNSP    - Convolutional Neural Network with Snapshot',
        'CNNSTFT  - Convolutional Neural Network with Short Time Fourier Transform',
        'CNNWT    - Convolutional Neural Network with Wavelet Transform',
        'ENN      - Elman Neural Network',
        'GRU      - Gated Recurrent Unit',
        'LSTM     - Long Short-Term Memory',
        'MAM      - Mamba Network',
        'MLP1     - Single Layer Perceptron',
        'MLP1FT   - Single Layer Perceptron with Fourier Transform',
        'MLP1PCA  - Single Layer Perceptron with Principal Components Analysis',
        'MLP1STFT - Single Layer Perceptron with Short Time Fourier Transform',
        'MLP1WT   - Single Layer Perceptron with Wavelet Transform',
        'MLP2     - Dense Neural Networks',
        'MLP2FT   - Dense Neural Networks with Fourier Transform',
        'MLP2PCA  - Dense Neural Networks with Principal Components Analysis',
        'MLP2STFT - Dense Neural Networks with Short Time Fourier Transform',
        'MLP2WT   - Dense Neural Networks with Wavelet Transform',
        'MLP3     - MultiLayer Perceptron',
        'MLP3FT   - MultiLayer Perceptron with Fourier Transform',
        'MLP3PCA  - MultiLayer Perceptron with Principal Components Analysis',
        'MLP3STFT - MultiLayer Perceptron with Short Time Fourier Transform',
        'MLP3WT   - MultiLayer Perceptron with Wavelet Transform',
        'RNN      - Recurrent Neural Network',
        'TRAN     - Transformer Network'
    ]
    psd_method_modules = {
        'CNNDEEP': f'{BASE_PATH}.cnndeep',
        'CNNFT': f'{BASE_PATH}.cnnft',
        'CNNSHAL': f'{BASE_PATH}.cnnshal',
        'CNNSP': f'{BASE_PATH}.cnnsp',
        'CNNSTFT': f'{BASE_PATH}.cnnstft',
        'CNNWT': f'{BASE_PATH}.cnnwt',
        'ENN': f'{BASE_PATH}.enn',
        'GRU': f'{BASE_PATH}.gru',
        'LSTM': f'{BASE_PATH}.lstm',
        'MAM': f'{BASE_PATH}.mamba',
        'MLP1': f'{BASE_PATH}.mlp1',
        'MLP1FT': f'{BASE_PATH}.mlp1ft',
        'MLP1PCA': f'{BASE_PATH}.mlp1pca',
        'MLP1STFT': f'{BASE_PATH}.mlp1stft',
        'MLP1WT': f'{BASE_PATH}.mlp1wt',
        'MLP2': f'{BASE_PATH}.mlp2',
        'MLP2FT': f'{BASE_PATH}.mlp2ft',
        'MLP2PCA': f'{BASE_PATH}.mlp2pca',
        'MLP2STFT': f'{BASE_PATH}.mlp2stft',
        'MLP2WT': f'{BASE_PATH}.mlp2wt',
        'MLP3': f'{BASE_PATH}.mlp3',
        'MLP3FT': f'{BASE_PATH}.mlp3ft',
        'MLP3PCA': f'{BASE_PATH}.mlp3pca',
        'MLP3STFT': f'{BASE_PATH}.mlp3stft',
        'MLP3WT': f'{BASE_PATH}.mlp3wt',
        'RNN': f'{BASE_PATH}.rnn',
        'TRAN': f'{BASE_PATH}.transformer'
    }
    available_feature_extractors = [
        '=== Time Domain Extractors ===',
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
        '',
        '=== Frequency Domain Extractors ===',
        'FGA  - Frequency Gradient Analysis',
        'SDCC - Simplified Digital Charge Collection',
        'DFT  - Discrete Fourier Transform',
        'WT1  - Wavelet Transform with signal ratio',
        'WT2  - Wavelet Transform with signal integration',
        'FS   - Fractal Spectrum',
        'SD   - Scalogram-based Discrimination',
        '',
        '=== Neural Network Extractors ===',
        'HQC  - Heterogeneous Quasi-Continuous Spiking Cortical Model',
        'LG   - Ladder Gradient',
        'PCNN - Pulse-Coupled Neural Network',
        'RCNN - Random-Coupled Neural Network',
        'SCM  - Spiking Cortical Model',
    ]
    feature_extractor_modules = {
        'CC': f'{TIME_FEAT_PATH}.cc',
        'CI': f'{TIME_FEAT_PATH}.ci',
        'FEPS': f'{TIME_FEAT_PATH}.feps',
        'GP': f'{TIME_FEAT_PATH}.gp',
        'LLR': f'{TIME_FEAT_PATH}.llr',
        'LMT': f'{TIME_FEAT_PATH}.lmt',
        'PCA': f'{TIME_FEAT_PATH}.pca',
        'PGA': f'{TIME_FEAT_PATH}.pga',
        'PR': f'{TIME_FEAT_PATH}.pr',
        'ZC': f'{TIME_FEAT_PATH}.zc',
        'FGA': f'{FREQ_FEAT_PATH}.fga',
        'SDCC': f'{FREQ_FEAT_PATH}.sdcc',
        'DFT': f'{FREQ_FEAT_PATH}.dft',
        'WT1': f'{FREQ_FEAT_PATH}.wt1',
        'WT2': f'{FREQ_FEAT_PATH}.wt2',
        'FS': f'{FREQ_FEAT_PATH}.fs',
        'SD': f'{FREQ_FEAT_PATH}.sd',
        'HQC': f'{NURAL_FEAT_PATH}.hqc',
        'LG': f'{NURAL_FEAT_PATH}.lg',
        'PCNN': f'{NURAL_FEAT_PATH}.pcnn',
        'RCNN': f'{NURAL_FEAT_PATH}.rcnn',
        'SCM': f'{NURAL_FEAT_PATH}.scm',
    }

    # Choose task (CLI or interactive)
    print('\nChoose the task: classification or regression:')
    print('Classification: Predict a class label (0 or 1) for each pulse signal.')
    print('Regression: Predict a PSD factor for each pulse signal.')
    task_mapping = {
        "classification": "classification",
        "class": "classification",
        "clf": "classification",
        "c": "classification",
        "regression": "regression",
        "reg": "regression",
        "regr": "regression",
        "r": "regression"
    }
    if args.task is None:
        task = input('Enter "classification" (or "c") or "regression" (or "r"): ').strip().lower()
        while task not in task_mapping:
            print('Invalid choice.')
            task = input('Enter "classification" (or "c") or "regression" (or "r"): ').strip().lower()
    else:
        task = args.task.strip().lower()
        if task not in task_mapping:
            print(f"Invalid --task '{args.task}'.")
            return
    task = task_mapping[task]

    # If regression, choose feature extractor for training PSD factors
    if task == 'regression':
        if args.feat is None:
            print('\nSelect a feature extractor to compute PSD factors for training:')
            print('\n'.join(available_feature_extractors))
            feat_name = input('Enter the feature extractor name (e.g., cc): ').strip().upper()
            while feat_name not in feature_extractor_modules:
                print(f"Invalid name. Available extractors: {', '.join(feature_extractor_modules.keys())}")
                feat_name = input('Enter the feature extractor name: ').strip().upper()
        else:
            feat_name = args.feat.strip().upper()
            if feat_name not in feature_extractor_modules:
                print(f"Invalid --feat '{args.feat}'. Available: {', '.join(feature_extractor_modules.keys())}")
                return
        feat_module_name = feature_extractor_modules[feat_name]
        try:
            feat_module = importlib.import_module(feat_module_name)
        except ImportError as e:
            print(f"Error: Could not import {feat_module_name}. Details: {e}")
            return
        if not hasattr(feat_module, 'get_psd_factor'):
            print(f"Error: {feat_module_name} does not implement 'get_psd_factor' function.")
            return
    else:
        feat_name = None

    # Choose PSD method
    if args.method is None:
        print('\nSelect a PSD method from the following options:')
        print('\n'.join(available_psd_methods))
        method_name = input('Enter the method name (e.g., mlp1): ').strip().upper()
        while method_name not in psd_method_modules:
            print(f"Invalid method name. Available methods: {', '.join(psd_method_modules.keys())}")
            method_name = input('Enter the method name: ').strip().upper()
    else:
        method_name = args.method.strip().upper()
        if method_name not in psd_method_modules:
            print(f"Invalid --method '{args.method}'. Available: {', '.join(psd_method_modules.keys())}")
            return
    module_name = psd_method_modules[method_name]
    try:
        method_module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error: Could not import {module_name}. Details: {e}")
        return

    # Check if the method supports the selected task
    if not hasattr(method_module, 'get_supported_tasks'):
        print(f"Error: {module_name} must implement 'get_supported_tasks()' to specify supported tasks.")
        return
    supported_tasks = method_module.get_supported_tasks()
    if task not in supported_tasks:
        print(f"Error: {method_name} does not support {task}.")
        print(f"Supported tasks: {', '.join(supported_tasks)}")
        return

    # Train or load model
    print('\nTrain a new model or load a pretrained model?')
    choice_mapping = {
        "yes": "yes",
        "y": "yes",
        "no": "no",
        "n": "no"
    }
    if args.train is None:
        choice = input('Is training required? (yes/y/no/n): ').strip().lower()
        while choice not in choice_mapping:
            print('Invalid choice. Please enter "yes", "y", "no", or "n".')
            choice = input('Is training required? (yes/y/no/n): ').strip().lower()
        choice = choice_mapping[choice]
    else:
        choice = args.train.strip().lower()
        if choice not in choice_mapping:
            print(f"Invalid --train '{args.train}'. Use yes or no.")
            return
    if choice == 'yes':
        if not hasattr(method_module, 'train'):
            print(f"Error: {module_name} does not implement a 'train' function.")
            return
        print('\nTraining the model...', flush=True)
        if task == 'classification':
            labels = Train_labels  # 0 and 1 for classification
        else:  # regression
            labels = feat_module.get_psd_factor(Train_data)  # PSD factors for regression
            
            # Check for NaN values in labels
            nan_mask = np.isnan(labels)
            nan_count = np.sum(nan_mask)
            total_signals = len(labels)
            if nan_count > 0:
                nan_percentage = (nan_count / total_signals) * 100
                print(f"Warning: {nan_count} signals ({nan_percentage:.2f}% of total {total_signals} signals) have NaN label values.")
                print("Removing samples with NaN labels...")
                # Remove samples with NaN labels
                valid_mask = ~nan_mask
                Train_data = Train_data[valid_mask]
                labels = labels[valid_mask]
                print(f"Remaining samples after NaN removal: {len(labels)}")
            
            labels = (labels - labels.min()) / (labels.max() - labels.min()) # Normalize labels for loss function requirements
            
        method_module.train(Train_data, labels, task=task, feat_name=feat_name)
    else:
        if not hasattr(method_module, 'load_model'):
            print(f"Error: {module_name} does not implement a 'load_model' function.")
            return
        print('\nLoading pre-trained model...', flush=True)
        method_module.load_model(task=task, feat_name=feat_name)

    # Test set evaluation
    print('\nEvaluating on the test set...', flush=True)
    Test_pred = method_module.test(Test_data, task=task, feat_name=feat_name)
    if task == 'classification':
        accuracy = np.mean(Test_pred == Test_labels)
        print("Test set accuracy: {:.2f}%".format(accuracy * 100))
    else:  # regression
        miu, sigma, fom = histogram_fitting_compute_fom(Test_pred, method_name + "_test", show_plot=(not cli_mode))
        print(f"Test set PSD factors computed. Figure of Merit (FOM): {fom}")
        print(f"Sample of Test PSD factors: {Test_pred[:5]}")

    # Optional validation set evaluation
    choice_mapping = {
        "yes": "yes",
        "y": "yes",
        "no": "no",
        "n": "no"
    }
    if args.validate is None:
        choice = input('\nPerform evaluation on the validation set? (yes/y/no/n): ').strip().lower()
        while choice not in choice_mapping:
            print('Invalid choice. Please enter "yes", "y", "no", or "n".')
            choice = input('Perform evaluation on the validation set? (yes/y/no/n): ').strip().lower()
        choice = choice_mapping[choice]
    else:
        choice = args.validate.strip().lower()
        if choice not in choice_mapping:
            print(f"Invalid --validate '{args.validate}'. Use yes or no.")
            return
    if choice == 'yes':
        print('\nEvaluating on the validation set...', flush=True)
        validation_start = time.time()
        Validation_pred = method_module.test(normalized_validation, task=task, feat_name=feat_name)
        validation_end = time.time()
        print(f"Validation process took {validation_end - validation_start:.4f} seconds.")
        os.makedirs('Output/Validation_results', exist_ok=True)
        if task == 'classification':
            Validation_labels = Validation_pred.astype(int)
            print("Validation class labels computed.")
            print(f"Sample of Validation class labels: {Validation_labels[:5]}")
            np.savetxt(f'Output/Validation_results/{method_name}_{task}.txt', Validation_labels, fmt='%d')
            print(f"Validation results saved to 'Output/Validation_results/{method_name}_{task}.txt'.")
        else:  # regression
            # Normalize Validation_pred to range [0, 1]
            Validation_pred = (Validation_pred - np.min(Validation_pred)) / (np.max(Validation_pred) - np.min(Validation_pred))
            miu, sigma, fom = histogram_fitting_compute_fom(Validation_pred, method_name, show_plot=(not cli_mode))
            print(f"Validation set PSD factors computed. Figure of Merit (FOM): {fom}")
            print(f"Sample of Validation PSD factors: {Validation_pred[:5]}")
            np.savetxt(f'Output/Validation_results/{method_name}_{task}_{feat_name}.txt', Validation_pred, fmt='%1.6f')
            print(f"Validation results saved to 'Output/Validation_results/{method_name}_{task}_{feat_name}.txt'.")    
    else:
        print('\nSkipping validation set evaluation.')


if __name__ == "__main__":
    main()
