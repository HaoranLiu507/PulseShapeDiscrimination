# Pulse Shape Discrimination (PSD)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="Taxonomy Overview.png" alt="Taxonomy Overview" width="2000">
</p>

## Table of Contents
- [Overview](#overview)
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Quickstart (Python, example)](#quickstart-python-example)
- [Methods](#methods)
- [Extending the toolbox](#extending-the-toolbox)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The Pulse Shape Discrimination (PSD) project is a comprehensive Python/MATLAB toolbox for discriminating between different types of radiation particles based on their pulse shapes in scintillation detector signals. PSD is critical in radiation detection for nuclear security, medical physics, and scientific research.

- If you find our work useful in your research or publication, please cite our work:

- Haoran Liu, Yihan Zhan, Mingzhe Liu, Yanhua Liu, Peng Li, Zhuo Zuo, Bingqi Liu, and Runxi Liu. **"Pulse Shape Discrimination Algorithms: Survey and Benchmark."** *arXiv preprint* arXiv:2508.02750, 2025. [[arXiv](https://arxiv.org/abs/2508.02750)]

## Project Description

This toolbox implements a broad collection of PSD algorithms spanning statistical time/frequency-domain methods, neural models, classic machine learning, and modern deep learning. Neutron vs. gamma discrimination is provided as a default demo. With your own datasets, other particle types can be used as well. For detailed discussions and benchmarking across methods, see our survey: [[arXiv](https://arxiv.org/abs/2508.02750)].

### Key Features

- **Multiple methodologies**: Time-domain, frequency-domain, neural network (spiking), machine learning, and deep learning
- **Flexible pre-processing**: Built-in filters (Butterworth, Chebyshev, Elliptic, Fourier, LMS, Median, Morphological, Moving Average, Wavelet, Wiener, Windowed-Sinc)
- **Comprehensive evaluation**: Figure of Merit (FOM) computation via histogram fitting
- **Modular design**: Clean extension points for new algorithms
- **Two task types**: Classification (labels) and regression (PSD factor)
- **MATLAB support**: MATLAB implementations for time/frequency-domain and spiking neural methods

## Project Structure

The project is organized as follows (directories like `Data/` and `Output/` are created or populated at runtime or by downloading the companion dataset):

```
PulseShapeDiscrimination/
├── Data/                      # Input data (see Dataset section)
│   ├── Reference_signal/      # Reference signals for GP and LLR
│   ├── Test/                  # Testing datasets
│   ├── Train/                 # Training datasets
│   └── Validation/            # Validation datasets
│
├── MATLAB/                    # MATLAB implementations
│   ├── Method/                # Statistical & spiking neural methods
│   ├── Utility/               # MATLAB utilities
│   ├── Frequency_domain_main.m
│   ├── Neural_network_main.m
│   └── Time_domain_main.m
│
├── Method/                    # All PSD methodologies (Python)
│   ├── Prior_knowledge_methods/
│   │   ├── Deep_learning/     # CNN, LSTM, MLP, Transformer, etc.
│   │   └── Machine_learning/  # SVM, KNN, DT, GMM, LVQ, etc.
│   └── Statistical_methods/
│       ├── Frequency_domain/  # WT, DFT, FGA, FS, SD, SDCC
│       ├── Neural_network/    # SCM, PCNN, RCNN, HQC, LG
│       └── Time_domain/       # CC, CI, GP, LLR, LMT, PR, ZC, PCA, PGA, FEPS
│
├── Output/                    # Created at runtime for results
│   ├── Trained_models/        # Saved models (method-dependent)
│   └── Validation_results/    # Validation outputs (.txt, plots)
│
├── Utility/                   # Utilities (Python)
│   ├── filters.py             # Signal filtering
│   ├── histogram_fitting_compute_fom.py # FOM calculation
│   └── Tempotron.py           # Tempotron neural model
│
├── Deep_learning_main.py
├── Frequency_domain_main.py
├── Machine_learning_main.py
├── Neural_network_main.py
├── Time_domain_main.py
├── README.md
└── requirements.txt
```

## Dataset

The companion dataset, pre-trained models, and experimental results are hosted on Zenodo and are required to reproduce the figures and benchmarks in the survey paper.

- Download: [Zenodo DOI](https://doi.org/10.5281/zenodo.16728732)
- After download, extract or place the folders so they appear under `PulseShapeDiscrimination/Data/` (and optionally `Output/` if provided). The default Python scripts expect these files:
  - `Data/Train/EJ299_33_AmBe_9414_neutron_train.txt`
  - `Data/Train/EJ299_33_AmBe_9414_gamma_train.txt`
  - `Data/Test/EJ299_33_AmBe_9414_neutron_test.txt`
  - `Data/Test/EJ299_33_AmBe_9414_gamma_test.txt`
  - `Data/Validation/EJ299_33_AmBe_9414.txt`
  - `Data/Reference_signal/EJ299_33_AmBe_9414_neutron_ref.txt` (for GP/LLR)
  - `Data/Reference_signal/EJ299_33_AmBe_9414_gamma_ref.txt` (for GP/LLR)

Data format: each `.txt` file is a numeric matrix with shape `(num_signals, num_samples)`, one pulse waveform per row.

MATLAB note: the `.m` demos expect `EJ299_33_AmBe_9414.txt` to be on the MATLAB path or in the same folder as the `.m` file; adjust the `importdata` path in `MATLAB/*_main.m` if you keep data under `Data/`.

## Installation

### Prerequisites
- Python 3.12 or higher
- MATLAB R2023a or higher (only for MATLAB implementations)
- Optional: CUDA toolkit for GPU-accelerated deep learning

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/HaoranLiu507/PulseShapeDiscrimination.git
   cd PulseShapeDiscrimination
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch (optional unless using deep learning), and any optional extras:
   ```bash
   # For the version used in development (with CUDA 12.6):
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

   # Note: If you have different hardware requirements, please visit:
   # https://pytorch.org/get-started/locally/
   # to select the appropriate PyTorch installation command for your system
   ```

   Optional extras:
   - `mamba_ssm` for the Mamba model: `pip install mamba-ssm`

   Development versions used:
   - torch==2.6.0, torchvision==0.21.0

5. Download the companion dataset from Zenodo

   - **Download link**: [Zenodo DOI](https://doi.org/10.5281/zenodo.16728732)
   - Contents include: datasets (AmBe, PuBe), pre-computed results, and per-dataset PSD parameters.
   - Place the extracted `Data/` folder into the project root. Refer to Dataset section above.

   See the Zenodo `README.txt` for additional details.

### Notes on plotting backends

The FOM plotting utility will try to use Matplotlib's `TkAgg` backend and automatically fall back to `Agg` if `TkAgg` is unavailable. In most cases you do not need to change the backend manually.

- If you are running on a headless system or prefer non-interactive behavior, use the CLI flags so that the main scripts disable interactive plotting and save FOM figures automatically during test/validation, or call `histogram_fitting_compute_fom(..., show_plot=False)` directly.
- Saved figures are written to the current working directory as `fom_plot_{method_name}.jpg`.
- Note: because the code sets the backend explicitly, an environment variable like `MPLBACKEND=Agg` may be overridden by the script.

### Notes on Tempotron (GPU-only)

The Tempotron implementation requires a CUDA-capable NVIDIA GPU and a CUDA-enabled PyTorch build. CPU (and Apple `mps`) execution is not supported.

- Ensure `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
- Install the correct CUDA wheel for your system (see PyTorch site), and make sure the NVIDIA driver is installed.
- The low-level class in `Utility/Tempotron.py` uses `cuda:0` by default. If you need to select another GPU, set `CUDA_VISIBLE_DEVICES` accordingly or adjust the file locally.

## Usage

The project provides separate main scripts for different PSD methodologies. All Python entry points are interactive and will prompt for filter choice, task (where applicable), method, and whether to train or load a model.

### Statistical Methods (Time Domain)
```bash
python Time_domain_main.py
```

### Statistical Methods (Frequency Domain)
```bash
python Frequency_domain_main.py
```

### Statistical Methods (Neural Network)
```bash
python Neural_network_main.py
```

### Machine Learning Methods
```bash
python Machine_learning_main.py
```

### Deep Learning Methods
```bash
python Deep_learning_main.py
```

### MATLAB
Open MATLAB, navigate to the `MATLAB/` directory, and run a main script. For example:
```matlab
run('Time_domain_main.m');
```
Repeat similarly for `Frequency_domain_main.m` and `Neural_network_main.m`.

### Quickstart (Python, example)

The following transcript shows a typical deep learning classification run:

1) Run: `python Deep_learning_main.py`
2) When prompted for filter: enter `0` (no filter) for a first run
3) Task: enter `c` for classification
4) Method: enter `MLP1` (or try `CNNDEEP`, `GRU`, etc.)
5) Train? enter `y` to train a new model
6) Validation? optional; enter `n` to skip on first run

Results and any saved outputs will appear under `Output/` (method-dependent). For regression tasks, you will additionally be asked to select a feature extractor used to compute target PSD factors.

### Command-line (non-interactive) usage

All main scripts now accept CLI flags. If a required flag is missing, the script will prompt interactively for that item only.

- Deep learning (`Deep_learning_main.py`)
  - Flags: `--task {classification,regression}`, `--method <MLP1|MLP2|MLP3|CNNDEEP|CNNFT|CNNSHAL|CNNSP|CNNSTFT|CNNWT|ENN|GRU|LSTM|RNN|TRAN|MAM>`, `--feat <CC|GP|...>` (when task=regression), `--filter {0..11}`, `--train {yes,no}`, `--validate {yes,no}`
  - Example (classification, train, no validation):
    ```bash
    python Deep_learning_main.py --task classification --method MLP1 --filter 0 --train yes --validate no
    ```
  - Example (regression with CC features, train and validate):
    ```bash
    python Deep_learning_main.py --task regression --feat CC --method CNNDEEP --filter 1 --train yes --validate yes
    ```

- Machine learning (`Machine_learning_main.py`)
  - Flags: `--task {classification,regression}`, `--method <BDT|DT|FCM|GMM|KNN|LINRE|LOGRE|LRSTFT|LVQ|SVM|TEM>`, `--feat <...>` (when task=regression), `--filter {0..11}`, `--train {yes,no}`, `--validate {yes,no}`
  - Example:
    ```bash
    python Machine_learning_main.py --task classification --method SVM --filter 0 --train yes --validate no
    ```

- Time-domain statistical (`Time_domain_main.py`)
  - Flags: `--method <CC|CI|FEPS|GP|LLR|LMT|PCA|PGA|PR|ZC>`, `--filter {0..11}`
  - Example:
    ```bash
    python Time_domain_main.py --method PR --filter 0
    ```

- Frequency-domain statistical (`Frequency_domain_main.py`)
  - Flags: `--method <FGA|SDCC|DFT|WT1|WT2|FS|SD>`, `--filter {0..11}`
  - Example:
    ```bash
    python Frequency_domain_main.py --method SD --filter 9
    ```

- Spiking neural models (`Neural_network_main.py`)
  - Flags: `--method <HQC|LG|PCNN|RCNN|SCM>`, `--filter {0..11}`, `--validate {yes,no}`
  - Example:
    ```bash
    python Neural_network_main.py --method SCM --filter 0 --validate yes
    ```

Notes:
- When any CLI flag is provided, interactive plotting is disabled. The entry scripts save FOM figures to files during test/validation. Some method modules may still attempt plots during training; those are suppressed in CLI mode and may not be saved unless the module passes `show_plot=False`.
- You can always run `python <main>.py --help` to see the available flags.

Filter IDs (0–11):
- 0: No Filter
- 1: Butterworth
- 2: Chebyshev (Type I)
- 3: Elliptic
- 4: Fourier (low‑pass in frequency domain)
- 5: Least Mean Square (LMS) adaptive
- 6: Median
- 7: Morphological (open/close)
- 8: Moving Average
- 9: Wavelet (db4)
- 10: Wiener
- 11: Windowed‑Sinc (FIR)

Sampling frequency note: for demonstration purposes the main scripts set `fs = 1000` Hz to parameterize filters. If your signals were sampled at a different rate, adjust `fs` (and filter parameters) near the top of each `*_main.py`.

### General Workflow

1. **Choose entry point**: one of the main scripts per method family
2. **Filtering**: optional signal denoising/filter selection
3. **Task**: choose classification or regression (ML/DL only)
4. **Method**: choose a PSD algorithm from the prompts
5. **Train or load**: ML/DL scripts support training or loading
6. **Evaluate**: automatic test-set evaluation; optional validation-set evaluation
7. **Analyze results**: view metrics, figures, and any saved outputs

**Notes:**
- MATLAB implementations are provided for statistical methods; ML/DL methods are Python-only.
- Python implementations are generally more feature-complete (filters, visualization).
- Default dataset is AmBe. To switch datasets, update the file paths near the top of each `*_main.py`.
- Some methods require additional data: GP and LLR need reference signals under `Data/Reference_signal/`, and SD builds a discrimination mask from training data.

### Non-interactive usage

Use the CLI flags described above to run in non-interactive mode. Missing flags will be prompted.

## Methods

### Time Domain Methods
- **Charge Comparison (CC)**: Compares charge in the slow component to total charge using integration windows
- **Charge Integration (CI)**: Uses trapezoidal integration to compare delayed gate charge to total charge
- **Falling-Edge Percentage Slope (FEPS)**: Measures slope between 60% and 10% thresholds on falling edge
- **Gatti Parameter (GP)**: Applies weighted linear operation using reference signal weights
- **Log-Likelihood Ratio (LLR)**: Applies weighted linear operation using PMF-based likelihood ratio between signal classes
- **Log Mean Time (LMT)**: Calculates natural logarithm of amplitude-weighted mean time
- **Principal Component Analysis (PCA)**: Projects signals onto first principal component from covariance matrix
- **Pulse Gradient Analysis (PGA)**: Measures gradient between peak and fixed time point after peak
- **Pattern Recognition (PR)**: Computes angle between post-peak portions of pulse and reference signals
- **Zero Crossing (ZC)**: Measures time from pulse start to zero crossing after taking the second derivative of the signal

### Frequency Domain Methods
- **Discrete Fourier Transform (DFT)**: Analyzes zero-frequency components of cosine and sine transforms
- **Frequency Gradient Analysis (FGA)**: Measures gradient between first two frequency components
- **Fractal Spectrum (FS)**: Computes fractal dimension from slope of log-transformed power spectra
- **Scalogram-based Discrimination (SD)**: Uses CWT analysis with discrimination mask from training data
- **Simplified Digital Charge Collection (SDCC)**: Analyzes decay rate differences in specific ROI
- **Wavelet Transform with signal ratio (WT1)**: Uses Haar wavelet to compute ratio of scale features
- **Wavelet Transform with signal integration (WT2)**: Employs Marr wavelet for signal integral ratios

### Neural Network Methods (Spiking)
- **Heterogeneous Quasi-Continuous Spiking Cortical Model (HQC)**: Combines two spiking models for enhanced feature extraction
- **Ladder Gradient (LG)**: Generates ignition map and computes discrimination factors based on slope
- **Pulse-Coupled Neural Network (PCNN)**: Generate ignition maps with feedback mechanisms and computes discrimination factors based on integration
- **Random-Coupled Neural Network (RCNN)**: Uses random coupling on the basis of PCNN
- **Spiking Cortical Model (SCM)**: Implements a simplified PCNN model for generating ignition maps

### Machine Learning Methods
- **Boosted Decision Tree (BDT)**: Uses AdaBoost with weak decision trees
- **Decision Tree (DT)**: Employs PCA for dimensionality reduction and decision tree
- **Fuzzy C-Means (FCM)**: Employs fuzzy clustering techniques
- **Gaussian Mixture Model (GMM)**: Implements GMM based on total charge and integral ratio
- **K-Nearest Neighbors (KNN)**: Uses segment-based features for KNN
- **Linear Regression (LINRE)**: Predicts PSD factors using PCA and linear regression
- **Logistic Regression (LOGRE)**: Performs binary classification with PCA-based features
- **Logistic Regression with Short-Time Fourier Transform (LRSTFT)**: Combines STFT features with logistic regression
- **Learning Vector Quantization (LVQ)**: Uses competitive learning with prototype vectors for classification
- **Support Vector Machine (SVM)**: Classifies using tail-to-total ratio and total charge features
- **Tempotron (GPU/CUDA required)**: Implements a spiking neural network for classification

### Deep Learning Methods
- **Convolutional Neural Network Variants**:
  - **1D CNN with Shallow Layers (CNNSHAL)**: Lightweight 1D convolutional network for  classification and regression
  - **CNN with Deep Layers (CNNDEEP)**: Deeper 2D convolutional architecture for enhanced feature extraction
  - **CNN with Fourier Transform (CNNFT)**: Processes 2D frequency-domain features using CNN
  - **CNN with Snapshot (CNNSP)**: Processes 2D snapshots of signals using CNN
  - **CNN with STFT (CNNSTFT)**: Processes 2D time-frequency features using CNN
  - **CNN with Wavelet Transform (CNNWT)**: Processes 2D wavelet-based features using CNN
- **Single Layer Perceptron (Small MLP) Variants**:
  - **Basic (MLP1)**: Simple perceptron for classification and regression
  - **With Fourier Transform (MLP1FT)**: Uses FFT magnitude features
  - **With PCA (MLP1PCA)**: Applies PCA for dimensionality reduction
  - **With STFT (MLP1STFT)**: Utilizes STFT features
  - **With Wavelet Transform (MLP1WT)**: Utilizes wavelet features
- **Dense Neural Network (Middle MLP) Variants**:
  - **Basic (MLP2)**: Multi-layer perceptron for classification and regression
  - **With Fourier Transform (MLP2FT)**: Incorporates FFT features for enhanced performance
  - **With PCA (MLP2PCA)**: Reduces dimensionality using PCA
  - **With STFT (MLP2STFT)**: Utilizes STFT features
  - **With Wavelet Transform (MLP2WT)**: Uses wavelet features
- **MultiLayer Perceptron (Large MLP) Variants**:
  - **Basic (MLP3)**: Deep perceptron for classification and regression
  - **With Fourier Transform (MLP3FT)**: Employs FFT features for improved accuracy
  - **With PCA (MLP3PCA)**: Reduces dimensionality for better performance
  - **With STFT (MLP3STFT)**: Utilizes STFT features
  - **With Wavelet Transform (MLP3WT)**: Utilizes wavelet features
- **Recurrent Neural Network Variants**:
  - **Elman Neural Network (ENN)**: Recurrent network with fully connected layers for sequence processing
  - **Gated Recurrent Unit (GRU)**: Gated memory mechanism for effective sequence modeling
  - **Long Short-Term Memory (LSTM)**: Advanced recurrent network for handling long-range dependencies
  - **Recurrent Neural Network (RNN)**: Basic recurrent architecture for sequence processing
- **Transformer Network (TRAN)**: Attention-based architecture for classification and regression
- **Mamba Network (MAM)**: State space model for classification and regression

## Extending the toolbox

Adding a new method is straightforward:

- Statistical methods (time/frequency/spiking): implement `get_psd_factor(data: np.ndarray) -> np.ndarray` in a new module under the appropriate `Method/Statistical_methods/...` subfolder and import it in the corresponding main script mapping.
- ML/DL methods: implement the following functions in `Method/Prior_knowledge_methods/...` modules so they can be discovered from the main scripts:
  - `get_supported_tasks() -> list[str]` returning `['classification']`, `['regression']`, or both
  - `train(data: np.ndarray, labels: np.ndarray, task: str, feat_name: Optional[str]) -> None`
  - `load_model(task: str, feat_name: Optional[str]) -> None`
  - `test(data: np.ndarray, task: str, feat_name: Optional[str]) -> np.ndarray`

Please also document method references and default hyperparameters in module-level docstrings.

## Contributing

Although this project incorporates most PSD algorithms developed over the last five decades, there are still methods that have not been included, as well as newly emerged approaches. Contributions to this project are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Guidelines
- Follow the existing code style and organization
- Add concise docstrings where needed (focus on why, not how)
- Include basic sanity checks or tests where practical
- Update the README if necessary

## Troubleshooting

- **Matplotlib/Tk errors on headless systems**: use CLI flags so interactive plotting is disabled and FOM figures are saved during test/validation, or call `histogram_fitting_compute_fom(..., show_plot=False)` explicitly. The plotting utility automatically falls back to the `Agg` backend if interactive backends are unavailable.
- **Shape errors**: Ensure `.txt` files are 2D arrays with one pulse per row.
- **NaN warnings**: Some methods may produce NaNs for certain signals/parameters. The scripts remove NaNs and continue; consider adjusting method hyperparameters.
- **Unsupervised clustering label flips (ML)**: The ML entry point automatically flips labels when needed via majority voting.
- **Tempotron fails to run / device errors**: Tempotron requires CUDA-enabled PyTorch and an NVIDIA GPU. Verify `torch.cuda.is_available()` is True, install the correct CUDA wheel, update your GPU drivers, and (optionally) set `CUDA_VISIBLE_DEVICES` to choose a GPU.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the researchers in the field of radiation detection and pulse shape discrimination. References for each PSD method can be found in the corresponding Python files.
