# Pulse Shape Discrimination (PSD)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Overview](#overview)
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The Pulse Shape Discrimination (PSD) project is a comprehensive Python/MATLAB toolbox for discriminating between different types of radiation particles based on their pulse shapes in scintillation detector signals. PSD is a critical technique in radiation detection that enables the separation of different particle types based on the characteristic shapes of their pulse signals, which is essential for applications in nuclear security, medical physics, and scientific research.

- If you find our work useful in your research or publication, please cite our work:

- Haoran Liu, Yihan Zhan, Mingzhe Liu, Yanhua Liu, Peng Li, Runxi Liu, Zhuo Zuo, and Bing-Qi Liu, **"Pulse Shape Discrimination Algorithms: Survey and Benchmark."** *Under Review,* vol. XX, no. XX, pp. XXXX-XXXX, XXX. 2025. [[arXiv](https://doi.org/)]

## Project Description

Pulse shape discrimination is essential in radiation detection systems to distinguish between different types of radiation. This project employs a variety of approaches to PSD, incorporating both statistical methods and prior knowledge models. Neutron and gamma-ray discrimination serve as the default application demo for this project. By incorporating additional datasets, other particle types can also be utilized. For a more detailed description of the PSD methods incorporated in this project, as well as comparisons and analyses between these methods, please refer to our review article: [[arXiv](https://doi.org/)].

### Key Features:

- **Multiple PSD Methodologies**: Implements time-domain, frequency-domain, neural network, machine learning, and deep learning approaches
- **Flexible Pre-processing**: Various signal filtering options including Butterworth, Chebyshev, Elliptic, and Wavelet filters
- **Comprehensive Evaluation**: Tools for computing the Figure of Merit (FOM) to quantitatively evaluate discrimination performance
- **Modular Design**: Easy to extend with new algorithms and methods
- **Two Task Types**: Supports both classification (particle class identification) and regression (PSD factor prediction)
- **MATLAB Support**: Includes MATLAB implementations for time-domain, frequency-domain, and neural network methods

## Project Structure

The project is organized into the following directories:

```
Pulse_shape_discrimination/
├── Data/                      # Contains input data files
│   ├── Reference_signal/      # Reference signals for GP and LLR methods
│   ├── Test/                  # Testing datasets
│   ├── Train/                 # Training datasets
│   └── Validation/            # Validation datasets
│
├── MATLAB/                    # Contains Statistical methods in MATLAB
│   ├── Data/                  # MATLAB data files
│   ├── Method/                # MATLAB implementations of PSD methods
│   ├── Utility/               # MATLAB utility functions
│   ├── Frequency_domain_main.m # Main MATLAB script for frequency domain methods
│   ├── Neural_network_main.m   # Main MATLAB script for neural network methods
│   └── Time_domain_main.m      # Main MATLAB script for time domain methods
│
├── Method/                    # Contains all PSD methodologies
│   ├── Prior_knowledge_methods/       # Model-based methods
│   │   ├── Deep_learning/             # CNN, LSTM, MLP, etc.
│   │   └── Machine_learning/          # SVM, KNN, DT, etc.
│   │
│   └── Statistical_methods/           # Statistical approaches
│       ├── Frequency_domain/          # WT, DFT, FGA, etc.
│       ├── Neural_network/            # SCM, PCNN, etc.
│       └── Time_domain/               # CC, CI, PGA, etc.
│
├── Output/                    # Output results directory
│   ├── Trained_models/        # Saved trained models
│   └── Validation_results/    # Results from validation
│
├── Utility/                   # Utility functions and tools
│   ├── filters.py             # Signal filtering implementations
│   ├── histogram_fitting_compute_fom.py # FOM calculation
│   └── Tempotron.py           # Tempotron neural model
│
├── Deep_learning_main.py      # Main executable for DL methods
├── Frequency_domain_main.py   # Main executable for frequency domain methods
├── Machine_learning_main.py   # Main executable for ML methods
├── Neural_network_main.py     # Main executable for neural network methods
├── README.md                  # This file
├── requirements.txt           # Required Python packages
└── Time_domain_main.py        # Main executable for time domain methods
```

## Installation

### Prerequisites
- Python 3.12 or higher
- MATLAB R2023a or higher
- CUDA toolkit (for GPU support)

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

4. Install PyTorch:
   ```bash
   # For the version used in development (with CUDA 12.6):
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

   # Note: If you have different hardware requirements, please visit:
   # https://pytorch.org/get-started/locally/
   # to select the appropriate PyTorch installation command for your system
   ```

   The development versions used were:
   - torch==2.6.0
   - torchvision==0.21.0

5. Download the Companion Dataset from Zenodo

   The complete dataset, pre-trained models, and experimental results are hosted on Zenodo. This companion package is required to reproduce the figures and benchmarks presented in our survey paper.

   - **Download link**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16728732.svg)](https://doi.org/10.5281/zenodo.16728732)

   The Zenodo repository contains:
   - **`Datasets/`**: Raw waveform data for both AmBe and PuBe sources.
   - **`Experimental Results/`**: Pre-computed models, validation metrics, and summary spreadsheets.
   - **`PSD Parameters for PuBe Dataset/`**: Python scripts with algorithm parameters tuned for the PuBe dataset.

   Please see the `README.txt` file in the Zenodo bundle for detailed instructions on using this data with the toolbox.

## Usage

The project provides separate main scripts for different PSD methodologies:

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
To run the MATLAB implementations, open MATLAB, navigate to the MATLAB/ directory, and execute one of the main scripts. For example:
```matlab
run('Time_domain_main.m');
```
Repeat similarly for Frequency_domain_main.m and Neural_network_main.m.

### General Workflow

1. **Choose Method**: Select the appropriate main script based on the method you want to use
2. **Apply Filtering** *(Python only)*: Select a filter to apply to the signals (or no filter)
3. **Select Task Type** *(ML/DL only)*: Choose between classification or regression task
4. **Choose Algorithm**: Select a specific PSD method from the available options
5. **Model Selection** *(ML/DL only)*: Choose to train a new model or load a pre-trained one
6. **Model Training** *(ML/DL only)*: Train and test the selected model
7. **Validation**: Evaluate on validation data
8. **Results Analysis**: View and analyze results in the Output directory *(Python only - MATLAB implementations do not save output as files)*

**Notes:**
* MATLAB implementations are provided for statistical methods, as many researchers in the PSD field are more familiar with MATLAB. However, machine learning and deep learning methods are only available in Python due to better library support.
* We recommend using the Python implementations for all methods due to their more mature functionality (filters, visualization, etc.). If you only intend to use Python, you can disregard the MATLAB directory.
* This repository uses a dataset detected from an AmBe neutron source by default. If you want to use other datasets, please change the path of data loading at the head of the main function of each PSD method category. For example, see lines 43-47 of 'Deep_learning_main.py'.
* **Important:** There are a few PSD methods (including GP, LLR, and SD) that require reference signals loaded from separate .txt files. If you changed the dataset, please make sure you update the reference signal from your new dataset. The reference loading procedure can be found in the PSD method .py file. For example, see lines 173-174 of 'Method\Statistical_methods\Frequency_domain\sd.py'.

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
- **Linear Regression with Short-Time Fourier Transform (LRSTFT)**: Combines STFT features with linear regression
- **Learning Vector Quantization (LVQ)**: Uses competitive learning with prototype vectors for classification
- **Support Vector Machine (SVM)**: Classifies using tail-to-total ratio and total charge features
- **Tempotron**: Implements a spiking neural network for classification

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

## Contributing

Although this project incorporates most PSD algorithms developed over the last five decades, there are still methods that have not been included, as well as newly emerged approaches. Contributions to this project are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Guidelines:
- Follow the existing code style and organization
- Add appropriate comments and documentation
- Include test cases for new features
- Update the README if necessary with new information

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the researchers in the field of radiation detection and pulse shape discrimination. References for each PSD method can be found in their corresponding Python files.
