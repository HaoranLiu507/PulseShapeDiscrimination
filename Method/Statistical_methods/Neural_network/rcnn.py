"""
Random-Coupled Neural Network (RCNN) implementation for pulse shape discrimination.

This implementation introduces randomness into the neural coupling mechanism to enhance
the network's ability to capture subtle temporal differences in pulse shapes. The model
combines Gaussian kernels with random matrices to create dynamic connection weights.

Reference:
- Liu, Haoran, Mingrong Xiang, Mingzhe Liu, Peng Li, Xue Zuo, Xin Jiang, and Zhuo Zuo.
  "Random-coupled Neural Network."
  Electronics 13, no. 21 (2024): 4297.
"""
import numpy as np
from scipy.signal import convolve2d


def fspecial_gaussian(dimension: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian kernel for spatial filtering.
    
    Generates a square kernel with specified size and standard deviation,
    normalized so the sum of all elements equals 1.
    
    Args:
        dimension: Size of the square kernel
        sigma: Standard deviation of the Gaussian distribution
        
    Returns:
        2D numpy array representing the normalized Gaussian kernel
    """
    ax = np.linspace(-(dimension - 1) / 2., (dimension - 1) / 2., dimension)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel


def rand_matrix(dimension: int, probability: float, flag: str, sigma: float) -> np.ndarray:
    """
    Generate a random weight matrix for neural coupling.
    
    Creates either a Gaussian-based or uniform random binary matrix based on the flag.
    For 'norm' flag, generates a normalized Gaussian kernel converted to binary form.
    
    Args:
        dimension: Size of the square matrix
        probability: Threshold probability for binary conversion
        flag: 'norm' for Gaussian-based matrix, otherwise uniform random
        sigma: Standard deviation for Gaussian kernel when flag is 'norm'
        
    Returns:
        Binary numpy array of shape (dimension, dimension)
    """
    if flag == 'norm':
        D = fspecial_gaussian(dimension, sigma)
        center = dimension // 2
        S = 1.0 / D[center, center] if D[center, center] != 0 else 1.0
        D = D * S
        return (np.random.rand(dimension, dimension) < D).astype(float)
    else:
        D = np.ones((dimension, dimension))
        return (np.random.rand(dimension, dimension) < (D * probability)).astype(float)


def rcnn(pulse_signal_vector: np.ndarray) -> np.ndarray:
    """
    Process a pulse signal using the Random-Coupled Neural Network model.
    
    Implements RCNN with dynamic random coupling weights generated from Gaussian kernels.
    The model features feedback mechanisms and dynamic thresholding to enhance
    discrimination capabilities.
    
    Args:
        pulse_signal_vector: 1D numpy array representing a single pulse signal
        
    Returns:
        1D numpy array representing the computed ignition map
    """
    # Ensure the pulse signal is 2D (1 x N)
    if pulse_signal_vector.ndim == 1:
        pulse_signal_vector = pulse_signal_vector.reshape(1, -1)
    rows, cols = pulse_signal_vector.shape

    # RCNN model parameters
    B = 0.4
    V = 1
    aT = 0.709
    vT = 0.101
    aF = 0.205
    iterations = 20  # number of iterations
    dimension = 9  # dimension of the Gaussian kernel
    sigma1 = 4  # standard deviation for primary Gaussian kernel
    sigma2 = 6  # standard deviation for secondary Gaussian matrix

    # Create a Gaussian kernel and set its center to 0
    Gaussian_kernel = fspecial_gaussian(dimension, sigma1)
    center = dimension // 2
    Gaussian_kernel[center, center] = 0

    # Initialize variables
    Y = np.zeros((rows, cols))
    U = np.zeros((rows, cols))
    E = np.ones((rows, cols))
    Ignition_map = np.zeros((rows, cols))

    for i in range(iterations):
        # Generate a random weight matrix using the Gaussian kernel and rand_matrix function
        Weight_matrix_random = Gaussian_kernel * rand_matrix(dimension, 0.1, 'norm', sigma2)

        # Compute link input via 2D convolution (using 'same' mode to retain size)
        L = convolve2d(Y, Weight_matrix_random, mode='same')

        # Update state U based on the previous state and the convolution result with the pulse signal
        U = U * np.exp(-aF) + pulse_signal_vector * (1 + V * B * L)

        # Pulse generator: generate a binary sequence (im2double equivalent)
        Y = (U > E).astype(float)

        # Update the dynamic threshold E
        E = np.exp(-aT) * E + vT * Y

        # Accumulate the results in the ignition map
        Ignition_map += Y

    return Ignition_map.flatten()


def get_psd_factor(
    pulse_signal: np.ndarray,
    ROI_end: int = 123
) -> np.ndarray:
    """
    Perform pulse shape discrimination using the Random-Coupled Neural Network.
    
    For each pulse signal, generates an ignition map using RCNN with random coupling
    and computes the discrimination factor as the sum of map values within a region
    of interest (ROI) starting from the pulse maximum. This approach effectively
    captures the temporal differences between class 1 and class 2 pulses.
    
    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal
        ROI_end: Number of points after the maximum position to include in the ROI (default: 123)
        
    Returns:
        1D numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i in range(num_signals):
        signal_vector = pulse_signal[i, :]

        # Compute the ignition map using RCNN
        ignition_map = rcnn(signal_vector)

        # Find the index of the maximum value in the pulse signal
        max_position = np.argmax(signal_vector)

        # Ensure ROI does not exceed the bounds of the ignition map
        end_index = min(max_position + ROI_end, len(ignition_map))

        # Sum the ignition map values within the ROI
        discrimination_factor = np.sum(ignition_map[max_position:end_index])
        discrimination_factors[i] = discrimination_factor

    return discrimination_factors
