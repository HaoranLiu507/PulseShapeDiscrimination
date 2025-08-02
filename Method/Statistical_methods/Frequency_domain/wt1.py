"""
Wavelet Transform (WT1) method for pulse shape discrimination.

This module implements a wavelet-based approach to pulse shape discrimination
using the Haar wavelet transform. The discrimination factor is computed as
the ratio of specific scale features (28th and 40th) that best capture 
the differences between neutron and gamma signals.

Reference:
- Yousefi, S., L. Lucchese, and M. D. Aspinall.
  "Digital discrimination of neutrons and gamma-rays in liquid scintillators using wavelets."
  Nuclear Instruments and Methods in Physics Research Section A: Accelerators,
  Spectrometers, Detectors and Associated Equipment 598, no. 2 (2009): 551-555.
"""

import numpy as np
from tqdm import tqdm


def haar(t: np.ndarray) -> np.ndarray:
    """
    Compute the normalized Haar wavelet function values.

    Args:
        t: A numpy array representing the time or scale variable

    Returns:
        A numpy array representing the normalized Haar wavelet
    """
    w = np.zeros_like(t)
    w[(t >= 0) & (t < 0.5)] = 1
    w[(t >= 0.5) & (t < 1)] = -1
    
    # Normalize the wavelet
    norm = np.linalg.norm(w)
    if norm != 0:
        w = w / norm
    return w


def wavelet_haar(scale: float, shift: float) -> np.ndarray:
    """
    Generate Haar wavelet at a specific scale and shift.

    Args:
        scale: The scale factor for the Haar wavelet
        shift: The translation or shift for the Haar wavelet

    Returns:
        A numpy array representing the Haar wavelet at the given scale and shift
    """
    t = np.arange(-shift, shift + 0.1, 0.1)
    return haar(t / scale)


def scale_haar(data: np.ndarray, scale: int, shift: int) -> np.ndarray:
    """
    Compute the scale function using the Haar wavelet transform.

    Args:
        data: 1D numpy array of input signal data
        scale: The number of scales for the Haar wavelet transform
        shift: The translation amount for the Haar wavelet function

    Returns:
        A numpy array of computed scale function values for each scale level
    """
    n = shift * 2
    scale_function = np.zeros(scale)
    
    # Compute the scale function for each scale level
    for s in range(1, scale + 1):
        # Generate the Haar wavelet at the current scale and shift
        wavelet = wavelet_haar(s, shift)
        
        # Compute wavelet transformation through convolution
        conv_result = np.convolve(data, wavelet, mode='same')
        
        # Compute the energy of the wavelet transform
        # Scale the convolution result by the time indices / sqrt(scale)
        time_indices = np.arange(1, len(conv_result) + 1)
        W = (time_indices / np.sqrt(s)) * conv_result
        
        # Calculate the energy as the sum of absolute values
        energy = np.sum(np.abs(W))
        
        # Normalize the scale function value
        scale_function[s - 1] = energy / (1 + n)
    
    return scale_function


def get_psd_factor(pulse_signal: np.ndarray, scale: int = 150, shift: int = 50,
                scale_feature_1: int = 27, scale_feature_2: int = 39) -> np.ndarray:
    """
    Performs pulse shape discrimination based on Wavelet Transform (WT).

    For each pulse signal (each row in the input array), the function computes a
    scale function using the Haar wavelet transform. It then derives the
    discrimination factor as the ratio of two scale features, where the two classes 
    of signals differ the most. By default, it uses the 28th and 40th scale features 
    (Python indices 27 and 39 respectively).

    Args:
        pulse_signal: A 2D numpy array where each row is a pulse signal
        scale: The scaling factor for the Haar wavelet transform (default: 100)
        shift: The translation amount for the Haar wavelet (default: 50)
        scale_feature_1: First scale feature index for discrimination (default: 27, corresponds to 28th feature)
        scale_feature_2: Second scale feature index for discrimination (default: 39, corresponds to 40th feature)

    Returns:
        A 1D numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.zeros(num_signals)

    # Use tqdm to show a progress bar during processing
    for i in tqdm(range(num_signals), desc="Processing signals", unit="signal"):
        signal_vector = pulse_signal[i, :]
        
        # Compute the wavelet scale function
        scale_function = scale_haar(signal_vector, scale, shift)
        
        # Compute the discrimination factor using the specified scale feature indices
        max_feature = max(scale_feature_1, scale_feature_2)
        if len(scale_function) > max_feature and scale_function[scale_feature_2] != 0:
            discrimination_factors[i] = scale_function[scale_feature_1] / scale_function[scale_feature_2]
        else:
            discrimination_factors[i] = np.nan

    return discrimination_factors
