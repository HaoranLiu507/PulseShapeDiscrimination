"""
Scalogram-based Discrimination (SD) method for pulse shape discrimination.

This module implements a wavelet-based approach to pulse shape discrimination
using continuous wavelet transform (CWT) analysis. The method generates a
discrimination mask from labeled training data and uses it to compute
discrimination factors for mixed signals.

Reference:
- Abdelhakim, Assem, and Ehab Elshazly.
  "Efficient pulse shape discrimination using scalogram image masking and decision tree."
  Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers,
  Detectors and Associated Equipment 1050 (2023): 168140.
"""

import numpy as np
import pywt
from tqdm import tqdm
from typing import Optional


def _normalize_signals(signals: np.ndarray) -> np.ndarray:
    """
    Normalize each pulse signal by its maximum absolute value.

    Args:
        signals: Input signals array of shape (n_signals, signal_length)

    Returns:
        Normalized signals array of the same shape
    """
    normalized = np.zeros_like(signals)
    for idx in range(signals.shape[0]):
        pulse = signals[idx, :]
        max_amplitude = np.max(np.abs(pulse))
        if max_amplitude < 1e-9:  # Avoid division by near-zero values
            normalized[idx, :] = pulse
        else:
            normalized[idx, :] = pulse / max_amplitude
    return normalized


def _convert_to_grayscale(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a matrix to grayscale values (0-255).

    Args:
        matrix: Input matrix to be converted to grayscale

    Returns:
        Grayscale matrix with values between 0 and 255
    """
    min_val, max_val = matrix.min(), matrix.max()
    return 255 * (matrix - min_val) / (max_val - min_val + 1e-9)


def _generate_discrimination_mask(
        class1_signals: np.ndarray,
        class2_signals: np.ndarray,
        wavelet: str = 'mexh',
        scales: Optional[np.ndarray] = None,
        threshold: int = 127
) -> np.ndarray:
    """
    Generate a binary discrimination mask from training data using wavelet analysis.

    Args:
        class1_signals: Training data for first class of signals
        class2_signals: Training data for second class of signals
        wavelet: Wavelet type to use (default: 'mexh')
        scales: Wavelet scales (default: np.arange(1,51))
        threshold: Grayscale threshold for binary conversion (default: 127)

    Returns:
        Binary mask for signal discrimination

    Raises:
        ValueError: If no discrimination features are found between signal classes
    """
    if scales is None:
        scales = np.arange(1, 51)

    # Use a subset of training samples to build the mask
    num_samples = min(class1_signals.shape[0], class2_signals.shape[0])
    training_size = max(1, int(num_samples / 15))
    difference_count = np.zeros((len(scales), class1_signals.shape[1]), dtype=int)

    for sample_idx in range(training_size):
        # Process one pulse from each class
        pulse1 = class1_signals[sample_idx, :]
        pulse2 = class2_signals[sample_idx, :]

        # Calculate CWT coefficients and corresponding energy scalograms
        coeffs1, _ = pywt.cwt(pulse1, scales, wavelet, method='fft')
        coeffs2, _ = pywt.cwt(pulse2, scales, wavelet, method='fft')
        energy1 = np.abs(coeffs1) ** 2
        energy2 = np.abs(coeffs2) ** 2

        # Convert energy scalograms to grayscale and then to binary ROI images
        gray1 = _convert_to_grayscale(energy1)
        gray2 = _convert_to_grayscale(energy2)
        roi1 = np.where(gray1 >= threshold, 1, 0)
        roi2 = np.where(gray2 >= threshold, 1, 0)

        # Accumulate absolute differences between the ROIs
        difference_count += np.abs(roi1 - roi2)

    nonzero_count = np.count_nonzero(difference_count)
    if nonzero_count == 0:
        raise ValueError("No discrimination features found between signal classes")

    difference_threshold = difference_count.sum() / nonzero_count
    mask = np.where(difference_count >= difference_threshold, 1, 0)
    return mask


def _discriminate_signals(
        mixed_signals: np.ndarray,
        discrimination_mask: np.ndarray,
        wavelet: str = 'mexh',
        scales: Optional[np.ndarray] = None,
        threshold: int = 127
) -> np.ndarray:
    """
    Compute discrimination factors for each signal using the provided mask.

    Args:
        mixed_signals: Array of mixed signals
        discrimination_mask: Pre-generated discrimination mask
        wavelet: Wavelet type (default: 'mexh')
        scales: Wavelet scales (default: np.arange(1,51))
        threshold: Grayscale threshold (default: 127)

    Returns:
        Discrimination factors for each signal
    """
    if scales is None:
        scales = np.arange(1, 51)

    num_signals = mixed_signals.shape[0]
    factors = np.zeros(num_signals)

    for idx in tqdm(range(num_signals), desc="Calculating Discrimination Factors", unit="signal"):
        signal = mixed_signals[idx, :]
        coeffs, _ = pywt.cwt(signal, scales, wavelet, method='fft')
        scalogram = np.abs(coeffs) ** 2
        gray_scalogram = _convert_to_grayscale(scalogram)
        roi = np.where(gray_scalogram >= threshold, 1, 0)
        # Apply the discrimination mask and compute the factor
        masked_roi = roi[discrimination_mask == 1]
        factors[idx] = np.sum(masked_roi) / np.sum(discrimination_mask)
    return factors


def get_psd_factor(pulse_signal: np.ndarray) -> np.ndarray:
    """
    Compute pulse shape discrimination factors using the Scalogram-based
    Discrimination (SD) method.

    The function performs the following steps:
      - Loads labeled training data for class 1 and class 2 signals
      - Normalizes all signals by their maximum absolute values
      - Generates a discrimination mask using wavelet analysis
      - Applies the mask to compute discrimination factors

    Args:
        pulse_signal: Array of mixed pulse signals

    Returns:
        Array of discrimination factors for the input signals
    """
    # SD requires labeled signals to generate mask
    class1_signals = np.loadtxt('Data/Train/EJ299_33_AmBe_9414_gamma_train.txt')
    class2_signals = np.loadtxt('Data/Train/EJ299_33_AmBe_9414_neutron_train.txt')

    # Normalize the signals
    class1_signals = _normalize_signals(class1_signals)
    class2_signals = _normalize_signals(class2_signals)

    # Ensure the mixed signals are normalized
    mixed_signals = _normalize_signals(pulse_signal)

    # Generate discrimination mask
    mask = _generate_discrimination_mask(
        class1_signals,
        class2_signals,
        wavelet='mexh',
        scales=np.arange(1, 51),
        threshold=127
    )

    # Compute and return the discrimination factors for the mixed signals
    return _discriminate_signals(
        mixed_signals,
        mask,
        wavelet='mexh',
        scales=np.arange(1, 51),
        threshold=127
    )
