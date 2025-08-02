"""
Simplified Digital Charge Collection (SDCC) method for pulse shape discrimination.

This module implements a simplified domain transformation approach to pulse shape
discrimination by analyzing the decay rate differences between class 1 and class 2
signals in a specific region of interest (ROI).

Reference:
- Shippen, David I., Malcolm J. Joyce, and Michael D. Aspinall.
  "A wavelet packet transform inspired method of neutron-gamma discrimination."
  IEEE Transactions on Nuclear Science 57, no. 5 (2010): 2617-2624.
"""

import numpy as np
from typing import Union


def get_psd_factor(pulse_signal: np.ndarray, ROI_left_endpoint: Union[int, None] = 70) -> np.ndarray:
    """
    Performs pulse shape discrimination based on Simplified Digital Charge Collection (SDCC).

    SDCC computes a simplified domain transformation to serve as the discrimination factor.
    For each pulse signal, the region of interest (ROI) is defined starting from the provided
    ROI_left_endpoint to the end of the signal. The discrimination factor is computed as the
    natural logarithm of the sum of the squared values of the ROI.

    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal
        ROI_left_endpoint: The starting index (0-based) for the region of interest where the decay
                          rate difference between class 1 and class 2 signals is the greatest (default: 70)

    Returns:
        A numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.zeros(num_signals)

    for i in range(num_signals):
        # Extract the pulse signal vector for the current pulse
        signal = pulse_signal[i, :]
        # Define the region of interest (ROI) starting from ROI_left_endpoint to the end
        ROI = signal[ROI_left_endpoint:]
        # Compute the discrimination factor as the logarithm of the sum of squared ROI values
        discrimination_factors[i] = np.log(np.sum(ROI ** 2))

    return discrimination_factors

