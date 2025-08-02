"""
Pulse Gradient Analysis (PGA) method for pulse shape discrimination.

The method computes discrimination factors by calculating the gradient 
between the pulse peak and a fixed time point after the peak, providing
a simple but effective measure of pulse shape differences.

Reference:
- D’Mellow, Bob, M. D. Aspinall, R. O. Mackin, Malcolm J. Joyce, and A. J. Peyton.
  "Digital discrimination of neutrons and γ-rays in liquid scintillators using pulse gradient analysis."
  Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers,
  Detectors and Associated Equipment 578, no. 1 (2007): 191-197.
"""

import numpy as np


def get_psd_factor(
        pulse_signal: np.ndarray,
        t: int = 20
) -> np.ndarray:
    """
    Calculate PSD factors using Pulse Gradient Analysis.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
        t: Number of samples after the peak to use for gradient calculation
           (default: 20)
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse, computed as
                      (V[peak + t] - V[peak]) / t. NaN values indicate invalid
                      calculations (e.g., peak + t beyond signal length)
    
    Note:
        The gradient is calculated as a simple finite difference over t samples,
        normalized by the time interval to give a rate of change.
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i, signal in enumerate(pulse_signal):
        # Find the peak value and its index
        max_index = np.argmax(signal)   # Adjust to MATLAB-style indexing
        max_value = signal[max_index ]  # Adjust the access to match Python's 0-indexing

        # Calculate the index for the sample t samples after the peak
        second_sample_index = max_index + t

        # Check if the second sample index is within bounds
        if second_sample_index < signal.shape[0]:
            second_sample_value = signal[second_sample_index]
            discrimination_factors[i] = (second_sample_value - max_value) / t
        else:
            # If out-of-bounds, assign NaN for this signal
            discrimination_factors[i] = np.nan

    return discrimination_factors

