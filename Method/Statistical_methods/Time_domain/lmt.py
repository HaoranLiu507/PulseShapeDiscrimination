"""
Log Mean Time (LMT) method for pulse shape discrimination.

The method computes discrimination factors by taking the natural logarithm 
of the amplitude-weighted mean time of each pulse signal.

References:
- Lee, H. S., H. Bhang, J. H. Choi, S. Choi, I. S. Hahn, E. J. Jeon, H. W. Joo et al.
  "Neutron calibration facility with an Am-Be source for pulse shape discrimination measurement of CsI (Tl) crystals."
  Journal of Instrumentation 9, no. 11 (2014): P11015.
"""

import numpy as np


def get_psd_factor(
        pulse_signal: np.ndarray
) -> np.ndarray:
    """
    Calculate PSD factors using the Log Mean Time method.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse, computed as
                      ln(mean_time) where mean_time is the amplitude-weighted
                      average time. NaN values indicate invalid calculations
                      (e.g., zero total amplitude)
    
    Note:
        The time series is 1-indexed, meaning the first sample is at t=1.
    """
    # Create a time series corresponding to each time point in the pulse signals (1-indexed)
    time_series = np.arange(1, pulse_signal.shape[1] + 1)

    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i, signal in enumerate(pulse_signal):
        # Compute the weighted sum of the pulse amplitudes using the time series
        weighted_sum = np.sum(signal * time_series)
        total_sum = np.sum(signal)

        # Calculate mean time and compute its natural logarithm as the discrimination factor
        if total_sum != 0:
            mean_time = weighted_sum / total_sum
            discrimination_factors[i] = np.log(mean_time)
        else:
            discrimination_factors[i] = np.nan

    return discrimination_factors

