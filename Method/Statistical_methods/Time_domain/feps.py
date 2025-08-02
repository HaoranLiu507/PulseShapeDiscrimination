"""
Falling-Edge Percentage Slope (FEPS) method for pulse shape discrimination.

The method calculates discrimination factors by measuring the slope between 
two threshold points (60% and 10% of maximum) on the falling edge of the pulse.

Reference:
- Zuo, Zhuo, YuLong Xiao, ZhenFeng Liu, BingQi Liu, and YuCheng Yan.
  "Discrimination of neutrons and gamma-rays in plastic scintillator based on falling-edge percentage slope method."
  Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers,
  Detectors and Associated Equipment 1010 (2021): 165483.
"""

import numpy as np

def get_psd_factor(
    pulse_signal: np.ndarray
) -> np.ndarray:
    """
    Calculate PSD factors using the Falling-Edge Percentage Slope method.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
    
    Returns:
        numpy.ndarray: Array of PSD factors (slopes) for each input pulse. NaN values
                      indicate invalid calculations (e.g., incorrect threshold crossings)
    
    Note:
        The method uses fixed thresholds of 60% (upper) and 10% (lower) of the pulse
        maximum to calculate the falling edge slope.
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i, signal in enumerate(pulse_signal):
        # Find the peak and its index
        max_position = np.argmax(signal)
        max_value = signal[max_position]

        if max_value <= 0:
            # Cannot determine a falling edge for non-positive peaks.
            # The discrimination factor remains as the default np.nan.
            continue

        # Define thresholds: 10% and 60% of the maximum value
        lower_threshold = max_value * 0.1
        upper_threshold = max_value * 0.6

        # Consider the falling edge part of the signal
        falling_edge = signal[max_position:]
        
        # Find the first index where the signal drops below the upper and lower thresholds
        upper_cross_indices = np.where(falling_edge <= upper_threshold)[0]
        lower_cross_indices = np.where(falling_edge <= lower_threshold)[0]

        # Check if both thresholds were crossed on the falling edge
        if upper_cross_indices.size > 0 and lower_cross_indices.size > 0:
            first_upper_cross = upper_cross_indices[0]
            first_lower_cross = lower_cross_indices[0]

            # The indices must be different for a valid slope calculation
            if first_upper_cross < first_lower_cross:
                # Convert back to original signal indices
                crossing_index_upper = first_upper_cross + max_position
                crossing_index_lower = first_lower_cross + max_position

                # Calculate slope using the actual signal values at the crossing points
                delta_y = signal[crossing_index_lower] - signal[crossing_index_upper]
                delta_x = crossing_index_lower - crossing_index_upper
                
                discrimination_factors[i] = delta_y / delta_x
        # If crossings are not found or not in the right order, the factor remains NaN.

    return discrimination_factors
