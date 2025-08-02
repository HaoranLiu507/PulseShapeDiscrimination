"""
Charge Comparison (CC) method for pulse shape discrimination.

This module implements the Charge Comparison method for pulse shape discrimination,
which compares the charge in the slow component of the pulse to the total charge to 
generate a discrimination factor.

Reference:
- Moszynski, Marek, G. Bizard, G. J. Costa, D. Durand, Y. El Masri, G. Guillaume, F. Hanappe et al.
  "Study of n-γ discrimination by digital charge comparison method for a large volume liquid scintillator."
  Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers,
  Detectors and Associated Equipment 317,no. 1-2 (1992): 262-272.
"""

import numpy as np
from typing import Union, List, Tuple

def get_psd_factor(
    pulse_signal: np.ndarray,
    gate_range: Union[List[int], Tuple[int, int, int]] = (10, 27, 90)
) -> np.ndarray:
    """
    Calculate PSD factors using the Charge Comparison method.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
        gate_range: Integration gate parameters (pre_gate, short_gate, long_gate)
                   - pre_gate: Number of samples before peak
                   - short_gate: Start of slow component after peak
                   - long_gate: End of integration window after peak
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse. NaN values indicate
                      invalid calculations (e.g., out of bounds or zero division)
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i, signal in enumerate(pulse_signal):
        max_position = np.argmax(signal)
        short_gate_start = max_position + gate_range[1]
        short_gate_end = max_position + gate_range[2] + 1
        long_gate_start = max(max_position - gate_range[0], 0)
        long_gate_end = max_position + gate_range[2] + 1

        # Ensure indices are within bounds and avoid division by zero
        if short_gate_end <= len(signal) and long_gate_end <= len(signal):
            total_component = signal[long_gate_start:long_gate_end]
            if np.sum(total_component) != 0:
                slow_component = signal[short_gate_start:short_gate_end]
                discrimination_factors[i] = np.sum(slow_component) / np.sum(total_component)
            else:
                discrimination_factors[i] = np.nan
        else:
            discrimination_factors[i] = np.nan

    return discrimination_factors
