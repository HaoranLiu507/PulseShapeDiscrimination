"""
Charge Integration (CI) method for pulse shape discrimination.

The method computes discrimination factors by comparing the charge in a 
delayed gate to the total charge of the pulse, using trapezoidal integration.

Reference:
- Pawełczak, I. A., S. A. Ouedraogo, A. M. Glenn, R. E. Wurtz, and L. F. Nakae.
  "Studies of neutron–γ pulse shape discrimination in EJ-309 liquid scintillator using charge integration method."
  Nuclear Instruments and Methods in Physics Research Section A:
  Accelerators, Spectrometers, Detectors and Associated Equipment 711 (2013): 21-26.
"""

import numpy as np
from typing import Union, List, Tuple

def get_psd_factor(
    pulse_signal: np.ndarray,
    gate_range: Union[List[int], Tuple[int, int, int]] = (14, 35, 65),
    dt: float = 1.0
) -> np.ndarray:
    """
    Calculate PSD factors using the Charge Integration method.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
        gate_range: Integration gate parameters (pre_gate, delay_gate, total_gate)
                   - pre_gate: Samples before peak for total gate start
                   - delay_gate: Length of delayed gate
                   - total_gate: Total integration length after peak
        dt: Time interval between samples for trapezoidal integration
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse. NaN values indicate
                      invalid calculations (e.g., out of bounds or zero division)
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i, signal in enumerate(pulse_signal):
        max_position = np.argmax(signal)
        # Convert to 1-indexed position to follow MATLAB logic
        p = max_position + 1

        # Calculate gate indices in 1-indexed space
        delay_gate_start = p + gate_range[2] - gate_range[1]
        delay_gate_end = p + gate_range[2]
        total_gate_start = p - gate_range[0]
        total_gate_end = p + gate_range[2]

        # Check that the computed gate indices are within valid bounds
        if (delay_gate_start < 1 or delay_gate_end > len(signal) or
            total_gate_start < 1 or total_gate_end > len(signal)):
            discrimination_factors[i] = np.nan
            continue

        # Convert 1-indexed positions to 0-indexed indices for Python slicing
        delay_start_idx = delay_gate_start - 1
        delay_end_idx = delay_gate_end  # Python slice end is exclusive
        total_start_idx = total_gate_start - 1
        total_end_idx = total_gate_end

        # Extract components from the pulse signal
        delay_component = signal[delay_start_idx:delay_end_idx]
        total_component = signal[total_start_idx:total_end_idx]

        # Compute the integrations using the trapezoidal rule with spacing dt
        delay_integration = np.trapz(delay_component, dx=dt)
        total_integration = np.trapz(total_component, dx=dt)

        # Avoid division by zero
        if total_integration != 0:
            discrimination_factors[i] = delay_integration / total_integration
        else:
            discrimination_factors[i] = np.nan

    return discrimination_factors
