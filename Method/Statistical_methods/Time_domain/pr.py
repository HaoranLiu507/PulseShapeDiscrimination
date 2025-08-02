"""
Pattern Recognition (PR) method for pulse shape discrimination.

The method computes discrimination factors by measuring the angle between 
each pulse and a reference pulse in vector space, using their post-peak 
portions for comparison.

Reference:
- Takaku, D., T. Oishi, and M. Baba.
  "Development of neutron-gamma discrimination technique using pattern-recognition method with digital signal processing."
  Prog. Nucl. Sci. Technol 1 (2011): 210-213.
"""

import numpy as np


def get_psd_factor(
        pulse_signal: np.ndarray
) -> np.ndarray:
    """
    Calculate PSD factors using Pattern Recognition.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse. The second
                     pulse (index 1) is used as the reference
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse, computed as
                      |arccos(cos θ)|, where θ is the angle between the pulse
                      and reference vectors
    
    Raises:
        ValueError: If N < 2 or if any post-peak signal has zero norm
    
    Note:
        Only the portion of each pulse after its peak is used for comparison.
        The cosine angle is computed using the dot product formula:
        cos θ = (a·b)/(|a|·|b|)
    """
    num_signals = pulse_signal.shape[0]
    if num_signals < 2:
        raise ValueError("pulse_signal must have at least two rows for reference selection.")

    # Initialize discrimination factors array
    discrimination_factors = np.zeros(num_signals)

    # Select the reference signal as the second row (index 1)
    reference_signal = pulse_signal[1, :]

    for i, signal in enumerate(pulse_signal):
        # Find the peak position (index of the maximum value)
        max_position = np.argmax(signal)

        # Trim the signal and the reference signal from the peak position onward
        trimmed_signal = signal[max_position:]
        trimmed_reference = reference_signal[max_position:]

        # Compute the scalar product and the norms of the trimmed vectors
        scalar_product = np.dot(trimmed_signal, trimmed_reference)
        norm_signal = np.linalg.norm(trimmed_signal)
        norm_reference = np.linalg.norm(trimmed_reference)

        # Check that neither norm is zero
        if norm_signal == 0 or norm_reference == 0:
            raise ValueError("Norm of one of the vectors is zero, cannot compute discrimination factor.")

        # Calculate the cosine of the angle between the two vectors
        cos_angle = scalar_product / (norm_signal * norm_reference)

        # Clamp the cosine value to the valid range for arccos
        cos_angle = max(min(cos_angle, 1), -1)

        # The discrimination factor is the absolute value of the angle (in radians)
        discrimination_factors[i] = abs(np.arccos(cos_angle))

    return discrimination_factors
