"""
Zero Crossing (ZC) method for pulse shape discrimination.

The method applies a recursive filter to transform pulses into bipolar signals, 
then computes discrimination factors based on the time interval between pulse 
start and zero-crossing point.

Reference:
- Sperr, P., H. Spieler, M. R. Maier, and D. Evers.
  "A simple pulse-shape discrimination circuit."
  Nuclear Instruments and Methods, vol. 116, no. 1, pp. 55-59, 1974,
  doi: 10.1016/0029-554X(74)90578-3.
- Pai, S., W. F. Piel, D. B. Fossan, and M. R. Maier.
  "A versatile electronic pulse-shape discriminator."
  Nuclear Instruments and Methods in Physics Research, Section A, vol. 278,
  no. 3, pp. 749-754, 1989, doi: 10.1016/0168-9002(89)91199-6.
"""

import numpy as np
from typing import Union, List, Tuple

def get_psd_factor(
    pulse_signal: np.ndarray,
    T: float = 1e-10,
    constant: float = 7e-9
) -> np.ndarray:
    """
    Calculate PSD factors using the Zero Crossing method.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
        T: Time step for the recursive filter (default: 1e-10)
        constant: Time constant for the recursive filter (default: 7e-9)
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse, computed as
                      the time interval between 0.1*peak_pos and the zero
                      crossing point. NaN values indicate no zero crossing found
    
    Note:
        The method applies a third-order recursive filter to transform each pulse
        into a bipolar signal. The filter transfer function is:
        H(s) = (1 - sτ/2)/(1 + sτ/2)³, where τ is the time constant.
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    # Compute the filter coefficient Alpha
    alpha = np.exp(-T / constant)

    # Precompute filter constants for recursive filtering
    coeff1 = T * alpha * (1 - (T / (2 * constant)))
    coeff2 = T * (alpha ** 2) * (1 + (T / (2 * constant)))

    # Process each pulse signal individually
    for i, signal in enumerate(pulse_signal):
        L = signal.shape[0]

        # Pad the signal with 3 zeros at the beginning for filtering
        padded_signal = np.concatenate((np.zeros(3), signal))

        # Initialize arrays for the processed signal
        processed_signal = np.zeros(L + 3)
        data_processed_signal = np.zeros(L + 3)

        # Apply the recursive filtering
        for n in range(3, L + 3):
            processed_signal[n] = (3 * alpha * processed_signal[n - 1] -
                                   3 * (alpha ** 2) * processed_signal[n - 2] +
                                   (alpha ** 3) * processed_signal[n - 3] +
                                   coeff1 * padded_signal[n - 1] -
                                   coeff2 * padded_signal[n - 2])
            data_processed_signal[n] = processed_signal[n]

        # Find the index of the maximum value in the processed signal
        max_idx = np.argmax(data_processed_signal)
        matlab_maxpos = max_idx + 1

        # Find the zero-crossing point after the maximum position
        stop_point = np.nan
        for j in range(max_idx, L + 3):
            if data_processed_signal[j] < 0:
                stop_point = j
                break

        # Compute the discrimination factor based on the zero-crossing point
        if np.isnan(stop_point):
            discrimination_factors[i] = np.nan
        else:
            matlab_stop = stop_point + 1
            start_point = round(matlab_maxpos * 0.1)
            discrimination_factors[i] = matlab_stop - start_point

    return discrimination_factors

