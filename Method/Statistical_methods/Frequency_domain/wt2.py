"""
Wavelet Transform (WT2) method with Marr wavelet for pulse shape discrimination.

This module implements a wavelet-based approach to pulse shape discrimination
using the Marr (Mexican Hat) wavelet transform. The discrimination factor is
computed as a ratio of signal integrals in different domains.

Reference:
- Langeveld, Willem GJ, Michael J. King, John Kwong, and Daniel T. Wakeford.
  "Pulse shape discrimination algorithms, figures of merit, and gamma-rejection for liquid and solid scintillators."
  IEEE Transactions on Nuclear Science 64, no. 7 (2017): 1801-1809.
"""

import numpy as np
from scipy.signal import convolve
from numpy import trapz

def marr_wavelet(t: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute the Marr (Mexican Hat) wavelet function values.

    The Marr wavelet is defined as:
        ψ(t) = (1 - t^2/sigma^2) * exp(-t^2/(2*sigma^2))

    Args:
        t: 1D numpy array of time values
        sigma: Standard deviation of the wavelet (default: 1.0)

    Returns:
        1D numpy array of Marr wavelet values
    """
    return (1 - (t ** 2) / (sigma ** 2)) * np.exp(-t ** 2 / (2 * sigma ** 2))


def get_psd_factor(
    pulse_signal: np.ndarray,
    sigma: float = np.sqrt(5),
    wavelet_length: int = 100,
    dt: float = 1.0
) -> np.ndarray:
    """
    Performs pulse shape discrimination based on the Wavelet Transform (WT).

    WT2 computes the ratio of the integral of the signal in the frequency domain obtained through 
    the Marr wavelet to the integral that best reflects the difference in signal characteristics, 
    which serves as the discrimination factor.

    Args:
        pulse_signal: A 2D numpy array where each row corresponds to a pulse signal
        sigma: Standard deviation for the Marr wavelet (default: sqrt(5))
        wavelet_length: Half-length of the wavelet; the wavelet is computed over 
                       [-wavelet_length, wavelet_length] (default: 100)
        dt: Time interval between samples (default: 1.0)

    Returns:
        A numpy array of discrimination factors
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.zeros(num_signals)  # Initialize the array to store PSD values

    # Define time vector for the wavelet and compute the Marr wavelet
    t_wavelet = np.linspace(-wavelet_length, wavelet_length, 2 * wavelet_length + 1)
    wavelet = marr_wavelet(t_wavelet, sigma)

    for i in range(num_signals):
        # Extract the current pulse signal
        signal = pulse_signal[i, :]

        # Create a time vector for integration (assuming dt spacing)
        time_vector = np.arange(0, len(signal)) * dt

        # Compute the integral of the original signal
        integral_signal = trapz(signal, time_vector)

        # Perform convolution of the signal with the Marr wavelet
        convolved_signal = convolve(signal, wavelet, mode='same')

        # Compute the integral of the positive part of the wavelet-transformed signal
        positive_convolved = np.maximum(0, convolved_signal)
        integral_wavelet = trapz(positive_convolved, time_vector)

        # Calculate the PSD value and store it in the array
        discrimination_factors[i] = 2 * integral_signal / (integral_signal + integral_wavelet)

    return discrimination_factors
