"""
Discrete Fourier Transform (DFT) method for pulse shape discrimination.

This module implements a frequency-domain approach to pulse shape discrimination
by analyzing the zero-frequency components of discrete cosine and sine transforms.
The discrimination factor is computed as a ratio of DFT components normalized by
the signal sum.

Reference:
- Safari, M. J., F. Abbasi Davani, H. Afarideh, S. Jamili, and E. Bayat.
  "Discrete fourier transform method for discrimination of digital scintillation pulses in mixed neutron-gamma fields."
  IEEE transactions on nuclear science 63, no. 1 (2016): 325-332.
"""

import numpy as np
from scipy.fft import dct, dst
from numpy.fft import fft


def get_psd_factor(pulse_signal: np.ndarray) -> np.ndarray:
    """
    Performs pulse shape discrimination based on Discrete Fourier Transform (DFT).

    This method transforms pulse signals into the frequency domain and computes
    the zero-frequency components of the discrete cosine and sine transforms.
    The discrimination factor is defined as:

        (sum(DFT^2) / (DST[0] * DCT[0])) * (1 / sum(trimmed_signal))

    where each pulse signal is trimmed from its maximum (peak) position onward.

    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal.

    Returns:
        A numpy array of discrimination factors for each pulse signal.
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i in range(num_signals):
        # Extract the pulse signal vector for the current pulse
        signal = pulse_signal[i, :]

        # Identify the maximum value and its position
        max_position = np.argmax(signal)

        # Trim the signal starting from the maximum position
        trimmed_signal = signal[max_position:]

        # Compute the discrete cosine transform (DCT) of the trimmed signal
        dct_values = dct(trimmed_signal, type=2, norm='ortho')

        # Compute the discrete sine transform (DST) of the trimmed signal
        dst_values = dst(trimmed_signal, type=2, norm='ortho')

        # Compute the discrete Fourier transform (DFT) of the trimmed signal
        fft_values = np.abs(fft(trimmed_signal))

        # To avoid division by zero, check the required components and total sum
        if dst_values[0] == 0 or dct_values[0] == 0 or np.sum(trimmed_signal) == 0:
            discrimination_factors[i] = np.nan
        else:
            discrimination_factors[i] = (np.sum(fft_values ** 2) / (dst_values[0] * dct_values[0])) \
                                        * (1 / np.sum(trimmed_signal))

    return discrimination_factors

