"""
Frequency Gradient Analysis (FGA) method for pulse shape discrimination.

This module implements a frequency-domain approach to pulse shape discrimination
by analyzing the gradient between the first two frequency components of the signal.
The discrimination factor is computed as a normalized difference between frequency
components scaled by the signal length.

Reference:
- Liu, Guofu, Malcolm J. Joyce, Xiandong Ma, and Michael D. Aspinall.
  "A digital method for the discrimination of neutrons and $\gamma $ rays with organic scintillation detectors using frequency gradient analysis."
  IEEE Transactions on Nuclear Science 57, no. 3 (2010): 1682-1691.
"""

import numpy as np
from typing import Union


def get_psd_factor(pulse_signal: np.ndarray, sample_frequency: Union[int, float] = 1) -> np.ndarray:
    """
    Performs pulse shape discrimination based on Frequency Gradient Analysis (FGA).

    FGA transforms time-domain pulse signals into frequency-domain signals and computes
    the gradient of the frequency values between the first and second frequencies. The
    discrimination factor is defined as:

        discrimination_factor = Length_signal * abs(X_0 - X_1) / sample_frequency

    where for each pulse signal:
      - Length_signal is the number of samples in the pulse
      - X_0 = Length_signal * |mean(pulse_signal)|
      - X_1 = |sum(pulse_signal * cos(2*pi*indices/Length_signal))| - sum(pulse_signal * sin(2*pi*indices/Length_signal))
      - indices are defined from 1 to Length_signal (to mimic MATLAB's 1-based indexing)

    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal
        sample_frequency: The sample frequency of the pulse signals in GSa/s (default: 1)

    Returns:
        A numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.zeros(num_signals)

    for i in range(num_signals):
        pulse = pulse_signal[i, :]
        length_signal = pulse.shape[0]

        # Create 1-based indices to mimic MATLAB
        indices = np.arange(1, length_signal + 1)

        X_0 = length_signal * abs(np.mean(pulse))
        X_1 = abs(np.sum(pulse * np.cos(2 * np.pi * indices / length_signal))) - np.sum(
            pulse * np.sin(2 * np.pi * indices / length_signal))

        # Compute the discrimination factor
        discrimination_factors[i] = length_signal * abs(X_0 - X_1) / sample_frequency

    return discrimination_factors
