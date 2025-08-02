"""
Fractal Spectrum (FS) method for pulse shape discrimination.

This module implements a frequency-domain approach to pulse shape discrimination
by analyzing the fractal dimension of pulse signals. The method transforms signals
into the frequency domain and computes discrimination factors based on the slope
of log-transformed power spectra.

Reference:
- Liu, Ming-Zhe, Bing-Qi Liu, Zhuo Zuo, Lei Wang, Gui-Bin Zan, and Xian-Guo Tuo.
  "Toward a fractal spectrum approach for neutron and gamma pulse shape discrimination."
  Chinese Physics C 40, no. 6 (2016): 066201.
"""

import numpy as np
from scipy.signal import periodogram
from typing import Optional, List, Union


def get_psd_factor(
        pulse_signal_original: np.ndarray,
        ROI_range: Optional[List[int]] = None,
        fs: Union[int, float] = 2
) -> np.ndarray:
    """
    Performs pulse shape discrimination based on the Fractal Spectrum (FS).

    FS transforms time-domain pulse signals into frequency-domain signals using the Fourier
    transform and computes the fractal dimension of the pulse as the discrimination factor.

    For each pulse signal:
      - The maximum is located and a region of interest (ROI) is extracted using the
        provided ROI_range. The ROI is defined from (max_index + ROI_range[0]) to
        (max_index + ROI_range[1]) inclusive.
      - A boxcar (rectangular) window is applied.
      - The power spectrum is estimated via the periodogram.
      - Both the power spectrum and the corresponding frequencies are logarithmically
        transformed.
      - A linear regression (excluding the first element) is performed on the log-log data
        to determine the slope.
      - The discrimination factor is computed as (b/a - slope), where a and b are empirical
        constants (a=2, b=5).

    Args:
        pulse_signal_original: 2D numpy array where each row is a pulse signal.
        ROI_range: List of two integers specifying the region-of-interest offsets.
                   Default is [60, 130] if not provided.
        fs: Sample rate in Hz. Default is 2.

    Returns:
        A numpy array of discrimination factors for each pulse signal.
    """
    if ROI_range is None:
        ROI_range = [60, 130]

    # Empirical constants
    a = 2
    b = 5

    num_signals = pulse_signal_original.shape[0]
    discrimination_factors = np.zeros(num_signals)

    for i in range(num_signals):
        pulse_signal = pulse_signal_original[i, :]

        # Find the maximum position in the pulse signal
        max_index = np.argmax(pulse_signal)

        # Extract the region of interest (ROI)
        # MATLAB indices are 1-based; np.argmax returns a 0-based index.
        # To mimic MATLAB: ROI = pulse_signal[max_index + ROI_range[0] : max_index + ROI_range[1] + 1]
        start_index = max_index + ROI_range[0]
        end_index = max_index + ROI_range[1] + 1  # +1 because Python slicing is exclusive at the end.
        pulse_signal_roi = pulse_signal[start_index:end_index]

        region_length = len(pulse_signal_roi)
        if region_length == 0:
            discrimination_factors[i] = np.nan
            continue

        # Create a boxcar (rectangular) window
        window = np.ones(region_length)

        # Estimate the power spectrum using a periodogram
        freq, power_spectrum = periodogram(pulse_signal_roi, window=window, nfft=region_length, fs=fs)

        # Log-transform the power spectrum and frequency.
        # Replace any non-positive values with a very small number to avoid log10(0).
        power_log = np.log10(np.where(power_spectrum <= 0, np.finfo(float).eps, power_spectrum))
        freq_log = np.log10(np.where(freq <= 0, np.finfo(float).eps, freq))

        # Perform linear regression on the log-log data using np.polyfit; we only need the slope.
        slope, _ = np.polyfit(power_log[1:], freq_log[1:], 1)

        # Compute the discrimination factor
        discrimination_factors[i] = (b / a) - slope

    return discrimination_factors
