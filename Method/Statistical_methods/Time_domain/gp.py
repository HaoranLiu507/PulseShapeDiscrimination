"""
Gatti Parameter (GP) method for pulse shape discrimination.

The method computes discrimination factors using a weighted linear operation, 
where weights are derived from reference class 1 and class 2 signals.

References:
- Gatti, E., and F. De Martini.
  "A new linear method of discrimination between elementary particles in scintillation counters."
  In Nuclear Electronics II. Proceedings of the Conference on Nuclear Electronics. V. II. 1962.
"""

import numpy as np

from collections import Counter

def get_psd_factor(pulse_signal: np.ndarray) -> np.ndarray:
    """
    Calculate PSD factors using the Gatti Parameter method.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse, computed as
                      weighted sums using the Gatti weight function
    
    Note:
        The weight function P is computed as (c1-c2)/(c1+c2) where c1 and c2 are the
        reference class 1 and class 2 signals respectively.
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.zeros(num_signals)

    # Load reference signals from files with error handling
    try:
        class1_signal = np.loadtxt('Data/Reference_signal/EJ299_33_AmBe_9414_neutron_ref.txt')
    except FileNotFoundError:
        raise FileNotFoundError("Class 1 reference signal file not found. Please add it to 'Data/Reference_signal/'.")
    
    try:
        class2_signal = np.loadtxt('Data/Reference_signal/EJ299_33_AmBe_9414_gamma_ref.txt')
    except FileNotFoundError:
        raise FileNotFoundError("Class 2 reference signal file not found. Please add it to 'Data/Reference_signal/'.")
    
    # Check if reference signals have the same length as pulse signals
    if len(class1_signal) != pulse_signal.shape[1] or len(class2_signal) != pulse_signal.shape[1]:
        raise ValueError(f"Reference signals length mismatch. Expected length {pulse_signal.shape[1]}, "
                       f"but got Class 1: {len(class1_signal)}, Class 2: {len(class2_signal)}")
    
    # Normalize reference signals to [0,1]
    class1_signal = (class1_signal - np.min(class1_signal)) / (np.max(class1_signal) - np.min(class1_signal))
    class2_signal = (class2_signal - np.min(class2_signal)) / (np.max(class2_signal) - np.min(class2_signal))

    # Compute the Gatti Parameter
    P = (class1_signal - class2_signal) / ((class1_signal + class2_signal) + 1e-10)

    # Compute the weighted sum (dot product) for each pulse signal as the discrimination factor
    for i in range(num_signals):
        discrimination_factors[i] = np.dot(pulse_signal[i, :], P)

    return discrimination_factors

