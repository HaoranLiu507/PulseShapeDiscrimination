"""
Log-Likelihood Ratio (LLR) method for pulse shape discrimination.

The method computes discrimination factors based on the probability mass 
function (PMF), which approximates the photon detection time probability 
density function (PDF). This PDF typically requires mixing the singlet 
nd triplet components of the pulse shape, convolving with the detector's
resolution time, and incorporating the dark noise component. In contrast 
to these cumbersome processes, directly calculating the PMF from pulse 
amplitudes is both efficient and effective.

References:
- Akashi-Ronquest, M., P-A. Amaudruz, M. Batygov, B. Beltran, M. Bodmer, Mark Guy Boulay, B. Broerman et al.
  "Improving photoelectron counting and particle identification in scintillation detectors with Bayesian techniques."
  Astroparticle physics 65 (2015): 40-54.
- Adhikari, P., R. Ajaj, M. Alpízar-Venegas, P-A. Amaudruz, D. J. Auty, M. Batygov, B. Beltran et al.
  "Pulse-shape discrimination against low-energy Ar-39 beta decays in liquid argon with 4.5 tonne-years of DEAP-3600 data."
  The European Physical Journal C 81 (2021): 1-13.
"""

import numpy as np

from collections import Counter

def calculate_pmf(data):
    """
    Calculate the probability mass function (PMF) for each value in the input signal sequence.
    
    Parameters:
        data (list): Input signal sequence, for example, a list with length 280.
        
    Returns:
        pmf (list): A list with the same length as the input signal, where each element is the probability
                    of the corresponding value at that position.
    """
    # Count the occurrences of each value
    total_samples = len(data)
    value_counts = Counter(data)
    
    # Calculate the probability for each value (count / total samples)
    prob_dict = {value: count / total_samples for value, count in value_counts.items()}
    
    # Generate the corresponding probability sequence based on the original signal values
    pmf = [prob_dict[value] for value in data]
    
    return pmf

def get_psd_factor(pulse_signal: np.ndarray) -> np.ndarray:
    """
    Calculate PSD factors using the Log-Likelihood Ratio method.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse, computed as
                      weighted sums using the Gatti weight function
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

    # Compute log-likelihood ratio using PMFs
    pmf1 = calculate_pmf(class1_signal)
    pmf2 = calculate_pmf(class2_signal)
    epsilon = 1e-10  # Small value to avoid division by zero and log(0)
    P2 = -(np.log((np.array(pmf1) + epsilon) / (np.array(pmf2) + epsilon)))

    # Compute the weighted sum (dot product) for each pulse signal as the discrimination factor
    for i in range(num_signals):
        discrimination_factors[i] = np.dot(pulse_signal[i, :], P2)

    return discrimination_factors

