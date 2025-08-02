"""
Spiking Cortical Model (SCM) implementation for pulse shape discrimination.

This implementation uses a basic spiking cortical model to process raw pulse signals
and generate ignition maps for discrimination.

Reference:
- Liu, Bing-Qi, Hao-Ran Liu, Lan Chang, Yu-Xin Cheng, Zhuo Zuo, and Peng Li.
  "Discrimination of neutrons and gamma-rays in plastic scintillator based on spiking cortical model."
  Nuclear Engineering and Technology 55, no. 9 (2023): 3359-3366.
"""
import numpy as np
from scipy.signal import convolve2d


def scm(pulse_signal_vector: np.ndarray) -> np.ndarray:
    """
    Process a pulse signal using the Spiking Cortical Model.
    
    Implements a basic SCM with optimized parameters for pulse shape discrimination between
    two particle classes. The model uses a specific connection weight matrix and features 
    membrane potential accumulation and dynamic threshold mechanisms.
    
    Args:
        pulse_signal_vector: 1D numpy array representing a pulse signal
        
    Returns:
        1D numpy array representing the computed ignition map
    """
    # Ensure the pulse signal is a 2D row vector
    pulse_signal_vector = np.atleast_2d(pulse_signal_vector)
    rows, cols = pulse_signal_vector.shape

    # Define the connection weight matrix for SCM
    weight_matrix = np.array([[0.1091, 0.1409, 0.1091],
                              [0.1409, 0.0, 0.1409],
                              [0.1091, 0.1409, 0.1091]])

    # Initialize variables
    output_action_potential = np.zeros((rows, cols))
    U = np.copy(output_action_potential)
    ignition_map = np.copy(output_action_potential)
    E = output_action_potential + 1.0

    # SCM model parameters
    iterations = 50
    membrane_attenuation = 0.8
    threshold_attenuation = 0.704
    absolute_refractory = 18.3

    for _ in range(iterations):
        conv_result = convolve2d(output_action_potential, weight_matrix, mode='same')
        U = membrane_attenuation * U + pulse_signal_vector * conv_result + pulse_signal_vector
        E = threshold_attenuation * E + absolute_refractory * output_action_potential
        X = 1.0 / (1.0 + np.exp(E - U))
        output_action_potential = (X > 0.5).astype(float)
        ignition_map += output_action_potential

    return ignition_map.flatten()


def get_psd_factor(
    pulse_signal: np.ndarray,
    ROI_end: int = 120
) -> np.ndarray:
    """
    Perform pulse shape discrimination using the Spiking Cortical Model.
    
    For each pulse signal, generates an ignition map using SCM and computes
    the discrimination factor as the sum of map values within a region of
    interest (ROI). The model's temporal processing characteristics enable
    effective separation of class 1 and class 2 pulses.
    
    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal
        ROI_end: Number of points after the maximum position defining the ROI (default: 120)
        
    Returns:
        1D numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i in range(num_signals):
        signal_vector = pulse_signal[i, :]

        # Compute the ignition map using SCM
        ignition_map = scm(signal_vector)

        # Find the maximum value position in the pulse signal
        max_position = np.argmax(signal_vector)

        # Ensure we do not exceed the bounds of the ignition map
        end_index = min(max_position + ROI_end, len(ignition_map) - 1)

        # Sum the ignition map values from the maximum position to end_index (inclusive)
        roi_sum = np.sum(ignition_map[max_position:end_index + 1])
        discrimination_factors[i] = roi_sum

    return discrimination_factors
