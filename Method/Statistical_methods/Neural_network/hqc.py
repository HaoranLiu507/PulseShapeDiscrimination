"""
Heterogeneous Quasi-Continuous Spiking Cortical Model (HQC-SCM) for pulse shape discrimination.

This implementation combines two different quasi-continuous spiking cortical models
(QCSCM_A and QCSCM_B) to process different segments of the pulse signal, providing
improved discrimination performance through heterogeneous processing.

Reference:
- Liu, Runxi, Haoran Liu, Bo Yang, Borui Gu, Zhengtong Yin, and Shan Liu.
  "Heterogeneous quasi-continuous spiking cortical model for pulse shape discrimination."
  Electronics 12, no. 10 (2023): 2234.
"""
import numpy as np
from scipy.signal import convolve2d


def qcs_cm_a(pulse_signal_vector: np.ndarray) -> np.ndarray:
    """
    Process a pulse signal vector using the first quasi‐continuous spiking cortical model (QCSCM_A).
    
    This model is optimized for processing the initial segment of the pulse signal with specific
    parameter settings that enhance early feature extraction.
    
    Args:
        pulse_signal_vector: 1D numpy array representing a pulse signal segment
        
    Returns:
        1D numpy array representing the ignition map for QCSCM_A
    """
    # Ensure the signal is 2D (row vector)
    pulse_signal_vector = np.atleast_2d(pulse_signal_vector)
    rows, cols = pulse_signal_vector.shape

    # Define weight matrix and model parameters
    weight_matrix = np.array([[0, 0, 0],
                              [0.5, 0, 0.5],
                              [0, 0, 0]])
    iterations = 50
    t_step = 0.5
    f = 0.3351
    g = 0.8359
    h = 7.9872

    # Adjust parameters by exponentiation as in MATLAB
    f = f ** t_step
    g = g ** t_step

    # Initialize variables
    Y = np.zeros((rows, cols))
    U = np.zeros((rows, cols))
    ignition_map_A = np.zeros((rows, cols))
    E = np.ones((rows, cols))  # equivalent to Y + 1

    # Loop from 1 to iterations with step t_step
    for _ in np.arange(1, iterations + t_step, t_step):
        conv_result = convolve2d(Y, weight_matrix, mode='same')
        U = f * U + pulse_signal_vector * conv_result + pulse_signal_vector
        E = g * E + h * Y
        X = 1.0 / (1.0 + np.exp(E - U))
        Y = (X > 0.5).astype(float)
        ignition_map_A += Y

    return ignition_map_A.flatten()


def qcs_cm_b(pulse_signal_vector: np.ndarray) -> np.ndarray:
    """
    Process a pulse signal vector using the second quasi‐continuous spiking cortical model (QCSCM_B).
    
    This model is optimized for processing the latter segment of the pulse signal with different
    parameter settings that enhance late feature extraction.
    
    Args:
        pulse_signal_vector: 1D numpy array representing a pulse signal segment
        
    Returns:
        1D numpy array representing the ignition map for QCSCM_B
    """
    # Ensure the signal is 2D (row vector)
    pulse_signal_vector = np.atleast_2d(pulse_signal_vector)
    rows, cols = pulse_signal_vector.shape

    # Define weight matrix and model parameters (different from QCSCM_A)
    weight_matrix = np.array([[0, 0, 0],
                              [0.5, 0, 0.5],
                              [0, 0, 0]])
    iterations = 50
    t_step = 0.5
    f = 0.3389
    g = 0.7831
    h = 8.6316

    # Adjust parameters by exponentiation
    f = f ** t_step
    g = g ** t_step

    # Initialize variables
    Y = np.zeros((rows, cols))
    U = np.zeros((rows, cols))
    ignition_map_B = np.zeros((rows, cols))
    E = np.ones((rows, cols))  # equivalent to Y + 1

    # Loop from 1 to iterations with step t_step
    for _ in np.arange(1, iterations + t_step, t_step):
        conv_result = convolve2d(Y, weight_matrix, mode='same')
        U = f * U + pulse_signal_vector * conv_result + pulse_signal_vector
        E = g * E + h * Y
        X = 1.0 / (1.0 + np.exp(E - U))
        Y = (X > 0.5).astype(float)
        ignition_map_B += Y

    return ignition_map_B.flatten()


def hqc_scm(pulse_signal_vector: np.ndarray, segment_point: int) -> np.ndarray:
    """
    Compute the combined ignition map using both QCSCM models.
    
    Segments the input signal at the specified point and processes each segment with
    its corresponding optimized model (QCSCM_A for early segment, QCSCM_B for late segment).
    
    Args:
        pulse_signal_vector: 1D numpy array representing a pulse signal
        segment_point: Index at which to split the signal into two parts
        
    Returns:
        1D numpy array representing the combined ignition map from both models
    """
    # Divide the signal into two segments
    pulse_signal_vector_A = pulse_signal_vector[:segment_point]
    pulse_signal_vector_B = pulse_signal_vector[segment_point:]

    ignition_map_A = qcs_cm_a(pulse_signal_vector_A)
    ignition_map_B = qcs_cm_b(pulse_signal_vector_B)

    # Combine the two ignition maps by concatenation
    return np.concatenate([ignition_map_A, ignition_map_B])


def get_psd_factor(
    pulse_signal: np.ndarray,
    ROI_end: int = 120
) -> np.ndarray:
    """
    Perform pulse shape discrimination using the Heterogeneous Quasi-Continuous Spiking Cortical Model.
    
    For each pulse signal, computes a segment point based on the mean signal across all pulses,
    then processes each segment with its corresponding QCSCM model. The discrimination factor
    is calculated as the sum of ignition map values in a region of interest (ROI).
    
    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal
        ROI_end: Number of points after the maximum position that define the ROI (default: 120)
        
    Returns:
        1D numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    # Compute the mean signal (across pulses) to determine the segment point
    mean_signal = np.mean(pulse_signal, axis=0)
    max_mean_signal_value = np.max(mean_signal)

    # Find the first index where the mean signal is within 4.91% to 5.01% of the max
    indices = np.where((mean_signal >= 0.0491 * max_mean_signal_value) &
                       (mean_signal <= 0.0501 * max_mean_signal_value))[0]
    segment_point = int(indices[0]) if indices.size > 0 else 0

    # Process each pulse signal
    for i in range(num_signals):
        signal_vector = pulse_signal[i, :]
        ignition_map_combined = hqc_scm(signal_vector, segment_point)
        max_position = np.argmax(signal_vector)
        end_index = min(max_position + ROI_end, ignition_map_combined.size)
        roi_sum = np.sum(ignition_map_combined[max_position:end_index])
        discrimination_factors[i] = roi_sum

    return discrimination_factors
